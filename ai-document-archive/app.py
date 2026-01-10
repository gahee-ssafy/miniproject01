import os
import io
import re
import json
import numpy as np
import cv2
import streamlit as st
import torch
import folium
from typing import Optional
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

# AI 모델 관련 라이브러리
from paddleocr import PaddleOCR
from sqlmodel import Field, Session, SQLModel, create_engine, select
from transformers import (
    AutoProcessor, AutoModelForImageClassification, 
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DetrImageProcessor, DetrForObjectDetection
)
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi

# 환경 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# [DEBUG] 하드웨어 가속 라이브러리의 충돌을 방지하는 설정
os.environ['DNNL_MAX_CPU_ISA'] = 'AVX2'

# ---------------------------------------------------------
# 1. DB 모델 및 초기화
# ---------------------------------------------------------
class Document(SQLModel, table=True):
    __table_args__ = {"extend_existing": True} 
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    doc_type: str 
    content: str 
    summary: str
    keywords: str
    structured_data: str 
    upload_date: datetime = Field(default_factory=datetime.now)
    image_data: bytes
    embedding: Optional[str] = None

engine = create_engine("sqlite:///archive.db")
SQLModel.metadata.create_all(engine)
kiwi = Kiwi()

# ---------------------------------------------------------
# 2. AI 모델 로딩 (캐싱)
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    ocr = PaddleOCR(lang='korean', show_log=False)
    dit_p = AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    dit_m = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    obj_p = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    obj_m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    sum_t = AutoTokenizer.from_pretrained("gogamza/kobart-summarization")
    sum_m = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-summarization")
    emb_m = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return (dit_p, dit_m, ocr, obj_p, obj_m, sum_t, sum_m, emb_m)

# ---------------------------------------------------------
# 3. 보조 분석 함수 (정규표현식 영수증 추출 추가)
# ---------------------------------------------------------

def extract_receipt_info(text):
    """영수증 텍스트에서 정규표현식을 통해 정형화된 정보를 추출합니다."""
    # 사업자 번호 추출
    biz_num_match = re.search(r'\d{3}[ -]?\d{2}[ -]?\d{5}', text)
    # 날짜 
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    # 금액
    total_price_match = re.search(r'(?:합\s*계|결제금액|총액)\s*[:\s]*([\d\s,]+)', text)
    # 품목 
    item_pattern = r'(\d{2,})?\s*([가-힣A-Z\(\)\[\]][가-힣A-Z0-9\(\)\[\]\-~ ]+?)(?=\s+\d+)'
    items = re.findall(item_pattern, text)
    
    # --- 출력 확인 구간 (추가된 부분) ---
    # print(f"\n[DEBUG] 1. 정규식 추출 결과 (items): {items}") 
    # -------------------------------

    res = []
    if biz_num_match: res.append(f"🏢 사업자 등록번호: {biz_num_match.group()}")
    print(f"\n[DEBUG] 사업자: {biz_num_match.group()}") 
    if date_match: res.append(f"📅 날짜: {date_match.group()}")
    if total_price_match:
        price = total_price_match.group(1).replace(" ", "").replace(",", "").strip()
        res.append(f"💰 총합계: {int(price):,}원")
    
    if items:
        valid_items = []
        # 1. 불용어 리스트 대폭 강화 (OCR 오타 대응)
        stopwords = [
        # 결제 관련
        '물품가액', '과세', '부가세', '부가서', '상품가격', '합계', '금액', '수량', '단가',
        # 점포/주소 관련 (이번에 추가!)
        '이마트', 'KMART', '대한민국', '고양시', '덕이동', '주소', '대표자', '전화',
        # 안내 문구 관련 (이번에 추가!)
        '환불', '환물', '교환', '편리', '등록', '영수증', '문의', '감사'
    ]
        
        for it in items:
            raw_name = it[1].strip()
            
            # [핵심 로직] 공백을 제거한 상태에서 비교합니다.
            # '합 계' -> '합계'로 변환해서 체크하니까 훨씬 잘 걸려요!
            clean_check_name = raw_name.replace(" ", "")
            
            # 불용어 중 하나라도 포함되어 있으면 패스!
            if any(stop.replace(" ", "") in clean_check_name for stop in stopwords):
                continue
            
            valid_items.append(raw_name)
        
        # 중복 제거 (set 활용)
        valid_items = list(dict.fromkeys(valid_items))

        if valid_items:
            item_str = f"🛒 품목: {valid_items[0]} 등 {len(valid_items)}건"
            res.append(item_str)
            print(f"[DEBUG] 최종 정제된 품목들: {valid_items}")
            
    return " | ".join(res) if res else "정보 추출 실패"


def extract_photo_metadata(image):
    metadata = {'width': image.width, 'height': image.height, 'camera_model': '정보 없음', 'taken_date': '정보 없음', 'location_address': '정보 없음', 'lat': None, 'lng': None}
    try:
        exif_data = image._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "Model": metadata['camera_model'] = str(value).strip()
                elif tag in ["DateTime", "DateTimeOriginal"]: metadata['taken_date'] = str(value).replace(':', '-', 2)
                elif tag == "GPSInfo" and isinstance(value, dict):
                    gps_data = {GPSTAGS.get(t, t): value[t] for t in value}
                    if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                        def to_decimal(dms, ref):
                            d, m, s = [float(x) for x in dms]
                            res = d + m/60.0 + s/3600.0
                            return -res if ref in ['S', 'W'] else res
                        metadata['lat'] = to_decimal(gps_data['GPSLatitude'], gps_data['GPSLatitudeRef'])
                        metadata['lng'] = to_decimal(gps_data['GPSLongitude'], gps_data['GPSLongitudeRef'])
                        try:
                            geolocator = Nominatim(user_agent="geo_archive_v4")
                            loc = geolocator.reverse(f"{metadata['lat']}, {metadata['lng']}", language='ko')
                            if loc: metadata['location_address'] = loc.address
                        except: pass
    except: pass
    return metadata

# ---------------------------------------------------------
# 4. 메인 프로세싱 함수 (수정됨)
# ---------------------------------------------------------
def process_document(uploaded_file, models):
    (dit_p, dit_m, ocr, obj_p, obj_m, sum_t, sum_m, emb_m) = models
    file_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(file_bytes))
    orig_img = raw_img.convert("RGB")
    
    # 1. 문서 분류
    inputs = dit_p(images=orig_img, return_tensors="pt")
    label = dit_m.config.id2label[dit_m(**inputs).logits.argmax(-1).item()].lower()
    
    # 2. OCR 수행 (분류 및 텍스트 확보)
    img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    ocr_res = ocr.ocr(img_cv, cls=False)
    full_text = " ".join([line[1][0] for line in ocr_res[0]]) if ocr_res and ocr_res[0] else ""

    # 문서/사진 판별
    is_doc = any(x in label for x in ['receipt', 'invoice', 'form', 'letter']) or len(full_text) > 40
    
    if is_doc:
        doc_type = "Document"
        # 이미지 전처리 (이진화 등)
        # 안녕하세요? 진화씨? 
        height, width = img_cv.shape[:2]
        img_cv_up = cv2.resize(img_cv, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(img_cv_up, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. 요약 로직 분기 (영수증 vs 일반 문서)
        receipt_summary = extract_receipt_info(full_text)
        
        if ('receipt' in label or 'invoice' in label) and receipt_summary:
            # 영수증이면 정규표현식으로 정형화된 요약 사용
            final_summary = f"🧾 [영수증] {receipt_summary}"
        else:
            # 일반 문서면 KoBART AI 요약 사용
            try:
                s_inputs = sum_t([full_text], max_length=128, return_tensors="pt", truncation=True)
                s_ids = sum_m.generate(s_inputs["input_ids"], num_beams=4, max_length=128)
                final_summary = sum_t.decode(s_ids[0], skip_special_tokens=True).strip()
            except:
                final_summary = f"{full_text[:30]}..."

        kw_list = [t.form for t in kiwi.tokenize(full_text) if t.tag in ['NNG', 'NNP']]
        final_keywords = ", ".join(list(dict.fromkeys(kw_list))[:10])
        structured_data = {}
    else:
        doc_type = "Photo"
        processed_img = np.array(orig_img)
        meta = extract_photo_metadata(raw_img)
        # 객체 탐지
        obj_inputs = obj_p(images=orig_img, return_tensors="pt")
        obj_outputs = obj_m(**obj_inputs)
        target_sizes = torch.tensor([orig_img.size[::-1]])
        results = obj_p.post_process_object_detection(obj_outputs, target_sizes=target_sizes, threshold=0.7)[0]
        objs = list(set([obj_m.config.id2label[l.item()] for l in results["labels"]]))
        
        final_keywords = generate_photo_keywords(meta, objs)
        final_summary = f"📸 [{meta['taken_date']}] 촬영 사진. 탐지: {', '.join(objs)}"
        structured_data = {'exif': meta, 'objects': objs}

    embedding = emb_m.encode(full_text + " " + final_keywords).tolist()
    return (doc_type, full_text, final_summary, final_keywords, structured_data, file_bytes, embedding, processed_img)

# ---------------------------------------------------------
# 5. UI (기존과 동일)
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Multi-Archive")
st.title("🌟 멀티모달 AI 통합 아카이브")

models = load_all_models()
t1, t2, t3, t4 = st.tabs(["📤 업로드", "🔍 검색", "📁 아카이브", "📍 지도"])

with t1:
    file = st.file_uploader("이미지 업로드", type=['jpg', 'png', 'jpeg'])
    if file:
        if "res" not in st.session_state or st.session_state.get("fname") != file.name:
            with st.spinner("분석 중..."):
                st.session_state.res = process_document(file, models)
                st.session_state.fname = file.name
        
        r = st.session_state.res
        col1, col2 = st.columns(2)
        col1.image(r[5], caption="원본")
        col2.image(r[7], caption="OCR 전처리 결과")
        
        st.write(f"**분류:** {r[0]} | **키워드:** `{r[3]}`")
        st.info(f"**요약:** {r[2]}")
        
        if st.button("🚀 최종 저장", type="primary"):
            with Session(engine) as session:
                new_doc = Document(filename=file.name, doc_type=r[0], content=r[1], 
                                   summary=r[2], keywords=r[3], 
                                   structured_data=json.dumps(r[4], ensure_ascii=False),
                                   image_data=r[5], embedding=json.dumps(r[6]))
                session.add(new_doc); session.commit()
            st.success("저장 완료!")

with t2:
    q = st.text_input("검색어 (객체, 장소, 내용 등)")
    if q:
        with Session(engine) as session:
            results = session.exec(select(Document).where((Document.content.contains(q)) | (Document.keywords.contains(q)))).all()
            for d in results:
                with st.expander(f"📄 {d.filename} ({d.doc_type})"):
                    sc1, sc2 = st.columns([1, 3])
                    sc1.image(d.image_data)
                    sc2.write(f"**요약:** {d.summary}")
                    sc2.write(f"**키워드:** `{d.keywords}`")

with t3:
    with Session(engine) as session:
        items = session.exec(select(Document).order_by(Document.upload_date.desc())).all()
        for item in items:
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                c1.image(item.image_data)
                c2.write(f"**{item.filename}** ({item.doc_type})")
                c2.caption(f"요약: {item.summary} | 키워드: {item.keywords}")
                if st.button("🗑️ 삭제", key=f"del_{item.id}"):
                    session.delete(item); session.commit(); st.rerun()

with t4:
    st.header("📍 사진 촬영 위치")
    with Session(engine) as session:
        # 오류 해결: st.all_docs가 아니라 변수에 데이터를 담아 함수에 전달해야 함
        all_docs = session.exec(select(Document)).all()
        if all_docs:
            # display_photo_locations 함수를 호출 (all_docs 인자 전달)
            # (해당 함수 내에서 lat/lng 추출 로직이 d.structured_data를 파싱하도록 되어 있는지 확인 필요)
            st.info(f"현재 {len(all_docs)}개의 데이터가 아카이브에 있습니다.")