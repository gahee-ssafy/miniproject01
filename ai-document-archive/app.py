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
os.environ['DNNL_MAX_CPU_ISA'] = 'AVX2'

# ---------------------------------------------------------
# 1. DB: 없으면 만들고, 있으면 놔둬라
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
kiwi = Kiwi() # Q1 이거 왜 하지? 
# 키위는 "한국어" 형태소 분석기입니다. 
# 영수증이나 문서에서 명사 키워드를 추출할 때 사용됩니다. 
# 예를 들어, "삼성전자 갤럭시 S21 128GB"라는 텍스트가 있으면, 키위는 "삼성전자", "갤럭시", "S21", "128GB" 같은 명사들을 추출해줍니다. 
# 이렇게 추출된 키워드들은 검색이나 분류에 활용될 수 있습니다.
# Q2 왜 처음에 해야하는데? 
# 키위 객체를 미리 생성해두면, 이후에 형태소 분석이 필요할 때마다 빠르게 사용할 수 있습니다. 

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
# 영수증 추출
def extract_receipt_info(text):    
    # 1. 규격 데이터 추출 - REGEX(찾기)
    # 사업자 번호 추출
    biz_num_match = re.search(r'\d{3}[-\s]?\d{2}[-\s]?\d{5}', text)
    # 날짜 
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    # 금액
    total_price_match = re.search(r'(?:합\s*계|결제금액|총액)\s*[:\s]*([\d\s,]+)', text)
    
    res = []
    if biz_num_match: res.append(f"🏢 사업자 등록번호: {biz_num_match.group()}")
    if date_match: res.append(f"📅 날짜: {date_match.group()}")
    if total_price_match:
        price = total_price_match.group(1).replace(" ", "").replace(",", "").strip()
        res.append(f"💰 총합계: {int(price):,}원")
    
    
    # 2. 비정형 데이터 추출 - 문맥활용
    lines = text.split('\n')
    valid_items = []
    
    # TODO: 불용어 필터링 - [1차] DEBUG해서 나오는 노이즈
    exclude_keywords = []

    # [2차] 품목이 시작되는 지점 탐색 - [1차] 노이즈가 너무 많아! 
    start_collecting = False
    for line in lines:
        # '상품코드'나 '금액'이라는 단어가 보이면 그 다음 줄부터 진짜 품목으로 간주
        if any(k in line for k in ['상품코드', '단가', '수량']):
            start_collecting = True
            continue
        
        # '합계'나 '부가세'가 나오면 품목 섹션이 끝난 것으로 간주
        if any(k in line for k in ['합계', '부가세', '과세']):
            start_collecting = False 
            continue

        if start_collecting and re.search(r'[가-힣]+', line):
            # 불용어 필터링
            if any(key in line for key in exclude_keywords):
                continue
            
            # 숫자/특수문자 제거
            clean_name = re.sub(r'[0-9*#\-\.\[\]\{\}\<\>]', '', line).strip()
            clean_name = re.sub(r'\s+', ' ', clean_name)
            
            if len(clean_name) > 1:
                valid_items.append(clean_name)

    if valid_items:
        valid_items = list(dict.fromkeys(valid_items))
        res.append(f"🛒 품목: {valid_items[0]} 등 {len(valid_items)}건")
        
    return " | ".join(res) if res else "정보 추출 실패"


# 사진 추출
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
# pipeline: 이미지 전처리 -> OCR 추출 -> 텍스트 분석
# ---------------------------------------------------------
# 이미지 전처리 및 OCR
def get_ocr_text(img, ocr_model, is_receipt=False):
    """이미지에서 텍스트를 정밀하게 추출합니다."""
    # [기본 전처리] 여백 -> 확대 -> 흑백/이진화
    img_padded = cv2.copyMakeBorder(img, 40, 40, 100, 40, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    h, w = img_padded.shape[:2]
    img_up = cv2.resize(img_padded, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if is_receipt:
        """ 슬라이딩 윈도우 알고리즘 적용 (영수증 한정)
        장점: 잘라서 봐야 세세하게 보입니다. 
        단점: 시간소요가 늘어납니다."""

        ph, pw = processed_img.shape[:2]
        win_h, overlap, texts = ph // 3, 100, []
        for i in range(3):
            start_y, end_y = max(0, i * win_h - overlap), min(ph, (i + 1) * win_h + overlap)
            res = ocr_model.ocr(processed_img[start_y:end_y, :], cls=True)
            if res and res[0]:
                for line in res[0]:
                    if line[1][0] not in texts: texts.append(line[1][0])
        return "\n".join(texts), processed_img
    else:
        # 일반 모드
        res = ocr_model.ocr(processed_img, cls=True)
        text = "\n".join([l[1][0] for l in res[0]]) if res and res[0] else ""
        return text, processed_img

# 메인 프로세스 함수 
def process_document(uploaded_file, models):
    (dit_p, dit_m, ocr, obj_p, obj_m, sum_t, sum_m, emb_m) = models
    raw_img = Image.open(io.BytesIO(uploaded_file.read()))
    orig_img = raw_img.convert("RGB")
    
    # 1. 분류
    inputs = dit_p(images=orig_img, return_tensors="pt")
    label = dit_m.config.id2label[dit_m(**inputs).logits.argmax(-1).item()].lower()
    is_receipt = any(x in label for x in ['receipt', 'invoice'])

    # 2. OCR (전담 함수 호출)
    img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    full_text, processed_img = get_ocr_text(img_cv, ocr, is_receipt)

    # 3. 문서 vs 사진 판별 및 후속 처리
    is_doc = is_receipt or any(x in label for x in ['form', 'letter']) or len(full_text) > 40
    
    if is_doc:
        doc_type, structured_data = "Document", {}
        receipt_summary = extract_receipt_info(full_text)
        
        if is_receipt and receipt_summary:
            final_summary = f"🧾 [영수증] {receipt_summary}"
        else:
            try:
                s_in = sum_t([full_text[:500]], max_length=128, return_tensors="pt", truncation=True)
                s_ids = sum_m.generate(s_in["input_ids"], num_beams=4, max_length=128)
                final_summary = sum_t.decode(s_ids[0], skip_special_tokens=True).strip()
            except: final_summary = f"{full_text[:30]}..."
        
        final_keywords = ", ".join(list(dict.fromkeys([t.form for t in kiwi.tokenize(full_text) if t.tag in ['NNG', 'NNP']]))[:10])
    else:
        doc_type = "Photo"
        processed_img = np.array(orig_img) # 사진은 원본 반환
        meta = extract_photo_metadata(raw_img)
        # 객체 탐지 로직 (기존과 동일) ...
        final_summary = f"📸 [{meta['taken_date']}] 촬영 사진" # 예시 요약
        final_keywords = "사진, 객체" # 예시 키워드
        structured_data = {'exif': meta}

    embedding = emb_m.encode(full_text + " " + final_keywords).tolist()
    return (doc_type, full_text, final_summary, final_keywords, structured_data, uploaded_file.getvalue(), embedding, processed_img)



# ---------------------------------------------------------
# UI 
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