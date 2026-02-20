![result](./ret.png)

## 프로젝트 구조

```
ai-document-archive/
├── app.py                # 메인 애플리케이션
├── requirements.txt      # 패키지 목록
├── archive.db           # SQLite 데이터베이스 (자동 생성)
└── test-image.jpg       # 테스트용 샘플 이미지
```

## 프로젝트 시작

```bash
# 1. 가상환경 버전1
conda env create -f environment.yml

# 2. 애플리케이션 실행
streamlit run app.py
```

```bash
# [참고] 가상환경 버전2
# 환경 이름: app, python 버전: 3.10
conda env create -n app python=3.10 -y

# 환경 활성화
conda activate app

# 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

1. http://localhost:8501 접속
2. **문서 업로드 탭**
   - PNG, JPG, JPEG 형식의 문서 이미지 업로드
   - 자동으로 문서 유형 분류 및 정보 추출
   - "저장" 버튼으로 데이터베이스에 보관
3. **문서 검색 탭**
   - 벡터 유사도 검색 또는 키워드 검색 선택
   - 검색어 입력 (예: "영수증", "계약서")
   - 검색 결과에서 이미지 미리보기 및 다운로드
4. **문서 목록 탭**
   - 저장된 모든 문서 목록 확인
   - 각 문서의 상세 정보 열람

## 주요 기능

- DiT 기반 문서 유형 자동 분류
- PaddleOCR 한국어 텍스트 추출 및 위치 검출
- Donut 모델 영수증 정보 자동 구조화
- LayoutLMv3 문서 레이아웃 분석 및 정보 추출
- Ko-SRoBERTa 벡터 임베딩 기반 의미 검색
- KoBART 한국어 문서 자동 요약
- SQLModel + SQLite 문서 메타데이터 관리
- Streamlit 웹 기반 사용자 인터페이스

# 배운점

1. 이미지 전처리 </br>
   - 이미지 전처리 파이프라인 최적화 (Advanced Pre-processing)
   - OCR 데이터 추출 정밀도 향상 (Sliding Window 도입)
   - 불용어 필터링
   - 정규표현식

   ### 이미지 전처리 결론

   "물리적 이미지 한계를 슬라이딩 윈도우 알고리즘으로 극복하고, 슬라이딩 윈도우를 통해 영수증 데이터의 정교함을 확보함."

2. Dit 기반 라우팅
3. Debug Print
