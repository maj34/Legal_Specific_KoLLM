# KoLLM: Korean-based Legal Language Model

본 프로젝트는 법률 도메인 특화 한국어 기반 질의응답 시스템 (LLM) 개발 프로젝트입니다.

## 설치 및 실행

### 환경 요구사항
- Python 3.8
- CUDA 11.7
- PyTorch 2.0.1
- Transformers 4.35.2

### 의존성 설치
```bash
pip install -r docker/requirements.txt
```

### Docker 환경
```bash
docker build -t kollm .
docker run --gpus all -it kollm
```

## 사용법

### 1. 데이터 수집
```bash
# 법률 기사 크롤링
python src/prepare_data/law_article_crawling.py

# AHA 데이터 크롤링
python src/prepare_data/AHA_Crawling.py
```

### 2. 모델 파인튜닝
```bash
# 기본 파인튜닝
python src/Fine_Tuning/fine_tuning.py --config_file configs/eeve.yaml

# 추가 학습
python src/Fine_Tuning/further_training_fine_tuning.py --config_file configs/eeve.yaml
```

### 3. RAG 시스템 실행
```bash
# 벡터 저장소 생성
python src/RAG/save_vectorstore.py

# RAG 평가 실행
python src/RAG/RAG.py
```

### 4. DPR 모델 학습
```bash
python src/dpr.py
```

## 설정 파일

### Fine-Tuning 설정 (configs/eeve.yaml)
- 모델: yanolja/EEVE-Korean-Instruct-10.8B-v1.0
- LoRA 설정: r=4, alpha=32, dropout=0.05
- 학습률: 0.0001, 배치 크기: 16

### RAG 설정 (configs/default.yaml)
- 임베딩 모델: jhgan/ko-sbert-nli
- 청크 크기: 300, 최대 문서 수: 3
- 검색 타입: similarity


## 프로젝트 구조

```
src/
├── Fine_Tuning/                 # 모델 파인튜닝 관련 코드
│   ├── configs/                 # 파인튜닝 설정 파일
│   ├── fine_tuning.py          # 기본 파인튜닝 스크립트
│   ├── further_training_fine_tuning.py  # 추가 학습 스크립트
│   └── scripts/                # 학습 실행 스크립트
├── RAG/                        # RAG 시스템 구현
│   ├── configs/                # RAG 설정 파일
│   ├── datastore.py           # 데이터 저장소 관리
│   ├── qa_module.py           # 질문-답변 모듈
│   ├── RAG.py                 # RAG 메인 실행 파일
│   ├── evaluation.py          # 평가 모듈
│   ├── save_vectorstore.py    # 벡터 저장소 생성
│   └── utils.py               # 유틸리티 함수
├── prepare_data/              # 데이터 수집 및 전처리
│   ├── law_article_crawling.py    # 법률 기사 크롤링
│   ├── AHA_Crawling.py            # AHA 데이터 크롤링
│   └── prompt/                    # 프롬프트 템플릿
├── dpr.py                     # DPR 모델 구현
├── evaluation.py              # 평가 함수
├── packed_dataset.py          # 데이터 패킹 구현
├── assert_packing_loss.py     # 패킹 손실 검증
└── monkey_patch_packing.py    # 패킹 최적화 패치
```

## 주요 기능

### 1. 데이터 수집 및 전처리
- 법률 기사 크롤링: 법률 관련 기사 수집
- AHA 법률 상담 데이터: AHA 플랫폼에서 법률 상담 질문-답변 데이터 수집
- 데이터 전처리: 수집된 데이터의 정제 및 구조화

### 2. 모델 Fine-Tuning
- Supervised Fine-Tuning: 다양한 한국어 LLM 모델에 대한 법률 도메인 특화 Fine-Tuning
  - 4비트 양자화와 LoRA를 활용한 메모리 효율적 학습
- 지원 모델: EEVE-Korean, KULLM3, Llama3-Instruct 등

### 3. RAG (Retrieval-Augmented Generation)
- 문서 검색: FAISS 벡터 검색과 BM25 키워드 검색 지원
- 문서 임베딩: 한국어 특화 임베딩 모델 활용
- 생성 모델: 검색된 법률 문서를 기반으로 한 답변 생성

### 4. DPR (Dense Passage Retrieval)
- 질문-문서 매칭: 질문과 법률 문서 간의 의미적 유사도 학습
- 이중 인코더 구조: 질문 인코더와 문서 인코더를 통한 효율적 검색

### 5. 평가 시스템
- 다중 평가 지표: BLEU-1, ROUGE-L F1, METEOR 점수 계산
- 한국어 특화 평가: Mecab 형태소 분석기를 활용한 정확한 평가
