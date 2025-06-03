#!/bin/bash
echo "📊 데이터 전처리 시작..."

# 가상환경 활성화
source venv/bin/activate

cd pipeline

# 전처리 실행
echo "1. 전처리 중..."
python preprocessing.py

# SVD++ 모델 학습
echo "2. SVD++ 모델 학습 중..."
python svdpp.py

# 콘텐츠 기반 매트릭스 생성
echo "3. 콘텐츠 기반 매트릭스 생성 중..."
python contentRecommender.py

cd ..
echo "✅ 데이터 전처리 완료!"