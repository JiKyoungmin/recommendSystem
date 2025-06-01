#!/bin/bash
echo "🚀 한끼모아 추천 시스템 초기 설정..."

# # Python 가상환경 생성
# echo "📦 가상환경 생성 중..."
# python3 -m venv venv

# 가상환경 활성화
echo "🔧 가상환경 활성화..."
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 필요한 디렉토리 생성
mkdir -p data result

# Python 패키지 설치
echo "📥 패키지 설치 중..."
pip install -r requirements.txt

echo "✅ 설정 완료!"
echo "📁 data/ 폴더에 rating.xlsx와 restaurant.json 파일을 업로드하세요."
echo "🚀 다음 명령어로 데이터 전처리를 진행하세요:"
echo "   source venv/bin/activate && ./setup_data.sh"