#!/bin/bash
echo "🚀 한끼모아 추천 시스템 초기 설정..."

# 기존 가상환경 확인
if [ -d "venv" ]; then
    echo "📦 기존 가상환경이 발견되었습니다."
    read -p "기존 가상환경을 사용하시겠습니까? (y/n): " use_existing
    
    if [ "$use_existing" = "n" ] || [ "$use_existing" = "N" ]; then
        echo "🗑️  기존 가상환경 삭제 중..."
        rm -rf venv
        echo "📦 새 가상환경 생성 중..."
        python3 -m venv venv
    else
        echo "📦 기존 가상환경 사용"
    fi
else
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
echo "🔧 가상환경 활성화..."
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 필요한 디렉토리 생성
mkdir -p data result

# Python 패키지 설치/업데이트
echo "📥 패키지 설치/업데이트 중..."
pip install -r requirements.txt

echo "✅ 설정 완료!"
echo "📁 data/ 폴더에 rating.xlsx와 restaurant.json 파일을 업로드하세요."
echo "🚀 다음 명령어로 데이터 전처리를 진행하세요:"
echo "   source venv/bin/activate && ./setup_data.sh"