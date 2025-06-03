#!/bin/bash
echo "🚀 서버 배포 중..."

# 기존 서버 종료
echo "기존 서버 종료 중..."
pkill -f api_server.py

# 최신 코드 받기
echo "최신 코드 받는 중..."
git pull origin main

# 가상환경 활성화
source venv/bin/activate

# 패키지 업데이트
echo "패키지 업데이트 중..."
pip install -r requirements.txt

# 서버 실행
echo "서버 시작 중..."
nohup python api_server.py > server.log 2>&1 &

echo "✅ 배포 완료! 포트 5000에서 실행 중"