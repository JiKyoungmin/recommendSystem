#!/bin/bash
echo "🚀 추천 서버 시작..."

# 가상환경 활성화
source venv/bin/activate

# 기존 서버가 실행 중이면 종료
if pgrep -f "api_server.py" > /dev/null; then
    echo "기존 서버 종료 중..."
    pkill -f api_server.py
    sleep 2
fi

# 서버 시작
echo "서버 시작 중..."
nohup python api_server.py > server.log 2>&1 &

# 서버 시작 확인
sleep 3
if pgrep -f "api_server.py" > /dev/null; then
    echo "✅ 서버가 성공적으로 시작되었습니다! (포트 5000)"
    echo "📊 로그 확인: tail -f server.log"
else
    echo "❌ 서버 시작 실패. 로그를 확인하세요: cat server.log"
fi