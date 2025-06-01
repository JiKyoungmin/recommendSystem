#!/bin/bash
echo "🛑 서버 종료 중..."

if pgrep -f "api_server.py" > /dev/null; then
    pkill -f api_server.py
    echo "✅ 서버가 종료되었습니다."
else
    echo "ℹ️  실행 중인 서버가 없습니다."
fi