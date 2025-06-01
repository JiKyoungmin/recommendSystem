#!/bin/bash
echo "📊 서버 상태 확인..."

# 프로세스 확인
if pgrep -f "api_server.py" > /dev/null; then
    echo "✅ 서버 실행 중"
    echo "🔍 프로세스 정보:"
    ps aux | grep api_server.py | grep -v grep
    
    echo ""
    echo "🌐 포트 상태:"
    netstat -tlnp | grep :5000
    
    echo ""
    echo "📄 최근 로그 (마지막 10줄):"
    tail -10 server.log
else
    echo "❌ 서버가 실행되지 않음"
    echo "📄 오류 로그 확인:"
    tail -20 server.log
fi