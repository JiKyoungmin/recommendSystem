from flask import Flask, request, jsonify
import sys
import logging
import os
from pipeline.hybridRecommender import get_restaurant_recommendations

# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('recommendation/restaurants', methods=['POST'])
def recommend_restaurants():
    """
    식당 추천 API 엔드포인트
    """
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다'}), 400
        
        user_id = data.get('userId')
        user_category = data.get('userCategory', [])
        
        if user_id is None:
            return jsonify({'error': 'userId가 필요합니다'}), 400
        
        logger.info(f"추천 요청 - 사용자: {user_id}, 카테고리: {user_category}")
        
        # 추천 실행 (예산은 나중에 사용자 프로필에서 가져올 예정)
        recommendations = get_restaurant_recommendations(
            user_id=int(user_id),
            budget=None,  # 현재는 예산 제한 없음
            top_n=10
        )
        
        # 응답 형식에 맞게 변환
        restaurant_ids = [rec['restaurant_id'] for rec in recommendations['recommendations']]
        
        response = {
            'restaurantUniqueIds': restaurant_ids
        }
        
        logger.info(f"추천 완료 - {len(restaurant_ids)}개 식당")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"추천 중 오류 발생: {str(e)}")
        return jsonify({
            'error': '서버 내부 오류',
            'restaurantUniqueIds': []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    서버 상태 확인
    """
    return jsonify({'status': 'healthy', 'message': '추천 서버가 정상 작동 중입니다'})

if __name__ == '__main__':
    # 개발용 설정
    app.run(host='0.0.0.0', port=5000, debug=False)