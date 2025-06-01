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

@app.route('/api/recommendation/restaurants', methods=['POST', 'GET'])
def recommend_restaurants():
    """
    식당 추천 API 엔드포인트
    """
    try:
        # GET 요청 처리
        if request.method == 'GET':
            user_id = request.args.get('userId')
            user_category = request.args.getlist('userCategory')  # 쿼리 파라미터에서 리스트로 받기
            remaining_budget = request.args.get('remainingBudget')
        
        # POST 요청 처리 (기존 방식 유지)
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': '요청 데이터가 없습니다'}), 400
            
            user_id = data.get('userId')
            user_category = data.get('userCategory', [])
            remaining_budget = data.get('remainingBudget')
        
        if user_id is None:
            return jsonify({'error': 'userId가 필요합니다'}), 400
        
        logger.info(f"추천 요청 - 사용자: {user_id}, 카테고리: {user_category}, 예산: {remaining_budget}")
        
        # 예산 처리 (문자열을 숫자로 변환)
        budget = None
        if remaining_budget:
            try:
                budget = int(remaining_budget)
            except ValueError:
                logger.warning(f"잘못된 예산 형식: {remaining_budget}")
        
        # 추천 실행
        recommendations = get_restaurant_recommendations(
            user_id=int(user_id),
            budget=budget,
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