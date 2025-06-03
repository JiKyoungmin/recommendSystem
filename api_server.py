from flask import Flask, request, jsonify
import sys
import logging
import os
from pipeline.hybridRecommender import get_restaurant_recommendations
import json
from datetime import datetime
import pandas as pd
from scheduler import start_auto_scheduler, stop_auto_scheduler, scheduler_instance
from pipeline.incremental_update import run_manual_update

# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/recommendation/restaurants', methods=['GET', 'POST'])
def recommend_restaurants():
    """
    식당 추천 API 엔드포인트
    """
    try:
        # GET 요청 처리
        if request.method == 'GET':
            user_id = request.args.get('userId')
            user_category = request.args.getlist('userCategory')  # 리스트로 받기
            remaining_budget = request.args.get('remainingBudget')
        
        # POST 요청 처리
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': '요청 데이터가 없습니다'}), 400
            
            user_id = data.get('userId')
            user_category = data.get('userCategory', [])  # 리스트로 받기
            remaining_budget = data.get('remainingBudget')
        
        if user_id is None:
            return jsonify({'error': 'userId가 필요합니다'}), 400
        
        logger.info(f"추천 요청 - 사용자: {user_id}, 선호 카테고리: {user_category}, 예산: {remaining_budget}")
        
        # 예산 처리 
        budget = None
        if remaining_budget:
            try:
                budget = int(remaining_budget)
            except ValueError:
                logger.warning(f"잘못된 예산 형식: {remaining_budget}")
        
        # 추천 실행
        recommendations = get_restaurant_recommendations(
            user_id=int(user_id),
            user_categories=user_category,  # 새로 추가
            budget=budget,
            top_n=10
        )
    
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

@app.route('/feedback/ratings', methods=['POST'])
def collect_feedback():
    """
    사용자 피드백 및 새로운 평점 데이터 수집
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다'}), 400
        
        # 데이터가 리스트인지 확인 (단일 객체면 리스트로 변환)
        if isinstance(data, dict):
            data = [data]
        
        logger.info(f"피드백 데이터 {len(data)}개 수신")
        
        # 피드백 데이터를 파일에 저장
        feedback_file = os.path.join('data', 'feedback_queue.jsonl')
        
        # 각 피드백을 JSONL 형태로 저장 (한 줄씩 추가)
        with open(feedback_file, 'a', encoding='utf-8') as f:
            for feedback_item in data:
                # 타임스탬프 추가
                feedback_item['timestamp'] = datetime.now().isoformat()
                f.write(json.dumps(feedback_item, ensure_ascii=False) + '\n')
        
        logger.info(f"피드백 데이터 저장 완료: {feedback_file}")
        
        return jsonify({
            'status': 'success',
            'message': f'{len(data)}개의 피드백이 저장되었습니다',
            'received_count': len(data)
        })
        
    except Exception as e:
        logger.error(f"피드백 수집 중 오류: {str(e)}")
        return jsonify({'error': '서버 내부 오류'}), 500

#자동 모델 업데이트
@app.route('/system/update-status', methods=['GET'])
def get_update_status():
    """
    모델 업데이트 상태 확인
    """
    try:
        # 마지막 업데이트 시간 확인
        update_log_file = os.path.join('result', 'last_update.txt')
        
        if os.path.exists(update_log_file):
            with open(update_log_file, 'r') as f:
                last_update = f.read().strip()
        else:
            last_update = "업데이트 기록 없음"
        
        # 대기 중인 피드백 수 확인
        feedback_file = os.path.join('data', 'feedback_queue.jsonl')
        pending_feedback = 0
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                pending_feedback = sum(1 for line in f if line.strip())
        
        return jsonify({
            'last_update': last_update,
            'pending_feedback_count': pending_feedback,
            'status': 'ready_for_update' if pending_feedback > 0 else 'up_to_date'
        })
        
    except Exception as e:
        logger.error(f"상태 확인 중 오류: {str(e)}")
        return jsonify({'error': '서버 내부 오류'}), 500

@app.before_first_request
def initialize_scheduler():
    """
    서버 시작 시 자동 스케줄러 초기화
    """
    try:
        start_auto_scheduler()
        logger.info("✅ 자동 업데이트 스케줄러 시작됨")
    except Exception as e:
        logger.error(f"❌ 스케줄러 시작 실패: {str(e)}")

@app.route('/system/manual-update', methods=['POST'])
def trigger_manual_update():
    """
    수동으로 모델 업데이트 트리거
    """
    try:
        logger.info("수동 모델 업데이트 요청 수신")
        
        # 백그라운드에서 업데이트 실행
        import threading
        
        def run_update():
            success = run_manual_update()
            if success:
                logger.info("✅ 수동 업데이트 완료")
            else:
                logger.error("❌ 수동 업데이트 실패")
        
        update_thread = threading.Thread(target=run_update, daemon=True)
        update_thread.start()
        
        return jsonify({
            'status': 'started',
            'message': '모델 업데이트가 백그라운드에서 시작되었습니다'
        })
        
    except Exception as e:
        logger.error(f"수동 업데이트 트리거 중 오류: {str(e)}")
        return jsonify({'error': '서버 내부 오류'}), 500

@app.route('/system/scheduler-info', methods=['GET'])
def get_scheduler_info():
    """
    스케줄러 정보 조회
    """
    try:
        global scheduler_instance
        
        if scheduler_instance:
            next_update = scheduler_instance.get_next_update_time()
            is_running = scheduler_instance.is_running
        else:
            next_update = "스케줄러 없음"
            is_running = False
        
        return jsonify({
            'scheduler_running': is_running,
            'next_scheduled_update': next_update,
            'update_frequency': 'Weekly (Sunday 02:00)'
        })
        
    except Exception as e:
        logger.error(f"스케줄러 정보 조회 중 오류: {str(e)}")
        return jsonify({'error': '서버 내부 오류'}), 500

# 서버 종료 시 스케줄러도 정리
import atexit

def cleanup_scheduler():
    """
    서버 종료 시 스케줄러 정리
    """
    try:
        stop_auto_scheduler()
        logger.info("스케줄러 정리 완료")
    except Exception as e:
        logger.error(f"스케줄러 정리 중 오류: {str(e)}")

atexit.register(cleanup_scheduler)

if __name__ == '__main__':
    # 개발용 설정
    app.run(host='0.0.0.0', port=5000, debug=False)