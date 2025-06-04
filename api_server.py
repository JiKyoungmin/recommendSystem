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
from pipeline.adaptive_weights import AdaptiveWeightManager

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
            user_category = request.args.getlist('userCategory')
            remaining_budget = request.args.get('remainingBudget')
        
        # POST 요청 처리
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': '요청 데이터가 없습니다', 'restaurantUniqueIds': []}), 400
            
            user_id = data.get('userId')
            user_category = data.get('userCategory', [])
            remaining_budget = data.get('remainingBudget')
        
        if user_id is None:
            return jsonify({'error': 'userId가 필요합니다', 'restaurantUniqueIds': []}), 400
        
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
            user_categories=user_category,
            budget=budget,
            top_n=10
        )
        
        # 안전한 데이터 추출
        restaurant_ids = []
        if recommendations and 'recommendations' in recommendations:
            for rec in recommendations['recommendations']:
                if 'restaurant_id' in rec:
                    # 정수형으로 변환 보장
                    try:
                        rest_id = int(rec['restaurant_id'])
                        restaurant_ids.append(rest_id)
                    except (ValueError, TypeError):
                        logger.warning(f"잘못된 restaurant_id 형식: {rec['restaurant_id']}")
        
        # 디버깅 로그 추가
        logger.info(f"추천 결과 상세:")
        logger.info(f"  전체 응답: {recommendations}")
        logger.info(f"  추출된 ID 수: {len(restaurant_ids)}")
        logger.info(f"  추출된 IDs: {restaurant_ids}")
        
        response = {
            'restaurantUniqueIds': restaurant_ids
        }
        
        logger.info(f"추천 완료 - {len(restaurant_ids)}개 식당")
        logger.info(f"최종 응답: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"추천 중 오류 발생: {str(e)}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return jsonify({
            'error': '서버 내부 오류',
            'restaurantUniqueIds': []
        }), 500

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

    # 전역 가중치 관리자
    weight_manager = AdaptiveWeightManager()

    @app.route('/feedback', methods=['POST'])
    def process_feedback():
        """
        사용자 피드백 처리 및 가중치 업데이트
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': '피드백 데이터가 없습니다'}), 400
            
            user_id = data.get('userId')
            feedback_score = data.get('feedback')
            
            if not user_id or feedback_score is None:
                return jsonify({'error': 'userId와 feedback이 필요합니다'}), 400
            
            # 가중치 업데이트
            updated_weights = weight_manager.update_weights_from_feedback(
                user_id=user_id,
                feedback_score=feedback_score,
                recommendation_method='hybrid'
            )
            
            logger.info(f"피드백 처리 완료 - 사용자: {user_id}, 점수: {feedback_score}")
            
            return jsonify({
                'status': 'success',
                'updated_weights': updated_weights
            })
            
        except Exception as e:
            logger.error(f"피드백 처리 중 오류: {str(e)}")
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

def initialize_scheduler():
    """
    스케줄러 초기화
    """
    try:
        start_auto_scheduler()
        logger.info("✅ 자동 업데이트 스케줄러 시작됨")
    except Exception as e:
        logger.error(f"❌ 스케줄러 시작 실패: {str(e)}")

if __name__ == '__main__':
    initialize_scheduler()
    
    logger.info("🚀 추천 서버 시작 중...")
    app.run(host='0.0.0.0', port=5000, debug=False)