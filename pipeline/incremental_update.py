import pandas as pd
import json
import os
import logging
import pickle
from datetime import datetime
from .preprocessing import ImprovedRestaurantDataPreprocessor
from .svdpp import SVDppRecommendationPipeline
from .contentRecommender import ContentBasedRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalUpdatePipeline:
    """
    주기적 모델 업데이트를 위한 증분 학습 파이프라인
    """
    
    def __init__(self, base_dir='.'):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.result_dir = os.path.join(base_dir, 'result')
        
        # 백업 디렉토리 생성
        self.backup_dir = os.path.join(base_dir, 'backup')
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def load_feedback_data(self):
        """
        피드백 큐에서 새로운 데이터 로드
        """
        feedback_file = os.path.join(self.data_dir, 'feedback_queue.jsonl')
        
        if not os.path.exists(feedback_file):
            logger.info("수집된 피드백 데이터가 없습니다")
            return []
        
        feedback_data = []
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line.strip()))
        
        logger.info(f"피드백 데이터 {len(feedback_data)}개 로드 완료")
        return feedback_data
    
    def process_new_ratings(self, feedback_data):
        """
        피드백 데이터에서 새로운 평점 데이터 추출 및 기존 데이터와 통합
        """
        if not feedback_data:
            logger.info("새로운 평점 데이터가 없습니다")
            return False
        
        # 기존 SVD 데이터 로드
        svd_data_path = os.path.join(self.result_dir, 'svd_data.csv')
        existing_svd_data = pd.read_csv(svd_data_path)
        
        # 새로운 평점 데이터 준비
        new_ratings = []
        new_restaurants = []
        
        for feedback in feedback_data:
            # 평점 데이터가 있는 경우만 처리
            if 'rating' in feedback and feedback['rating'] is not None:
                new_ratings.append({
                    'userId': int(feedback['userId']),
                    'restaurantId': int(feedback['restaurantId']),
                    'rating': float(feedback['rating'])
                })
            
            # 새로운 식당 정보가 있으면 수집
            if 'restaurantCategory' in feedback and 'menuAverage' in feedback:
                new_restaurants.append({
                    'id': int(feedback['restaurantId']),
                    'category': feedback['restaurantCategory'],
                    'menu_average': float(feedback['menuAverage'])
                })
        
        logger.info(f"새로운 평점: {len(new_ratings)}개, 새로운 식당: {len(new_restaurants)}개")
        
        if new_ratings:
            # 새로운 평점을 기존 데이터에 추가
            new_ratings_df = pd.DataFrame(new_ratings)
            updated_svd_data = pd.concat([existing_svd_data, new_ratings_df], ignore_index=True)
            
            # 중복 제거 (같은 사용자가 같은 식당에 여러 평점을 준 경우 최신 것만 유지)
            updated_svd_data = updated_svd_data.drop_duplicates(
                subset=['userId', 'restaurantId'], 
                keep='last'
            )
            
            # 업데이트된 데이터 저장
            updated_svd_data.to_csv(svd_data_path, index=False)
            logger.info(f"SVD 데이터 업데이트 완료: {len(updated_svd_data)}개 평점")
            
            return True
        
        return False
    
    def process_new_restaurants(self, feedback_data):
        """
        새로운 식당 정보를 restaurants.csv에 추가
        """
        restaurants_path = os.path.join(self.data_dir, 'restaurants.csv')
        existing_restaurants = pd.read_csv(restaurants_path)
        
        new_restaurants = []
        for feedback in feedback_data:
            if ('restaurantId' in feedback and 'restaurantCategory' in feedback 
                and 'menuAverage' in feedback):
                
                restaurant_id = int(feedback['restaurantId'])
                
                # 이미 존재하는 식당인지 확인
                if restaurant_id not in existing_restaurants['id'].values:
                    new_restaurants.append({
                        'id': restaurant_id,
                        'name': feedback.get('restaurantName', f'Restaurant_{restaurant_id}'),
                        'category': feedback['restaurantCategory'],
                        'menu_average': float(feedback['menuAverage'])
                    })
        
        if new_restaurants:
            new_restaurants_df = pd.DataFrame(new_restaurants)
            updated_restaurants = pd.concat([existing_restaurants, new_restaurants_df], ignore_index=True)
            
            # 중복 제거
            updated_restaurants = updated_restaurants.drop_duplicates(subset=['id'], keep='last')
            updated_restaurants.to_csv(restaurants_path, index=False)
            
            logger.info(f"새로운 식당 {len(new_restaurants)}개 추가")
            return True
        
        return False
    
    def backup_current_models(self):
        """
        현재 모델들을 백업
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        files_to_backup = [
            'svdpp_model.pkl',
            'prediction_matrix.csv',
            'content_based_matrix.csv'
        ]
        
        for filename in files_to_backup:
            source_path = os.path.join(self.result_dir, filename)
            if os.path.exists(source_path):
                backup_path = os.path.join(self.backup_dir, f'{timestamp}_{filename}')
                import shutil
                shutil.copy2(source_path, backup_path)
                logger.info(f"백업 완료: {filename}")
    
    def retrain_models(self):
        """
        업데이트된 데이터로 모델 재학습
        """
        logger.info("모델 재학습 시작")
        
        try:
            # SVD++ 모델 재학습
            svd_pipeline = SVDppRecommendationPipeline(self.result_dir)
            success = svd_pipeline.run_full_pipeline(quick_search=True, stability_test=False)
            
            if not success:
                logger.error("SVD++ 재학습 실패")
                return False
            
            # 콘텐츠 기반 매트릭스 재생성
            restaurants_path = os.path.join(self.data_dir, 'restaurants.csv')
            content_features_path = os.path.join(self.data_dir, 'content_features.csv')
            mappings_path = os.path.join(self.result_dir, 'restaurant_real_id_mappings.pkl')
            svd_data_path = os.path.join(self.result_dir, 'svd_data.csv')
            
            if os.path.exists(content_features_path):
                content_recommender = ContentBasedRecommender(
                    restaurants_path, content_features_path, mappings_path
                )
                content_recommender.build_similarity_matrix()
                
                svd_data = pd.read_csv(svd_data_path)
                content_matrix = content_recommender.generate_user_restaurant_matrix(svd_data)
                
                content_matrix_path = os.path.join(self.result_dir, 'content_based_matrix.csv')
                content_matrix.to_csv(content_matrix_path)
                
                logger.info("콘텐츠 기반 매트릭스 재생성 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"모델 재학습 중 오류: {str(e)}")
            return False
    
    def archive_processed_feedback(self):
        """
        처리된 피드백 데이터를 아카이브하고 큐 파일 초기화
        """
        feedback_file = os.path.join(self.data_dir, 'feedback_queue.jsonl')
        
        if os.path.exists(feedback_file):
            # 아카이브 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_file = os.path.join(self.data_dir, f'processed_feedback_{timestamp}.jsonl')
            
            # 파일 이동
            import shutil
            shutil.move(feedback_file, archive_file)
            
            logger.info(f"피드백 데이터 아카이브: {archive_file}")
    
    def update_last_update_time(self):
        """
        마지막 업데이트 시간 기록
        """
        update_file = os.path.join(self.result_dir, 'last_update.txt')
        with open(update_file, 'w') as f:
            f.write(datetime.now().isoformat())
    
    def run_incremental_update(self):
        """
        전체 증분 업데이트 프로세스 실행
        """
        logger.info("🔄 증분 업데이트 프로세스 시작")
        
        try:
            # 1. 피드백 데이터 로드
            feedback_data = self.load_feedback_data()
            
            if not feedback_data:
                logger.info("업데이트할 데이터가 없습니다")
                return True
            
            # 2. 현재 모델 백업
            self.backup_current_models()
            
            # 3. 새로운 평점 데이터 처리
            ratings_updated = self.process_new_ratings(feedback_data)
            
            # 4. 새로운 식당 정보 처리
            restaurants_updated = self.process_new_restaurants(feedback_data)
            
            # 5. 데이터가 업데이트되었으면 모델 재학습
            if ratings_updated or restaurants_updated:
                retrain_success = self.retrain_models()
                
                if not retrain_success:
                    logger.error("모델 재학습 실패")
                    return False
            
            # 6. 처리된 피드백 아카이브
            self.archive_processed_feedback()
            
            # 7. 업데이트 시간 기록
            self.update_last_update_time()
            
            logger.info("✅ 증분 업데이트 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 증분 업데이트 중 오류: {str(e)}")
            return False


# 수동 실행을 위한 함수
def run_manual_update():
    """
    수동으로 모델 업데이트 실행
    """
    pipeline = IncrementalUpdatePipeline()
    return pipeline.run_incremental_update()

if __name__ == "__main__":
    # 테스트 실행
    success = run_manual_update()
    if success:
        print("✅ 모델 업데이트 성공")
    else:
        print("❌ 모델 업데이트 실패")