import pandas as pd
import numpy as np
import pickle
import logging
import time
import os
from collections import defaultdict
from surprise import SVDpp, Dataset, Reader
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate, KFold
from surprise import accuracy
import random

RANDOM_SEED = 662
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVDppRecommendationPipeline:
    """
    SVD++ 기반 추천 시스템 파이프라인
    한끼모아 프로젝트용 식당 추천 알고리즘
    전처리된 데이터를 불러와서 SVD++ 모델 학습 및 추천 수행
    """
    
    def __init__(self, result_path):
        """
        파이프라인 초기화
        
        Args:
            result_path: 전처리된 데이터가 있는 폴더 경로
        """
        self.result_path = result_path
        self.processed_data = None
        self.surprise_data = None
        self.restaurant_mappings = None
        self.best_model = None
        self.best_params = None
        self.prediction_matrix = None
        
    def load_preprocessed_data(self):
        """
        전처리된 데이터 로드
        preprocessing 파이프라인에서 생성된 result 폴더의 파일들을 로드
        """
        logger.info("📁 전처리된 데이터 로드 시작")
        
        # 1. CSV 데이터 로드
        csv_path = os.path.join(self.result_path, 'svd_data.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"전처리된 CSV 파일을 찾을 수 없습니다: {csv_path}")
        
        self.processed_data = pd.read_csv(csv_path)
        logger.info(f"CSV 데이터 로드 완료: {self.processed_data.shape}")
        
        # 2. 매핑 딕셔너리 로드
        mapping_path = os.path.join(self.result_path, 'restaurant_mappings.pkl')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"매핑 파일을 찾을 수 없습니다: {mapping_path}")
        
        with open(mapping_path, 'rb') as f:
            self.restaurant_mappings = pickle.load(f)
        logger.info(f"매핑 딕셔너리 로드 완료: {len(self.restaurant_mappings['restaurant_to_id'])}개 식당")
        
        # 3. Surprise 데이터셋 로드 (있으면)
        dataset_path = os.path.join(self.result_path, 'surprise_dataset.pkl')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                self.surprise_data = pickle.load(f)
            logger.info("기존 Surprise 데이터셋 로드 완료")
        else:
            # 없으면 CSV에서 새로 생성
            logger.info("Surprise 데이터셋 새로 생성")
            self.surprise_data = self.create_surprise_dataset(self.processed_data)
        
        return True
    
    def create_surprise_dataset(self, svd_data):
        """
        Surprise 라이브러리용 데이터셋 생성
        SVD++ 모델 학습을 위한 데이터 형태로 변환
        """
        logger.info("🎯 Surprise 데이터셋 생성")
        
        reader = Reader(rating_scale=(1, 5))
        surprise_data = Dataset.load_from_df(svd_data, reader)
        
        logger.info("✅ Surprise 데이터셋 생성 완료")
        return surprise_data
    
    def get_stable_performance(self, surprise_data, n_runs=5):
        """
        여러 seed로 실행해서 안정적인 성능 평가
        
        Args:
            surprise_data: Surprise 데이터셋
            n_runs: 실행할 seed 개수
            
        Returns:
            dict: 평균 성능과 분산 정보
        """
        logger.info(f"🎯 안정적인 성능 평가 시작 ({n_runs}회 실행)")
        
        seeds = [42, 123, 456, 789, 999][:n_runs]
        rmse_scores = []
        mae_scores = []
        
        for i, seed in enumerate(seeds, 1):
            logger.info(f"  실행 {i}/{n_runs} (seed: {seed})")
            
            # 각 seed로 모델 생성
            model = SVDpp(
                n_factors=50, 
                n_epochs=20, 
                lr_all=0.01,
                reg_all=0.05,
                random_state=seed
            )
            
            # 교차검증으로 성능 측정
            cv_results = cross_validate(
                model, surprise_data, 
                measures=['RMSE', 'MAE'], 
                cv=3, 
                verbose=False
            )
            
            rmse_scores.append(cv_results['test_rmse'].mean())
            mae_scores.append(cv_results['test_mae'].mean())
        
        # 결과 정리
        performance_stats = {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'rmse_min': np.min(rmse_scores),
            'rmse_max': np.max(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'mae_min': np.min(mae_scores),
            'mae_max': np.max(mae_scores),
            'individual_rmse': rmse_scores,
            'individual_mae': mae_scores
        }
        
        # 결과 로깅
        logger.info("📊 안정적인 성능 평가 완료:")
        logger.info(f"  RMSE: {performance_stats['rmse_mean']:.4f} ± {performance_stats['rmse_std']:.4f}")
        logger.info(f"  RMSE 범위: [{performance_stats['rmse_min']:.4f}, {performance_stats['rmse_max']:.4f}]")
        logger.info(f"  MAE: {performance_stats['mae_mean']:.4f} ± {performance_stats['mae_std']:.4f}")
        
        return performance_stats

    def optimize_hyperparameters(self, surprise_data, quick_search=True):
        """
        SVD++ 하이퍼파라미터 최적화
        GridSearchCV를 통한 최적 파라미터 탐색
        """
        logger.info("⚙️ 하이퍼파라미터 최적화 시작")
        
        if quick_search:
            # 빠른 탐색용 파라미터
            param_grid = {
                'n_factors': [50, 100],
                'n_epochs': [20, 30], 
                'lr_all': [0.01, 0.02],
                'reg_all': [0.02, 0.05],
                'random_state': [RANDOM_SEED]
            }
            cv_folds = 3
        else:
            # 정밀 탐색용 파라미터
            param_grid = {
                'n_factors': [50, 100, 150],
                'n_epochs': [20, 30, 40],
                'lr_all': [0.005, 0.01, 0.02],
                'reg_all': [0.02, 0.05, 0.1],
                'random_state': [RANDOM_SEED]
            }
            cv_folds = 5
        
        start_time = time.time()
        gs = GridSearchCV(
            SVDpp,
            param_grid,
            measures=['rmse', 'mae'],
            cv=cv_folds,
            n_jobs=-1,
            joblib_verbose=1
        )
        
        gs.fit(surprise_data)
        end_time = time.time()
        
        self.best_params = gs.best_params['rmse']
        
        logger.info(f"최적화 완료! 소요 시간: {end_time - start_time:.2f}초")
        logger.info(f"최적 파라미터: {self.best_params}")
        logger.info(f"최적 RMSE: {gs.best_score['rmse']:.4f}")
        logger.info(f"최적 MAE: {gs.best_score['mae']:.4f}")
        
        return gs.best_params['rmse'], gs.best_score
    
    def train_final_model(self, surprise_data, best_params):
        """
        최적 파라미터로 최종 모델 학습
        전체 데이터셋을 사용하여 최종 SVD++ 모델 구축
        """
        logger.info("🎓 최종 모델 학습 시작")
        
        # 최적 파라미터로 모델 생성
        self.best_model = SVDpp(**best_params)
        
        # 전체 데이터로 학습
        trainset = surprise_data.build_full_trainset()
        self.best_model.fit(trainset)
        
        logger.info("✅ 최종 모델 학습 완료")
        return self.best_model
    
    def generate_prediction_matrix(self, surprise_data):
        """
        전체 사용자-식당 예측 평점 매트릭스 생성
        모든 사용자에 대해 모든 식당의 예측 평점 계산
        """
        logger.info("📊 예측 평점 매트릭스 생성 시작")
        
        # 전체 trainset에서 사용자와 아이템 ID 추출
        full_trainset = surprise_data.build_full_trainset()
        
        # 내부 ID를 원본 ID로 변환
        unique_users = [full_trainset.to_raw_uid(uid) for uid in full_trainset.all_users()]
        unique_items = [full_trainset.to_raw_iid(iid) for iid in full_trainset.all_items()]
        
        unique_users = sorted(unique_users)
        unique_items = sorted(unique_items)
        
        logger.info(f"매트릭스 크기: {len(unique_users)} x {len(unique_items)}")
        
        # 예측 매트릭스 초기화
        prediction_matrix = np.zeros((len(unique_users), len(unique_items)))
        
        start_time = time.time()
        total_predictions = len(unique_users) * len(unique_items)
        
        # 모든 사용자-아이템 조합 예측
        for i, user_id in enumerate(unique_users):
            for j, item_id in enumerate(unique_items):
                pred = self.best_model.predict(user_id, item_id)
                prediction_matrix[i, j] = pred.est
            
            # 진행 상황 출력
            if (i + 1) % 10 == 0:
                progress = ((i + 1) * len(unique_items)) / total_predictions * 100
                logger.info(f"예측 진행률: {progress:.1f}%")
        
        end_time = time.time()
        logger.info(f"예측 매트릭스 생성 완료! 소요 시간: {end_time - start_time:.2f}초")
        
        # DataFrame으로 변환
        self.prediction_matrix = pd.DataFrame(
            prediction_matrix,
            index=unique_users,
            columns=unique_items
        )
        
        return self.prediction_matrix
    
    def save_model_results(self):
        """
        모델 학습 결과만 저장
        전처리된 데이터는 이미 있으므로 모델과 예측 매트릭스만 저장
        """
        logger.info("💾 모델 결과 저장 시작")
        
        # 1. 최종 모델 저장
        if self.best_model is not None:
            model_path = os.path.join(self.result_path, 'svdpp_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'params': self.best_params
                }, f)
            logger.info(f"모델 저장: {model_path}")
        
        # 2. 예측 매트릭스 저장
        if self.prediction_matrix is not None:
            matrix_path = os.path.join(self.result_path, 'prediction_matrix.csv')
            self.prediction_matrix.to_csv(matrix_path)
            logger.info(f"예측 매트릭스 저장: {matrix_path}")
        
        logger.info("✅ 모델 결과 저장 완료")
    
    def run_full_pipeline(self, quick_search=True, stability_test=True):
        """
        전체 파이프라인 실행
        전처리된 데이터 로드부터 모델 학습, 결과 저장까지 모든 과정을 순차 실행
        
        Args:
            quick_search: True면 빠른 하이퍼파라미터 탐색, False면 정밀 탐색
            stability_test: True면 안정적인 성능 평가 수행
        """
        logger.info("🚀 SVD++ 추천 시스템 파이프라인 시작")

        
        try:
            # 1. 전처리된 데이터 로드
            self.load_preprocessed_data()
            
            # 2. 안정적인 성능 평가 (선택사항)
            if stability_test:
                stability_results = self.get_stable_performance(self.surprise_data, n_runs=3)
            
                # 성능이 너무 불안정하면 경고
                if stability_results['rmse_std'] > 0.7:
                    logger.warning(f"⚠️  RMSE 표준편차가 높음: {stability_results['rmse_std']:.4f}")
                    logger.warning("데이터 증가 후 재평가 권장")

            # 2. 하이퍼파라미터 최적화
            best_params, best_scores = self.optimize_hyperparameters(
                self.surprise_data, quick_search=quick_search
            )
            
            # 3. 최종 모델 학습
            self.train_final_model(self.surprise_data, best_params)
            
            # 4. 예측 매트릭스 생성
            self.generate_prediction_matrix(self.surprise_data)
            
            # 5. 결과 저장 (모델과 예측 매트릭스만)
            self.save_model_results()
            
            logger.info("🎉 SVD++ 파이프라인 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 중 오류 발생: {str(e)}")
            return False
    
    def get_recommendations(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        특정 사용자에 대한 식당 추천
        
        Args:
            user_id: 사용자 ID
            n_recommendations: 추천할 식당 수
            exclude_rated: 이미 평가한 식당 제외 여부
        
        Returns:
            추천 식당 리스트 (식당명, 예측평점)
        """
        if self.prediction_matrix is None:
            logger.error("예측 매트릭스가 생성되지 않았습니다.")
            return None
        
        if user_id not in self.prediction_matrix.index:
            logger.warning(f"사용자 ID {user_id}를 찾을 수 없습니다.")
            return None
        
        # 사용자의 예측 평점 추출
        user_predictions = self.prediction_matrix.loc[user_id]
        
        # 이미 평가한 식당 제외 (선택사항)
        if exclude_rated and self.processed_data is not None:
            rated_restaurants = self.processed_data[
                self.processed_data['userId'] == user_id
            ]['restaurantId'].tolist()
            
            for restaurant_id in rated_restaurants:
                if restaurant_id in user_predictions.index:
                    user_predictions = user_predictions.drop(restaurant_id)
        
        # 상위 N개 추천
        top_recommendations = user_predictions.nlargest(n_recommendations)
        
        # 식당 이름으로 변환
        recommendations = []
        for restaurant_id, predicted_rating in top_recommendations.items():
            restaurant_name = self.restaurant_mappings['id_to_restaurant'][restaurant_id]
            recommendations.append({
                'restaurant_name': restaurant_name,
                'predicted_rating': round(predicted_rating, 2),
                'restaurant_id': restaurant_id
            })
        
        return recommendations

# 사용 예시
if __name__ == "__main__":
    import os
    
    # 현재 스크립트의 디렉토리 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')  # 상위 폴더의 data 폴더
    result_dir = os.path.join(current_dir, '..', 'result')  # 상위 폴더의 result 폴더
    
    # 결과 폴더가 없으면 생성
    os.makedirs(result_dir, exist_ok=True)
    
    input_file_path=os.path.join(data_dir, 'rating.xlsx')
    mappings_output_path=os.path.join(result_dir, 'restaurant_mappings.pkl')
    csv_output_path=os.path.join(result_dir, 'svd_data.csv')
    surprise_output_path=os.path.join(result_dir, 'surprise_dataset.pkl')

    # 파이프라인 초기화 (전처리된 데이터가 있는 result 폴더 지정)
    pipeline = SVDppRecommendationPipeline(result_dir)
    
    # 전체 파이프라인 실행
    success = pipeline.run_full_pipeline(quick_search=True, stability_test=True)
    
    if success:
        # 사용자 0에 대한 추천 예시
        recommendations = pipeline.get_recommendations(user_id=0, n_recommendations=10)
        if recommendations:
            print("\n🍽️  추천 식당:")
            for rec in recommendations:
                print(f"- {rec['restaurant_name']}: {rec['predicted_rating']}점")
