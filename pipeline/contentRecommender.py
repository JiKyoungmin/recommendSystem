import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """
    식당 콘텐츠 기반 추천 시스템
    식당×식당 유사도를 사용자×식당 예측 평점으로 변환
    """
    
    def __init__(self, restaurants_path, content_features_path, mappings_path):
        """
        초기화: 필요한 데이터 로드
        
        Args:
            restaurants_path: restaurants.csv 경로
            content_features_path: content_features.csv 경로  
            mappings_path: restaurant_mappings.pkl 경로
        """
        self.restaurants = pd.read_csv(restaurants_path)
        self.content_features = pd.read_csv(content_features_path)
        
        # 매핑 정보 로드
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        # 실제 ID 매핑 정보 (새로운 매핑 파일 구조에 맞게 수정)
        if 'all_mappings' in mappings:
            # 새로운 구조: restaurant_real_id_mappings.pkl
            self.survey_to_id = mappings['all_mappings']['survey_to_id']
            self.id_to_survey = mappings['all_mappings']['id_to_survey']
        else:
            # 기존 구조: restaurant_mappings.pkl
            self.survey_to_id = mappings['restaurant_to_id']
            self.id_to_survey = mappings['id_to_restaurant']
        
        self.restaurant_similarity_matrix = None
        self.restaurant_info_dict = None
        
        # 식당 정보 딕셔너리 생성
        self._create_restaurant_info_dict()
        
        logger.info(f"콘텐츠 기반 추천 시스템 초기화 완료")
        logger.info(f"식당 수: {len(self.restaurants)}, 매핑 수: {len(self.survey_to_id)}")
    
    def _create_restaurant_info_dict(self):
        """식당 정보를 딕셔너리로 변환 (빠른 조회를 위해)"""
        self.restaurant_info_dict = {}
        
        for _, restaurant in self.restaurants.iterrows():
            self.restaurant_info_dict[restaurant['id']] = {
                'name': restaurant['name'],
                'category': restaurant['category'],
                'menu_average': restaurant['menu_average']
            }
    
    def build_similarity_matrix(self):
        """
        식당×식당 콘텐츠 유사도 매트릭스 생성
        카테고리와 가격대를 기반으로 유사도 계산
        """
        logger.info("콘텐츠 유사도 매트릭스 생성 시작")
        
        # 수치형 특성만 추출 (카테고리 더미 변수 + 정규화된 가격)
        feature_columns = [col for col in self.content_features.columns 
                          if col not in ['id', 'name']]
        
        # 콘텐츠 특성 매트릭스
        feature_matrix = self.content_features[feature_columns].values
        
        # 표준화 (중요: 가격과 카테고리 스케일이 다르므로)
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(feature_matrix_scaled)
        
        # 실제 ID를 매핑 ID로 변환하는 딕셔너리 생성
        real_to_mapping_id = {}
        for mapping_id, survey_name in self.id_to_survey.items():
            # 타입 통일 (정수로 변환)
            mapping_id = int(mapping_id) if isinstance(mapping_id, str) else mapping_id
            
            # 설문 이름에서 실제 ID 찾기
            if survey_name in self.survey_to_id:
                real_id = self.survey_to_id[survey_name]
                real_to_mapping_id[real_id] = mapping_id
        
        # content_features의 실제 ID를 매핑 ID로 변환
        content_real_ids = self.content_features['id'].values
        mapped_ids = []
        valid_indices = []
        
        for i, real_id in enumerate(content_real_ids):
            if real_id in real_to_mapping_id:
                mapped_ids.append(real_to_mapping_id[real_id])
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            logger.error("매칭되는 식당이 없습니다!")
            return None
        
        # 매칭된 식당들만으로 유사도 매트릭스 생성
        filtered_similarity = similarity_matrix[np.ix_(valid_indices, valid_indices)]
        
        # DataFrame으로 변환 (매핑 ID 기준)
        self.restaurant_similarity_matrix = pd.DataFrame(
            filtered_similarity,
            index=mapped_ids,
            columns=mapped_ids
        )
        
        logger.info(f"유사도 매트릭스 생성 완료: {filtered_similarity.shape}")
        logger.info(f"매핑된 식당 수: {len(mapped_ids)}")
        
        return self.restaurant_similarity_matrix
    
    def predict_user_rating_for_restaurant(self, user_ratings, target_restaurant_id, min_similarity=0.1):
        """
        특정 사용자의 특정 식당에 대한 예상 평점 계산
        
        Args:
            user_ratings: 사용자의 과거 평점 딕셔너리 {restaurant_id: rating}
            target_restaurant_id: 예측하고 싶은 식당 ID
            min_similarity: 최소 유사도 임계값
            
        Returns:
            float: 예상 평점 (1-5점)
        """
        if self.restaurant_similarity_matrix is None:
            logger.error("유사도 매트릭스가 생성되지 않았습니다!")
            return 2.5  # 기본값
        
        # 타겟 식당이 유사도 매트릭스에 있는지 확인
        if target_restaurant_id not in self.restaurant_similarity_matrix.columns:
            # 전체 평균 반환
            return np.mean(list(user_ratings.values())) if user_ratings else 2.5
        
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        # 사용자가 평가한 각 식당과의 유사도를 기반으로 가중 평균 계산
        for rated_restaurant_id, rating in user_ratings.items():
            if (rated_restaurant_id in self.restaurant_similarity_matrix.index and 
                rated_restaurant_id != target_restaurant_id):
                
                # 유사도 가져오기
                similarity = self.restaurant_similarity_matrix.loc[rated_restaurant_id, target_restaurant_id]
                
                # 최소 유사도 이상인 경우만 고려
                if similarity >= min_similarity:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
        
        # 가중 평균 계산
        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum
            # 1-5 범위로 제한
            return max(1.0, min(5.0, predicted_rating))
        else:
            # 유사한 식당이 없으면 사용자 평균 반환
            return np.mean(list(user_ratings.values())) if user_ratings else 2.5
    
    def generate_user_restaurant_matrix(self, svd_data):
        """
        사용자×식당 콘텐츠 기반 예측 평점 매트릭스 생성 (정수 컬럼 보장)
        """
        logger.info("사용자×식당 콘텐츠 기반 예측 매트릭스 생성 시작")
        
        unique_users = sorted(svd_data['userId'].unique())
        unique_restaurants = sorted(svd_data['restaurantId'].unique())
        
        # === 식당 ID를 정수로 변환 ===
        unique_restaurants = [int(rid) for rid in unique_restaurants]
        
        logger.info(f"예측 매트릭스 크기: {len(unique_users)} x {len(unique_restaurants)}")
        logger.info(f"식당 ID 샘플: {unique_restaurants[:10]}")
        
        # 사용자별 과거 평점 딕셔너리 생성
        user_ratings_dict = {}
        for user_id in unique_users:
            user_data = svd_data[svd_data['userId'] == user_id]
            # 여기서도 식당 ID를 정수로 변환
            user_ratings_dict[user_id] = {int(rid): rating for rid, rating in zip(user_data['restaurantId'], user_data['rating'])}
        
        # 예측 매트릭스 초기화
        prediction_matrix = np.zeros((len(unique_users), len(unique_restaurants)))
        
        # 각 사용자-식당 조합에 대해 예측 평점 계산
        for i, user_id in enumerate(unique_users):
            user_ratings = user_ratings_dict[user_id]
            
            for j, restaurant_id in enumerate(unique_restaurants):
                if restaurant_id in user_ratings:
                    # 이미 평가한 식당은 실제 평점 사용
                    prediction_matrix[i, j] = user_ratings[restaurant_id]
                else:
                    # 평가하지 않은 식당은 콘텐츠 기반으로 예측
                    predicted_rating = self.predict_user_rating_for_restaurant(
                        user_ratings, restaurant_id
                    )
                    prediction_matrix[i, j] = predicted_rating
            
            if (i + 1) % 10 == 0:
                logger.info(f"예측 진행률: {(i+1)/len(unique_users)*100:.1f}%")
        
        # DataFrame으로 변환 (정수 컬럼)
        content_based_matrix = pd.DataFrame(
            prediction_matrix,
            index=unique_users,
            columns=unique_restaurants  # 이미 정수로 변환됨
        )
        
        logger.info("콘텐츠 기반 예측 매트릭스 생성 완료")
        return content_based_matrix

    
    def get_similar_restaurants(self, restaurant_id, top_n=10):
        """
        특정 식당과 유사한 식당들 반환 (디버깅용)
        
        Args:
            restaurant_id: 기준 식당 ID
            top_n: 반환할 유사 식당 수
            
        Returns:
            list: 유사한 식당들의 (식당ID, 유사도) 튜플 리스트
        """
        if (self.restaurant_similarity_matrix is None or 
            restaurant_id not in self.restaurant_similarity_matrix.columns):
            return []
        
        # 해당 식당과의 유사도 추출
        similarities = self.restaurant_similarity_matrix.loc[:, restaurant_id]
        
        # 자기 자신 제외하고 상위 N개 추출
        similar_restaurants = similarities[similarities.index != restaurant_id].nlargest(top_n)
        
        return [(rest_id, sim_score) for rest_id, sim_score in similar_restaurants.items()]


# 사용 예시
if __name__ == "__main__":
    import os
    
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    result_dir = os.path.join(current_dir, '..', 'result')
    
    restaurants_path = os.path.join(data_dir, 'restaurants.csv')
    content_features_path = os.path.join(data_dir, 'content_features.csv')
    mappings_path = os.path.join(result_dir, 'restaurant_real_id_mappings.pkl')  # 새로운 매핑 파일
    svd_data_path = os.path.join(result_dir, 'svd_data.csv')
    
    # 콘텐츠 기반 추천 시스템 초기화
    recommender = ContentBasedRecommender(
        restaurants_path, content_features_path, mappings_path
    )
    
    # 유사도 매트릭스 생성
    similarity_matrix = recommender.build_similarity_matrix()
    
    # SVD 데이터 로드
    svd_data = pd.read_csv(svd_data_path)
    
    # 콘텐츠 기반 예측 매트릭스 생성
    content_matrix = recommender.generate_user_restaurant_matrix(svd_data)
    
    # 결과 저장
    content_matrix_path = os.path.join(result_dir, 'content_based_matrix.csv')
    content_matrix.to_csv(content_matrix_path)
    
    print(f"콘텐츠 기반 예측 매트릭스 저장 완료: {content_matrix_path}")
    print(f"매트릭스 크기: {content_matrix.shape}")
    
    # 예시: 특정 식당과 유사한 식당들 확인
    sample_restaurant_id = svd_data['restaurantId'].iloc[0]
    similar_restaurants = recommender.get_similar_restaurants(sample_restaurant_id, top_n=5)
    
    print(f"\n식당 ID {sample_restaurant_id}와 유사한 식당들:")
    for rest_id, similarity in similar_restaurants:
        print(f"  식당 ID {rest_id}: 유사도 {similarity:.3f}")