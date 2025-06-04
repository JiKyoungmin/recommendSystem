import pandas as pd
import numpy as np
import pickle
import logging
from .contentRecommender import ContentBasedRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommendationSystem:
    """
    하이브리드 추천 시스템: SVD++ + 콘텐츠 기반 필터링
    """
    
    def __init__(self, svd_matrix_path, restaurants_path, content_features_path, mappings_path):
        """
        초기화: SVD++ 결과와 콘텐츠 기반 추천기 로드
        
        Args:
            svd_matrix_path: SVD++ 예측 매트릭스 경로
            restaurants_path: restaurants.csv 경로
            content_features_path: content_features.csv 경로
            mappings_path: 매핑 파일 경로
        """
        # SVD++ 예측 매트릭스 로드
        self.svd_matrix = pd.read_csv(svd_matrix_path, index_col=0)
        
        # === SVD 매트릭스 컬럼을 정수로 변환 ===
        try:
            # 모든 컬럼을 정수로 변환 시도
            int_columns = []
            for col in self.svd_matrix.columns:
                try:
                    int_columns.append(int(col))
                except ValueError:
                    # 변환 불가능한 컬럼은 원본 유지
                    int_columns.append(col)
            
            self.svd_matrix.columns = int_columns
            logger.info(f"SVD 매트릭스 컬럼을 정수로 변환: {len(int_columns)}개")
            
        except Exception as e:
            logger.warning(f"SVD 매트릭스 컬럼 변환 실패: {e}")

        # 콘텐츠 기반 추천기 초기화
        self.content_recommender = ContentBasedRecommender(
            restaurants_path, content_features_path, mappings_path
        )
        
        # 식당 정보 로드
        self.restaurants = pd.read_csv(restaurants_path)
        self.restaurant_info = self.restaurants.set_index('id').to_dict('index')
        
        # 콘텐츠 기반 매트릭스 생성
        self.content_matrix = None
        
        logger.info("하이브리드 추천 시스템 초기화 완료")
    
    def prepare_content_matrix(self, svd_data_path):
        """
        콘텐츠 기반 예측 매트릭스 생성
        
        Args:
            svd_data_path: SVD 입력 데이터 경로
        """
        logger.info("콘텐츠 기반 매트릭스 준비 중...")
        
        # 유사도 매트릭스 생성
        self.content_recommender.build_similarity_matrix()
        
        # SVD 데이터 로드
        svd_data = pd.read_csv(svd_data_path)
        
        # 콘텐츠 기반 예측 매트릭스 생성
        self.content_matrix = self.content_recommender.generate_user_restaurant_matrix(svd_data)
        
        logger.info("콘텐츠 기반 매트릭스 준비 완료")

    def get_hybrid_recommendations(self, user_id, user_categories=None, budget=None, top_n=10, alpha=0.7, category_boost=0.2):
        """
        하이브리드 추천 수행
        
        Args:
            user_id: 사용자 ID
            user_categories: 사용자 선호 카테고리 리스트
            budget: 예산 제한 (None이면 제한 없음)
            top_n: 추천할 식당 수
            alpha: SVD++ 가중치 (0.7이면 SVD++ 70%, 콘텐츠 30%)
            category_boost: 선호 카테고리 가중치 (0.2 = 20% 추가 점수)
            
        Returns:
            list: 추천 식당 정보 리스트
        """
        logger.info(f"사용자 {user_id}에 대한 하이브리드 추천 시작 (선호 카테고리: {user_categories})")

        # 1. 사용자 존재 확인
        if user_id not in self.svd_matrix.index:
            logger.warning(f"사용자 {user_id}가 SVD 매트릭스에 없습니다")
            return []
        
        if self.content_matrix is None:
            logger.warning("콘텐츠 매트릭스가 준비되지 않았습니다")
            return self._fallback_recommendation(user_id, budget, top_n)
        
        if user_id not in self.content_matrix.index:
            logger.warning(f"사용자 {user_id}가 콘텐츠 매트릭스에 없습니다")
            return self._fallback_recommendation(user_id, budget, top_n)

        # 2. SVD++ 점수 추출
        svd_scores = self.svd_matrix.loc[user_id]
        logger.info(f"SVD 점수 추출 완료: {len(svd_scores)}개 식당")

        # 3. 콘텐츠 기반 점수 추출
        content_scores = self.content_matrix.loc[user_id]
        logger.info(f"콘텐츠 점수 추출 완료: {len(content_scores)}개 식당")

        # 4. 공통 식당 찾기 (두 매트릭스에 모두 존재하는 식당들)
        svd_restaurants = set(svd_scores.index)
        content_restaurants = set(content_scores.index)
        valid_restaurants = svd_restaurants.intersection(content_restaurants)
        
        logger.info(f"공통 식당 수: {len(valid_restaurants)}개")
        
        if len(valid_restaurants) == 0:
            logger.warning("공통 식당이 없습니다. SVD++만으로 추천합니다")
            return self._fallback_recommendation(user_id, budget, top_n)

        # 5. 식당 ID를 정수로 변환하여 딕셔너리 생성
        svd_scores_int = {}
        content_scores_dict = {}
        
        for restaurant_id in valid_restaurants:
            try:
                # 정수 변환 시도
                rest_id_int = int(restaurant_id) if str(restaurant_id).isdigit() else restaurant_id
                svd_scores_int[rest_id_int] = svd_scores[restaurant_id]
                content_scores_dict[rest_id_int] = content_scores[restaurant_id]
            except (ValueError, TypeError):
                # 변환 실패시 원본 사용
                svd_scores_int[restaurant_id] = svd_scores[restaurant_id]
                content_scores_dict[restaurant_id] = content_scores[restaurant_id]

        # 6. 하이브리드 점수 계산
        hybrid_scores = {}
        budget_excluded = 0
        category_boosted = 0
        
        for restaurant_id in svd_scores_int.keys():
            svd_score = svd_scores_int[restaurant_id]
            content_score = content_scores_dict[restaurant_id]
            
            # 기본 하이브리드 점수 계산
            hybrid_score = alpha * svd_score + (1 - alpha) * content_score
            
            # 카테고리 가중치 적용
            if user_categories and restaurant_id in self.restaurant_info:
                restaurant_category = self.restaurant_info[restaurant_id]['category']
                if restaurant_category in user_categories:
                    hybrid_score = hybrid_score * (1 + category_boost)
                    category_boosted += 1
                    logger.debug(f"카테고리 부스트 적용: 식당 {restaurant_id} ({restaurant_category}) - 점수: {hybrid_score:.2f}")
            
            # 예산 필터링
            budget_ok = True
            if budget is not None and restaurant_id in self.restaurant_info:
                menu_price = self.restaurant_info[restaurant_id]['menu_average']
                if menu_price > budget:
                    budget_ok = False
                    budget_excluded += 1
                    logger.debug(f"식당 {restaurant_id} 예산 초과: {menu_price} > {budget}")
            
            if budget_ok:
                hybrid_scores[restaurant_id] = {
                    'hybrid_score': hybrid_score,
                    'svd_score': svd_score,
                    'content_score': content_score
                }

        # 로깅 정보
        if user_categories:
            logger.info(f"선호 카테고리 부스트 적용: {category_boosted}개 식당")
        
        if budget is not None:
            logger.info(f"예산 필터링으로 제외된 식당: {budget_excluded}개")
        
        logger.info(f"최종 후보 식당 수: {len(hybrid_scores)}개")

        # 7. 추천 가능한 식당이 없는 경우
        if len(hybrid_scores) == 0:
            logger.warning("예산/카테고리 조건을 만족하는 식당이 없습니다")
            return self._fallback_recommendation(user_id, budget, top_n)

        # 8. 상위 N개 식당 선택
        sorted_restaurants = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )[:top_n]

        # 9. 추천 결과 생성
        recommendations = []
        for rest_id, scores in sorted_restaurants:
            restaurant_info = self.restaurant_info.get(rest_id, {})
            
            recommendations.append({
                'restaurant_id': rest_id,
                'restaurant_name': restaurant_info.get('name', 'Unknown'),
                'category': restaurant_info.get('category', 'Unknown'),
                'menu_average': restaurant_info.get('menu_average', 0),
                'hybrid_score': round(scores['hybrid_score'], 2),
                'svd_score': round(scores['svd_score'], 2),
                'content_score': round(scores['content_score'], 2)
            })

        logger.info(f"하이브리드 추천 완료: {len(recommendations)}개 식당")
        return recommendations
        
    def _fallback_recommendation(self, user_id, budget, top_n):
        """
        공통 식당이 없을 때 SVD++만으로 추천 (대안책)
        """
        logger.info("대안책: SVD++만으로 추천 진행")
        
        svd_scores = self.svd_matrix.loc[user_id]
        fallback_scores = {}
        
        for restaurant_id, score in svd_scores.items():
            rest_id = int(restaurant_id) if str(restaurant_id).isdigit() else restaurant_id
            
            # 예산 필터링
            if budget is not None and rest_id in self.restaurant_info:
                menu_price = self.restaurant_info[rest_id]['menu_average']
                if menu_price > budget:
                    continue
            
            fallback_scores[rest_id] = {
                'hybrid_score': score,
                'svd_score': score,
                'content_score': 0.0  # 콘텐츠 점수 없음
            }
        
        # 상위 N개 선택
        sorted_restaurants = sorted(
            fallback_scores.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )[:top_n]
        
        recommendations = []
        for rest_id, scores in sorted_restaurants:
            restaurant_info = self.restaurant_info.get(rest_id, {})
            
            recommendations.append({
                'restaurant_id': rest_id,
                'restaurant_name': restaurant_info.get('name', 'Unknown'),
                'category': restaurant_info.get('category', 'Unknown'),
                'menu_average': restaurant_info.get('menu_average', 0),
                'hybrid_score': round(scores['hybrid_score'], 2),
                'svd_score': round(scores['svd_score'], 2),
                'content_score': 0.0
            })
        
        logger.warning(f"대안책으로 {len(recommendations)}개 식당 추천")
        return recommendations
    def get_recommendations_json(self, user_id, user_categories=None, budget=None, top_n=10, alpha=0.7):
        """
        API 응답용 JSON 형태로 추천 결과 반환
        
        Returns:
            dict: JSON 형태의 추천 결과
        """
        recommendations = self.get_hybrid_recommendations(user_id, user_categories, budget, top_n, alpha)
        
        # JSON 형태로 변환
        result = {
            'user_id': user_id,
            'recommendations': [
                {
                    'restaurant_id': rec['restaurant_id'],
                    'restaurant_name': rec['restaurant_name'],
                    'category': rec['category'],
                    'menu_average': rec['menu_average'],
                    'predicted_rating': rec['hybrid_score']
                }
                for rec in recommendations
            ],
            'recommendation_count': len(recommendations),
            'algorithm': f"Hybrid (SVD++: {alpha*100}%, Content: {(1-alpha)*100}%)"
        }
        
        return result

def get_restaurant_recommendations(user_id, user_categories=None, budget=None, top_n=10):
    """
    API 서버에서 호출할 메인 추천 함수
    """
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if 'pipeline' in current_dir:
        base_dir = os.path.dirname(current_dir) 
    else:
        base_dir = current_dir
    
    data_dir = os.path.join(base_dir, 'data')
    result_dir = os.path.join(base_dir, 'result')
    
    svd_matrix_path = os.path.join(result_dir, 'prediction_matrix.csv')
    restaurants_path = os.path.join(data_dir, 'restaurants.csv')
    content_features_path = os.path.join(data_dir, 'content_features.csv')
    mappings_path = os.path.join(result_dir, 'restaurant_real_id_mappings.pkl')
    svd_data_path = os.path.join(result_dir, 'svd_data.csv')
    
    # 파일 존재 여부 확인 및 디버깅 정보 출력
    logger.info(f"파일 경로 확인:")
    logger.info(f"  SVD 매트릭스: {svd_matrix_path} (존재: {os.path.exists(svd_matrix_path)})")
    logger.info(f"  식당 데이터: {restaurants_path} (존재: {os.path.exists(restaurants_path)})")
    logger.info(f"  콘텐츠 특성: {content_features_path} (존재: {os.path.exists(content_features_path)})")
    logger.info(f"  매핑 파일: {mappings_path} (존재: {os.path.exists(mappings_path)})")
    
    try:
        # 하이브리드 시스템 초기화
        hybrid_system = HybridRecommendationSystem(
            svd_matrix_path, restaurants_path, content_features_path, mappings_path
        )
        
        # 콘텐츠 매트릭스 준비
        hybrid_system.prepare_content_matrix(svd_data_path)
        
        # 추천 수행 (user_categories 전달)
        result = hybrid_system.get_recommendations_json(user_id, user_categories, budget, top_n)
        
        return result
    
    except Exception as e:
        logger.error(f"추천 시스템 오류: {e}")
        return {
            'user_id': user_id,
            'recommendations': [],
            'recommendation_count': 0,
            'algorithm': 'Hybrid (Error)',
            'error': str(e)
        }
            
    

# 사용 예시
# 메인 실행 부분도 안전하게 수정

if __name__ == "__main__":
    # 테스트 실행
    test_user_id = 0
    test_budget = 30000  # 3만원 예산
    
    recommendations = get_restaurant_recommendations(
        user_id=test_user_id,
        budget=test_budget,
        top_n=10
    )
    
    print("=== 하이브리드 추천 결과 ===")
    print(f"사용자 ID: {recommendations['user_id']}")
    print(f"예산 제한: {test_budget}원")
    print(f"추천 식당 수: {recommendations['recommendation_count']}")
    
    # 안전한 키 접근
    if 'algorithm' in recommendations:
        print(f"알고리즘: {recommendations['algorithm']}")
    
    # 오류 메시지 출력
    if 'error' in recommendations:
        print(f"오류: {recommendations['error']}")
    
    # 추천 목록 출력
    if recommendations['recommendations']:
        print("\n추천 식당 목록:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"{i}. {rec['restaurant_name']} ({rec['category']})")
            print(f"   가격: {rec['menu_average']}원, 예상평점: {rec['predicted_rating']}")
    else:
        print("\n추천 가능한 식당이 없습니다.")