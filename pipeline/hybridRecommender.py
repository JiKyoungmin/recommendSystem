import pandas as pd
import numpy as np
import pickle
import logging
from contentRecommender import ContentBasedRecommender

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

    def get_hybrid_recommendations(self, user_id, budget=None, top_n=10, alpha=0.7):
        """
        하이브리드 추천 수행
        
        Args:
            user_id: 사용자 ID
            budget: 예산 제한 (None이면 제한 없음)
            top_n: 추천할 식당 수
            alpha: SVD++ 가중치 (0.7이면 SVD++ 70%, 콘텐츠 30%)
            
        Returns:
            list: 추천 식당 정보 리스트
        """
        logger.info(f"사용자 {user_id}에 대한 하이브리드 추천 시작")
    
        # 사용자 존재 확인
        if user_id not in self.svd_matrix.index:
            logger.warning(f"사용자 {user_id}가 SVD 매트릭스에 없음")
            return []
        
        if self.content_matrix is None:
            logger.error("콘텐츠 매트릭스가 준비되지 않음")
            return []
        
        if user_id not in self.content_matrix.index:
            logger.warning(f"사용자 {user_id}가 콘텐츠 매트릭스에 없음")
            return []
        
        # 사용자별 점수 가져오기
        svd_scores = self.svd_matrix.loc[user_id]
        content_scores = self.content_matrix.loc[user_id]
        
        # SVD 매트릭스 컬럼을 정수로 변환
        svd_scores_int = pd.Series(index=[int(col) for col in svd_scores.index], data=svd_scores.values)
        
        # 공통 식당 ID 찾기
        common_restaurants = set(svd_scores_int.index).intersection(set(content_scores.index))
        
        logger.info(f"SVD 식당 수: {len(svd_scores)}")
        logger.info(f"콘텐츠 식당 수: {len(content_scores)}")
        logger.info(f"공통 식당 수: {len(common_restaurants)}")
        
        if len(common_restaurants) == 0:
            logger.error("공통 식당이 없음")
            return []
        
        # === 핵심 수정: restaurants.csv에 있는 식당만 필터링 ===
        valid_restaurants = [rid for rid in common_restaurants if rid in self.restaurant_info]
        invalid_count = len(common_restaurants) - len(valid_restaurants)
        
        logger.info(f"restaurants.csv에 있는 유효한 식당: {len(valid_restaurants)}개")
        if invalid_count > 0:
            logger.info(f"restaurants.csv에 없는 식당 {invalid_count}개 제외됨")
        
        if len(valid_restaurants) == 0:
            logger.warning("유효한 식당이 없음 (모든 식당이 restaurants.csv에 없음)")
            return []
        
        # 하이브리드 점수 계산 (유효한 식당만 대상)
        hybrid_scores = {}
        budget_excluded = 0
        
        for restaurant_id in valid_restaurants:
            svd_score = svd_scores_int[restaurant_id]
            content_score = content_scores[restaurant_id]
            
            # 가중 평균
            hybrid_score = alpha * svd_score + (1 - alpha) * content_score
            
            # 예산 필터링
            budget_ok = True
            if budget is not None:
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
        
        logger.info(f"예산 내 추천 가능 식당: {len(hybrid_scores)}개")
        if budget_excluded > 0:
            logger.info(f"예산 초과로 제외된 식당: {budget_excluded}개")
        
        # 예산 때문에 추천할 식당이 없으면 예산 제한 해제
        if len(hybrid_scores) == 0 and budget is not None:
            logger.warning("예산 필터링 후 추천 가능한 식당이 없음 - 예산 제한 해제하고 재시도")
            return self.get_hybrid_recommendations(user_id, budget=None, top_n=top_n, alpha=alpha)
        
        if len(hybrid_scores) == 0:
            logger.warning("최종적으로 추천 가능한 식당이 없음")
            return []
        
        # 점수 순 정렬
        sorted_restaurants = sorted(
            hybrid_scores.items(), 
            key=lambda x: x[1]['hybrid_score'], 
            reverse=True
        )[:top_n]
        
        # 결과 포맷팅 (이제 모든 식당이 유효함)
        recommendations = []
        for rest_id, scores in sorted_restaurants:
            restaurant_info = self.restaurant_info[rest_id]  # .get() 대신 직접 접근 (이미 검증됨)
            
            recommendations.append({
                'restaurant_id': rest_id,
                'restaurant_name': restaurant_info['name'],
                'category': restaurant_info['category'],
                'menu_average': restaurant_info['menu_average'],
                'hybrid_score': round(scores['hybrid_score'], 2),
                'svd_score': round(scores['svd_score'], 2),
                'content_score': round(scores['content_score'], 2)
            })
        
        logger.info(f"하이브리드 추천 완료: {len(recommendations)}개 식당 (모두 유효한 식당)")
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

    def get_recommendations_json(self, user_id, budget=None, top_n=10, alpha=0.7):
        """
        API 응답용 JSON 형태로 추천 결과 반환
        
        Returns:
            dict: JSON 형태의 추천 결과
        """
        recommendations = self.get_hybrid_recommendations(user_id, budget, top_n, alpha)
        
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


# API 서버용 추천 함수
# hybrid_recommender.py의 get_restaurant_recommendations 함수 수정

def get_restaurant_recommendations(user_id, budget=None, top_n=10):
    """
    API 서버에서 호출할 메인 추천 함수 (경로 수정)
    """
    import os
    
    # 파일 경로 수정: pipeline 폴더에서 실행되더라도 recommendSystem/result를 참조
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # pipeline 폴더에서 실행 중이라면 상위 폴더로 이동
    if 'pipeline' in current_dir:
        base_dir = os.path.dirname(current_dir)  # pipeline의 상위 폴더 (recommendSystem)
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
        
        # 추천 수행
        result = hybrid_system.get_recommendations_json(user_id, budget, top_n)
        
        return result
        
    except Exception as e:
        logger.error(f"추천 시스템 오류: {e}")
        # 오류 시에도 완전한 응답 구조 반환
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