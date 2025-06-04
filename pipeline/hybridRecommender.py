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
        """
        # 매핑 정보 먼저 로드
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        # 매핑 딕셔너리 설정
        if 'all_mappings' in mappings:
            self.survey_to_id = mappings['all_mappings']['survey_to_id']
            self.id_to_survey = mappings['all_mappings']['id_to_survey']
        else:
            self.survey_to_id = mappings['restaurant_to_id']
            self.id_to_survey = mappings['id_to_restaurant']
        
        logger.info(f"매핑 정보 로드: {len(self.survey_to_id)}개")
        
        # SVD++ 예측 매트릭스 로드
        self.svd_matrix = pd.read_csv(svd_matrix_path, index_col=0)
        
        # SVD 매트릭스 컬럼을 정수로 변환
        try:
            int_columns = []
            for col in self.svd_matrix.columns:
                try:
                    int_columns.append(int(col))
                except ValueError:
                    int_columns.append(col)
            
            self.svd_matrix.columns = int_columns
            logger.info(f"SVD 매트릭스 컬럼을 정수로 변환: {len(int_columns)}개")
            
        except Exception as e:
            logger.warning(f"SVD 매트릭스 컬럼 변환 실패: {e}")

        # 식당 정보 로드
        self.restaurants = pd.read_csv(restaurants_path)
        
        # 매핑 정보를 활용한 통합 식당 정보 딕셔너리 생성
        self._create_restaurant_info_dict()
        
        # 콘텐츠 기반 추천기 초기화
        self.content_recommender = ContentBasedRecommender(
            restaurants_path, content_features_path, mappings_path
        )
        
        # 콘텐츠 기반 매트릭스 초기화
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

    def _create_restaurant_info_dict(self):
        """
        실제 ID + 가상 ID 통합 식당 정보 딕셔너리 생성
        매핑 정보를 활용해 가상 ID에도 식당 정보 제공
        """
        self.restaurant_info = {}  # restaurant_info_dict → restaurant_info로 변경
        
        # 1단계: restaurants.csv의 실제 식당 정보 로드
        for _, restaurant in self.restaurants.iterrows():
            self.restaurant_info[restaurant['id']] = {
                'name': restaurant['name'],
                'category': restaurant['category'],
                'menu_average': restaurant['menu_average']
            }
        
        logger.info(f"실제 식당 정보 로드: {len(self.restaurant_info)}개")
        
        # 2단계: 매핑 정보를 활용해 가상 ID 정보 생성
        virtual_count = 0
        
        # survey_to_id에서 가상 ID들 찾기
        for survey_name, mapped_id in self.survey_to_id.items():
            # 가상 ID (0~1000 범위의 작은 숫자) 처리
            if mapped_id < 1000 and mapped_id not in self.restaurant_info:
                # 매핑된 실제 ID가 있는지 확인
                real_restaurant_info = None
                
                # 같은 설문명으로 실제 ID에 매핑된 식당이 있는지 찾기
                for other_survey, other_id in self.survey_to_id.items():
                    if (other_survey == survey_name and 
                        other_id > 1000 and 
                        other_id in self.restaurant_info):
                        real_restaurant_info = self.restaurant_info[other_id]
                        break
                
                # 실제 식당 정보가 있으면 사용, 없으면 기본값 생성
                if real_restaurant_info:
                    self.restaurant_info[mapped_id] = real_restaurant_info.copy()
                    logger.debug(f"가상 ID {mapped_id}: {survey_name} → 실제 식당 정보 복사")
                else:
                    # 설문명에서 카테고리 추정
                    estimated_category = self._estimate_category_from_name(survey_name)
                    
                    self.restaurant_info[mapped_id] = {
                        'name': survey_name,
                        'category': estimated_category,
                        'menu_average': 20000  # 기본 평균 가격
                    }
                    logger.debug(f"가상 ID {mapped_id}: {survey_name} → 새 정보 생성 ({estimated_category})")
                
                virtual_count += 1
        
        logger.info(f"가상 식당 정보 생성: {virtual_count}개")
        logger.info(f"통합 식당 정보 딕셔너리 완성: {len(self.restaurant_info)}개")
        
        # 추천에서 사용되는 ID들이 포함되어 있는지 확인
        sample_virtual_ids = [id for id in self.restaurant_info.keys() if id < 50]
        logger.info(f"가상 ID 샘플 (0~49): {sample_virtual_ids[:10]}")
    
        # 각 ID별 정보 확인
        for sample_id in sample_virtual_ids[:5]:
            info = self.restaurant_info[sample_id]
            logger.info(f"ID {sample_id}: {info['name']} ({info['category']}, {info['menu_average']}원)")

    def _estimate_category_from_name(self, restaurant_name):
        """
        식당명에서 카테고리 추정
        """
        name_lower = restaurant_name.lower()
        
        # 간단한 키워드 기반 카테고리 추정
        if any(keyword in name_lower for keyword in ['피자', 'pizza', '버거', 'burger', '스테이크', '파스타']):
            return '양식'
        elif any(keyword in name_lower for keyword in ['라멘', '우동', '돈까스', '카츠', '초밥', '일식']):
            return '일식'
        elif any(keyword in name_lower for keyword in ['짜장', '짬뽕', '탕수육', '중식']):
            return '중식'
        elif any(keyword in name_lower for keyword in ['떡볶이', '김밥', '분식']):
            return '분식'
        elif any(keyword in name_lower for keyword in ['쌀국수', '팟타이', '아시안']):
            return '아시안'
        else:
            return '한식'  # 기본값

    def get_hybrid_recommendations(self, user_id, user_categories=None, budget=None, top_n=10, alpha=0.7, category_boost=0.7):
        """
        하이브리드 추천 수행
        
        Args:
            user_id: 사용자 ID
            user_categories: 사용자 선호 카테고리 리스트
            budget: 예산 제한 (None이면 제한 없음)
            top_n: 추천할 식당 수
            alpha: SVD++ 가중치 (0.7이면 SVD++ 70%, 콘텐츠 30%)
            category_boost: 선호 카테고리 가중치 (0.7 = 70% 추가 점수)
            
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

        # 6. 예산별 식당 분류 및 점수 계산 
        budget_within = {}  # 예산 내 식당
        budget_over = {}    # 예산 초과 식당
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
            
            # 예산별 분류
            if budget is not None and restaurant_id in self.restaurant_info:
                menu_price = self.restaurant_info[restaurant_id]['menu_average']
                
                if menu_price <= budget:
                    # 예산 내 식당
                    budget_within[restaurant_id] = {
                        'hybrid_score': hybrid_score,
                        'svd_score': svd_score,
                        'content_score': content_score,
                        'menu_price': menu_price,
                        'budget_status': 'within'
                    }
                else:
                    # 예산 초과 식당 - 페널티 적용
                    budget_excess = menu_price - budget
                    budget_ratio = menu_price / budget
                    
                    # 페널티 계산 (초과 정도에 따라 차등)
                    if budget_ratio <= 1.2:  # 20% 이하 초과
                        penalty = 0.3
                    elif budget_ratio <= 1.5:  # 50% 이하 초과
                        penalty = 0.5
                    else:  # 50% 초과
                        penalty = 0.7
                    
                    penalized_score = hybrid_score * (1 - penalty)
                    
                    budget_over[restaurant_id] = {
                        'hybrid_score': penalized_score,
                        'original_score': hybrid_score,
                        'svd_score': svd_score,
                        'content_score': content_score,
                        'menu_price': menu_price,
                        'budget_status': 'over',
                        'budget_excess': budget_excess,
                        'penalty_applied': penalty
                    }
            else:
                # 예산 제한 없음 또는 가격 정보 없음
                budget_within[restaurant_id] = {
                    'hybrid_score': hybrid_score,
                    'svd_score': svd_score,
                    'content_score': content_score,
                    'menu_price': self.restaurant_info.get(restaurant_id, {}).get('menu_average', 0),
                    'budget_status': 'no_limit'
                }
        
        # 7. 우선순위 기반 추천 목록 생성
        final_recommendations = self._create_priority_based_recommendations(
            budget_within, budget_over, top_n
        )
        
        # 로깅
        logger.info(f"예산 내 식당: {len(budget_within)}개, 예산 초과 식당: {len(budget_over)}개")
        logger.info(f"최종 추천: {len(final_recommendations)}개")
        
        return final_recommendations
        
    def _fallback_recommendation(self, user_id, budget, top_n):
        """
        공통 식당이 없을 때 SVD++만으로 추천 (대안책)
        """
        logger.info("대안책: SVD++만으로 추천 진행")
        
        svd_scores = self.svd_matrix.loc[user_id]
        budget_within = {}
        budget_over = {}
        
        for restaurant_id, score in svd_scores.items():
            rest_id = int(restaurant_id) if str(restaurant_id).isdigit() else restaurant_id
            
            # 예산별 분류
            if budget is not None and rest_id in self.restaurant_info:
                menu_price = self.restaurant_info[rest_id]['menu_average']
                if menu_price <= budget:
                    budget_within[rest_id] = {
                        'hybrid_score': score,
                        'svd_score': score,
                        'content_score': 0.0,
                        'menu_price': menu_price,
                        'budget_status': 'within'
                    }
                else:
                    # 예산 초과 식당 페널티 적용
                    budget_ratio = menu_price / budget
                    penalty = 0.1 if budget_ratio <= 1.2 else (0.3 if budget_ratio <= 1.5 else 0.5)
                    penalized_score = score * (1 - penalty)
                    
                    budget_over[rest_id] = {
                        'hybrid_score': penalized_score,
                        'original_score': score,
                        'svd_score': score,
                        'content_score': 0.0,
                        'menu_price': menu_price,
                        'budget_status': 'over',
                        'budget_excess': menu_price - budget,
                        'penalty_applied': penalty
                    }
            else:
                budget_within[rest_id] = {
                    'hybrid_score': score,
                    'svd_score': score,
                    'content_score': 0.0,
                    'menu_price': self.restaurant_info.get(rest_id, {}).get('menu_average', 0),
                    'budget_status': 'no_limit'
                }
        
        # 우선순위 기반 추천 생성
        recommendations = self._create_priority_based_recommendations(
            budget_within, budget_over, top_n
        )
        
        logger.warning(f"대안책으로 {len(recommendations)}개 식당 추천")
        return recommendations

    def _create_priority_based_recommendations(self, budget_within, budget_over, top_n):
        """
        예산 내 식당을 우선하되, 부족할 경우 예산 초과 식당도 포함
        
        Args:
            budget_within: 예산 내 식당 딕셔너리
            budget_over: 예산 초과 식당 딕셔너리
            top_n: 목표 추천 개수
        
        Returns:
            list: 우선순위 기반 추천 목록
        """
        recommendations = []
        
        # 1단계: 예산 내 식당 우선 추천
        within_sorted = sorted(
            budget_within.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )
        
        # 예산 내 식당으로 추천 목록 채우기
        for rest_id, scores in within_sorted[:top_n]:
            restaurant_info = self.restaurant_info.get(rest_id, {})
            
            recommendations.append({
                'restaurant_id': rest_id,
                'restaurant_name': restaurant_info.get('name', 'Unknown'),
                'category': restaurant_info.get('category', 'Unknown'),
                'menu_average': scores['menu_price'],
                'hybrid_score': round(scores['hybrid_score'], 2),
                'svd_score': round(scores['svd_score'], 2),
                'content_score': round(scores['content_score'], 2),
                'budget_status': scores['budget_status']
            })
        
        # 2단계: 부족하면 예산 초과 식당 추가
        if len(recommendations) < top_n and budget_over:
            remaining_slots = top_n - len(recommendations)
            
            # 예산 초과 식당을 페널티 적용된 점수로 정렬
            over_sorted = sorted(
                budget_over.items(),
                key=lambda x: x[1]['hybrid_score'],  # 이미 페널티 적용된 점수
                reverse=True
            )
            
            for rest_id, scores in over_sorted[:remaining_slots]:
                restaurant_info = self.restaurant_info.get(rest_id, {})
                
                recommendations.append({
                    'restaurant_id': rest_id,
                    'restaurant_name': restaurant_info.get('name', 'Unknown'),
                    'category': restaurant_info.get('category', 'Unknown'),
                    'menu_average': scores['menu_price'],
                    'hybrid_score': round(scores['hybrid_score'], 2),
                    'original_score': round(scores['original_score'], 2),
                    'svd_score': round(scores['svd_score'], 2),
                    'content_score': round(scores['content_score'], 2),
                    'budget_status': scores['budget_status'],
                    'budget_excess': scores['budget_excess'],
                    'penalty_applied': scores['penalty_applied']
                })
            
            logger.info(f"예산 부족으로 초과 식당 {remaining_slots}개 추가 (페널티 적용)")
        
        return recommendations

    def get_recommendations_json(self, user_id, user_categories=None, budget=None, top_n=10, alpha=0.7):
        """
        API 응답용 JSON 형태로 추천 결과 반환
        
        Returns:
            dict: JSON 형태의 추천 결과
        """
        try:
            recommendations = self.get_hybrid_recommendations(user_id, user_categories, budget, top_n, alpha)
            
            logger.info(f"JSON 변환 전 추천 수: {len(recommendations) if recommendations else 0}")
            
            # 안전한 데이터 변환
            recommendation_list = []
            if recommendations:
                for rec in recommendations:
                    try:
                        recommendation_list.append({
                            'restaurant_id': int(rec['restaurant_id']),  # 정수 보장
                            'restaurant_name': str(rec.get('restaurant_name', 'Unknown')),
                            'category': str(rec.get('category', 'Unknown')),
                            'menu_average': float(rec.get('menu_average', 0)),
                            'predicted_rating': float(rec.get('hybrid_score', 0))
                        })
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"추천 데이터 변환 실패: {rec}, 오류: {e}")
                        continue
            
            result = {
                'user_id': int(user_id),
                'recommendations': recommendation_list,
                'recommendation_count': len(recommendation_list),
                'algorithm': f"Hybrid (SVD++: {alpha*100:.1f}%, Content: {(1-alpha)*100:.1f}%)"
            }
            
            logger.info(f"JSON 변환 완료 - 최종 추천 수: {len(recommendation_list)}")
            return result
            
        except Exception as e:
            logger.error(f"JSON 변환 중 오류: {e}")
            return {
                'user_id': int(user_id) if user_id else 0,
                'recommendations': [],
                'recommendation_count': 0,
                'algorithm': 'Hybrid (Error)',
                'error': str(e)
            }

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
    
    # 파일 존재 여부 확인
    required_files = {
        'SVD 매트릭스': svd_matrix_path,
        '식당 데이터': restaurants_path,
        '콘텐츠 특성': content_features_path,
        '매핑 파일': mappings_path,
        'SVD 데이터': svd_data_path
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(name)
            logger.error(f"누락된 파일: {name} - {path}")
    
    if missing_files:
        error_msg = f"필수 파일 누락: {', '.join(missing_files)}"
        logger.error(error_msg)
        return {
            'user_id': user_id,
            'recommendations': [],
            'recommendation_count': 0,
            'algorithm': 'Hybrid (File Missing)',
            'error': error_msg
        }
    
    try:
        # 동적 가중치 관리자 초기화
        from .adaptive_weights import AdaptiveWeightManager
        weight_manager = AdaptiveWeightManager()
        
        # 사용자별 최적화된 가중치 가져오기
        user_weights = weight_manager.get_user_weights(user_id)
        adaptive_alpha = user_weights['alpha']
        adaptive_category_boost = user_weights['category_boost']
        
        logger.info(f"사용자 {user_id} 적응형 가중치: α={adaptive_alpha:.3f}, boost={adaptive_category_boost:.3f}")
        
        # 하이브리드 시스템 초기화
        hybrid_system = HybridRecommendationSystem(
            svd_matrix_path, restaurants_path, content_features_path, mappings_path
        )
        
        # 콘텐츠 매트릭스 준비
        hybrid_system.prepare_content_matrix(svd_data_path)
        
        # 동적 가중치로 추천 수행
        result = hybrid_system.get_recommendations_json(
            user_id, user_categories, budget, top_n, 
            alpha=adaptive_alpha  # 동적 가중치 사용
        )
        
        logger.info(f"적응형 추천 완료 - {result.get('recommendation_count', 0)}개 추천")
        return result
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"추천 시스템 오류: {e}")
        logger.error(f"상세 오류:\n{error_detail}")
        
        return {
            'user_id': user_id,
            'recommendations': [],
            'recommendation_count': 0,
            'algorithm': 'Hybrid (Error)',
            'error': str(e),
            'error_detail': error_detail
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