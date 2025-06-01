import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

class RecommendationComparison:
    """
    두 가지 콘텐츠 기반 필터링 방법의 정확도 비교
    """
    
    def __init__(self, svd_data_path, prediction_matrix_path, restaurants_path, content_features_path, mappings_path):
        """
        데이터 로드 및 초기화
        """
        # 데이터 로드
        self.svd_data = pd.read_csv(svd_data_path)
        self.prediction_matrix = pd.read_csv(prediction_matrix_path, index_col=0)
        self.restaurants = pd.read_csv(restaurants_path)
        self.content_features = pd.read_csv(content_features_path)
        
        # 매핑 정보 로드
        import pickle
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        self.restaurant_to_id = mappings['restaurant_to_id']  # 식당명 → 연속번호
        self.id_to_restaurant = mappings['id_to_restaurant']  # 연속번호 → 식당명
        
        # 식당 정보 딕셔너리 생성 (실제 ID 기준)
        self.restaurant_info = self.restaurants.set_index('id').to_dict('index')
        
        # 연속번호 → 실제ID 매핑 생성
        self.sequential_to_real_id = {}
        for sequential_id, restaurant_name in self.id_to_restaurant.items():
            # 식당명으로 실제 ID 찾기
            matching_restaurants = self.restaurants[self.restaurants['name'] == restaurant_name]
            if len(matching_restaurants) > 0:
                real_id = matching_restaurants.iloc[0]['id']
                self.sequential_to_real_id[sequential_id] = real_id
        
        # 데이터 매칭 확인
        svd_restaurant_ids = set(self.svd_data['restaurantId'].unique())
        real_restaurant_ids = set(self.restaurant_info.keys())
        
        # 매핑 가능한 ID들
        mappable_svd_ids = set(self.sequential_to_real_id.keys())
        mapped_real_ids = set(self.sequential_to_real_id.values())
        
        matched_ids = svd_restaurant_ids.intersection(mappable_svd_ids)
        
        print(f"데이터 로드 완료:")
        print(f"- 평점 데이터: {len(self.svd_data)}개")
        print(f"- 사용자 수: {self.svd_data['userId'].nunique()}명")
        print(f"- 식당 수: {len(self.restaurants)}개")
        print(f"\n매핑 정보:")
        print(f"- 식당명→연속번호 매핑: {len(self.restaurant_to_id)}개")
        print(f"- 연속번호→실제ID 매핑: {len(self.sequential_to_real_id)}개")
        print(f"\n데이터 매칭 현황:")
        print(f"- SVD 데이터 식당 수: {len(svd_restaurant_ids)}개")
        print(f"- 실제 식당 정보 수: {len(real_restaurant_ids)}개") 
        print(f"- 매핑 가능한 식당: {len(matched_ids)}개")
        print(f"- 매칭률: {len(matched_ids)/len(svd_restaurant_ids)*100:.1f}%")
        
        if len(matched_ids) > 0:
            print(f"매핑 예시:")
            sample_sequential_id = list(matched_ids)[0]
            sample_real_id = self.sequential_to_real_id[sample_sequential_id]
            sample_name = self.id_to_restaurant[sample_sequential_id]
            print(f"  연속번호 {sample_sequential_id} → 식당명 '{sample_name}' → 실제ID {sample_real_id}")
        
        # 매칭률이 낮으면 경고
        match_rate = len(matched_ids) / len(svd_restaurant_ids) if svd_restaurant_ids else 0
        if match_rate < 0.8:
            print(f"⚠️ 경고: 매칭률이 낮습니다 ({match_rate:.1%}). 데이터를 확인하세요.")
        else:
            print(f"✅ 매핑 성공!")
        
    def prepare_content_similarity_matrix(self):
        """
        방법 1: 식당×식당 콘텐츠 유사도 매트릭스 생성
        """
        print("\n=== 방법 1: 식당×식당 콘텐츠 유사도 매트릭스 생성 ===")
        
        # 콘텐츠 특성에서 수치형 특성만 추출
        feature_columns = [col for col in self.content_features.columns 
                          if col not in ['id', 'name']]
        
        content_matrix = self.content_features[feature_columns].values
        
        # 표준화
        scaler = StandardScaler()
        content_matrix_scaled = scaler.fit_transform(content_matrix)
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(content_matrix_scaled)
        
        # 실제 ID를 연속번호로 매핑하는 딕셔너리 생성
        real_to_sequential_id = {v: k for k, v in self.sequential_to_real_id.items()}
        
        # content_features의 실제 ID를 연속번호로 변환
        content_real_ids = self.content_features['id'].values
        sequential_ids = []
        
        for real_id in content_real_ids:
            if real_id in real_to_sequential_id:
                sequential_ids.append(real_to_sequential_id[real_id])
            else:
                sequential_ids.append(-1)  # 매칭되지 않는 경우
        
        # 매칭된 식당들만 필터링
        valid_indices = [i for i, seq_id in enumerate(sequential_ids) if seq_id != -1]
        valid_sequential_ids = [sequential_ids[i] for i in valid_indices]
        
        if len(valid_indices) == 0:
            print("⚠️ 경고: 콘텐츠 특성과 SVD 데이터 간 매칭되는 식당이 없습니다.")
            # 빈 매트릭스 생성
            self.content_similarity_df = pd.DataFrame()
            return self.content_similarity_df
        
        # 유효한 식당들만으로 유사도 매트릭스 생성
        filtered_similarity_matrix = similarity_matrix[np.ix_(valid_indices, valid_indices)]
        
        # 데이터프레임으로 변환 (연속번호 기준)
        self.content_similarity_df = pd.DataFrame(
            filtered_similarity_matrix, 
            index=valid_sequential_ids, 
            columns=valid_sequential_ids
        )
        
        print(f"콘텐츠 유사도 매트릭스 생성 완료: {filtered_similarity_matrix.shape}")
        print(f"매칭된 식당 수: {len(valid_sequential_ids)}개")
        print(f"연속번호 범위: {min(valid_sequential_ids)} ~ {max(valid_sequential_ids)}")
        
        return self.content_similarity_df
    
    def generate_user_profiles(self):
        """
        방법 2: 사용자 프로필 생성 (선호 카테고리, 가격대 기반)
        """
        print("\n=== 방법 2: 사용자 프로필 생성 ===")
        
        user_profiles = {}
        debug_count = 0
        
        for user_id in self.svd_data['userId'].unique():
            # 사용자가 평가한 식당들
            user_ratings = self.svd_data[self.svd_data['userId'] == user_id]
            
            # 높게 평가한 식당들 (4점 이상)
            high_rated = user_ratings[user_ratings['rating'] >= 4.0]
            
            # 선호 카테고리 추출
            preferred_categories = []
            total_spending = 0
            count = 0
            matched_restaurants = 0
            
            for _, rating in high_rated.iterrows():
                sequential_restaurant_id = rating['restaurantId']
                
                # 연속번호를 실제 ID로 변환
                if sequential_restaurant_id in self.sequential_to_real_id:
                    real_restaurant_id = self.sequential_to_real_id[sequential_restaurant_id]
                    
                    if real_restaurant_id in self.restaurant_info:
                        restaurant = self.restaurant_info[real_restaurant_id]
                        preferred_categories.append(restaurant['category'])
                        total_spending += restaurant['menu_average']
                        count += 1
                        matched_restaurants += 1
            
            # 디버깅 정보 (처음 3명만)
            if debug_count < 3:
                print(f"  사용자 {user_id}: 평가한 식당 {len(user_ratings)}개, 높은 평가 {len(high_rated)}개, 매칭된 식당 {matched_restaurants}개")
                debug_count += 1
            
            # 가장 많이 선호하는 카테고리
            if preferred_categories:
                from collections import Counter
                category_counts = Counter(preferred_categories)
                top_categories = [cat for cat, _ in category_counts.most_common(3)]
            else:
                # 기본값으로 전체 평가에서 카테고리 추출
                all_categories = []
                for _, rating in user_ratings.iterrows():
                    sequential_restaurant_id = rating['restaurantId']
                    
                    if sequential_restaurant_id in self.sequential_to_real_id:
                        real_restaurant_id = self.sequential_to_real_id[sequential_restaurant_id]
                        
                        if real_restaurant_id in self.restaurant_info:
                            all_categories.append(self.restaurant_info[real_restaurant_id]['category'])
                
                if all_categories:
                    from collections import Counter
                    category_counts = Counter(all_categories)
                    top_categories = [cat for cat, _ in category_counts.most_common(2)]
                else:
                    top_categories = ['한식']  # 최종 기본값
            
            # 평균 지출 금액 (전체 평가 기준으로 확장)
            if count == 0:
                # 높게 평가한 식당이 없으면 전체 평가에서 계산
                for _, rating in user_ratings.iterrows():
                    sequential_restaurant_id = rating['restaurantId']
                    
                    if sequential_restaurant_id in self.sequential_to_real_id:
                        real_restaurant_id = self.sequential_to_real_id[sequential_restaurant_id]
                        
                        if real_restaurant_id in self.restaurant_info:
                            total_spending += self.restaurant_info[real_restaurant_id]['menu_average']
                            count += 1
            
            avg_budget = total_spending / count if count > 0 else 20000
            
            user_profiles[user_id] = {
                'preferred_categories': top_categories,
                'avg_budget': avg_budget,
                'price_sensitivity': 0.7  # 기본값
            }
        
        # 전체 통계 출력
        total_matched = sum(1 for profile in user_profiles.values() if profile['avg_budget'] != 20000)
        print(f"사용자 프로필 생성 완료: {len(user_profiles)}명 (실제 데이터 기반: {total_matched}명)")
        
        # 샘플 프로필 출력
        sample_users = list(user_profiles.keys())[:3]
        for user_id in sample_users:
            profile = user_profiles[user_id]
            print(f"  사용자 {user_id}: 선호카테고리 {profile['preferred_categories']}, 평균예산 {profile['avg_budget']:.0f}원")
        
        self.user_profiles = user_profiles
        return user_profiles
    
    def method1_content_based_prediction(self, user_id, target_restaurant_id):
        """
        방법 1: 식당×식당 유사도 기반 예측
        """
        # 사용자가 평가한 식당들
        user_ratings = self.svd_data[self.svd_data['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return 2.5  # 기본값
        
        weighted_sum = 0
        similarity_sum = 0
        valid_similarities = 0
        
        for _, rating in user_ratings.iterrows():
            rated_restaurant_id = rating['restaurantId']
            user_rating = rating['rating']
            
            # 두 식당 간 유사도 확인
            if (rated_restaurant_id in self.content_similarity_df.index and 
                target_restaurant_id in self.content_similarity_df.columns):
                
                similarity = self.content_similarity_df.loc[rated_restaurant_id, target_restaurant_id]
                
                # 자기 자신과의 유사도(1.0) 제외하고 계산
                if rated_restaurant_id != target_restaurant_id and similarity > 0.1:
                    weighted_sum += similarity * user_rating
                    similarity_sum += similarity
                    valid_similarities += 1
        
        # 디버깅 정보 출력 (처음 몇 개만)
        if hasattr(self, '_debug_count') and self._debug_count < 3:
            print(f"  [방법1] 사용자 {user_id}, 식당 {target_restaurant_id}")
            print(f"    평가한 식당 수: {len(user_ratings)}")
            print(f"    유효한 유사도: {valid_similarities}")
            print(f"    가중합: {weighted_sum:.3f}, 유사도합: {similarity_sum:.3f}")
            self._debug_count += 1
        
        if similarity_sum > 0 and valid_similarities > 0:
            prediction = weighted_sum / similarity_sum
            return max(1.0, min(5.0, prediction))  # 1-5 범위로 제한
        else:
            # 전체 평균으로 대체
            return user_ratings['rating'].mean() if len(user_ratings) > 0 else 2.5
    
    def method2_user_profile_prediction(self, user_id, target_restaurant_id):
        """
        방법 2: 사용자 프로필 기반 예측
        """
        # 사용자 프로필 확인
        if user_id not in self.user_profiles:
            if hasattr(self, '_debug_count2') and self._debug_count2 < 3:
                print(f"  [방법2 오류] 사용자 {user_id}의 프로필이 없음")
                self._debug_count2 += 1
            return 2.5  # 기본값
        
        user_profile = self.user_profiles[user_id]
        
        # 연속번호를 실제 ID로 변환
        if target_restaurant_id in self.sequential_to_real_id:
            real_restaurant_id = self.sequential_to_real_id[target_restaurant_id]
        else:
            if hasattr(self, '_debug_count2') and self._debug_count2 < 3:
                print(f"  [방법2 오류] 연속번호 {target_restaurant_id}를 실제 ID로 변환할 수 없음")
                self._debug_count2 += 1
            return 2.5  # 기본값
        
        # 실제 ID로 식당 정보 확인
        if real_restaurant_id not in self.restaurant_info:
            if hasattr(self, '_debug_count2') and self._debug_count2 < 3:
                print(f"  [방법2 오류] 실제 식당 ID {real_restaurant_id} 정보가 없음")
                self._debug_count2 += 1
            return 2.5  # 기본값
        
        restaurant = self.restaurant_info[real_restaurant_id]
        
        # 카테고리 점수 (0-1)
        if restaurant['category'] in user_profile['preferred_categories']:
            category_score = 1.0
        else:
            category_score = 0.3
        
        # 가격 점수 (0-1) - 더 관대한 계산
        price_diff = abs(restaurant['menu_average'] - user_profile['avg_budget'])
        max_price_diff = user_profile['avg_budget'] * 0.8  # 예산의 80% 차이까지 허용
        
        if price_diff <= max_price_diff:
            price_score = 1.0 - (price_diff / max_price_diff) * 0.3  # 최소 0.7점
        else:
            price_score = 0.7 * np.exp(-(price_diff - max_price_diff) / max_price_diff)
        
        # 최종 콘텐츠 점수 (1-5 스케일로 변환)
        content_score = (category_score * 0.6 + price_score * 0.4) * 3 + 2  # 2-5점 범위
        
        # 디버깅 정보 출력 (처음 몇 개만)
        if hasattr(self, '_debug_count2') and self._debug_count2 < 3:
            print(f"  [방법2] 사용자 {user_id}, 연속번호 {target_restaurant_id} → 실제ID {real_restaurant_id}")
            print(f"    식당명: {self.id_to_restaurant.get(target_restaurant_id, 'Unknown')}")
            print(f"    선호 카테고리: {user_profile['preferred_categories']}")
            print(f"    식당 카테고리: {restaurant['category']} (매치: {restaurant['category'] in user_profile['preferred_categories']})")
            print(f"    평균 예산: {user_profile['avg_budget']:.0f}, 식당 가격: {restaurant['menu_average']}")
            print(f"    가격 차이: {price_diff:.0f}, 허용 범위: {max_price_diff:.0f}")
            print(f"    카테고리 점수: {category_score:.2f}, 가격 점수: {price_score:.2f}")
            print(f"    최종 점수: {content_score:.2f}")
            self._debug_count2 += 1
        
        final_score = min(5.0, max(1.0, content_score))
        return final_score
    
    def hybrid_prediction(self, method, user_id, target_restaurant_id, alpha=0.7):
        """
        하이브리드 예측 (SVD++ + 콘텐츠 기반)
        alpha: SVD++ 가중치
        """
        # SVD++ 예측값 - 컬럼명을 정수와 문자열 모두 시도
        svd_score = 2.5  # 기본값
        
        if user_id in self.prediction_matrix.index:
            # 정수 컬럼명 시도
            if target_restaurant_id in self.prediction_matrix.columns:
                svd_score = self.prediction_matrix.loc[user_id, target_restaurant_id]
            # 문자열 컬럼명 시도
            elif str(target_restaurant_id) in self.prediction_matrix.columns:
                svd_score = self.prediction_matrix.loc[user_id, str(target_restaurant_id)]
        
        # 콘텐츠 기반 예측값
        if method == 1:
            content_score = self.method1_content_based_prediction(user_id, target_restaurant_id)
        else:
            content_score = self.method2_user_profile_prediction(user_id, target_restaurant_id)
        
        # 하이브리드 결합
        hybrid_score = alpha * svd_score + (1 - alpha) * content_score
        
        # 디버깅 정보 (처음 몇 개만)
        if hasattr(self, '_debug_count3') and self._debug_count3 < 3:
            print(f"  [하이브리드] 방법{method}, 사용자 {user_id}, 식당 {target_restaurant_id}")
            print(f"    SVD++ 점수: {svd_score:.2f}")
            print(f"    콘텐츠 점수: {content_score:.2f}")
            print(f"    하이브리드 점수: {hybrid_score:.2f} (alpha={alpha})")
            self._debug_count3 += 1
        
        return max(1.0, min(5.0, hybrid_score))  # 1-5 범위로 제한
    
    def evaluate_methods(self, test_size=0.2, alpha=0.7):
        """
        두 방법의 정확도 평가
        """
        print(f"\n=== 정확도 평가 시작 (alpha={alpha}) ===")
        
        # 디버깅 카운터 초기화
        self._debug_count = 0
        self._debug_count2 = 0
        self._debug_count3 = 0
        
        # 1. 콘텐츠 유사도 매트릭스 준비
        self.prepare_content_similarity_matrix()
        
        # 2. 사용자 프로필 생성
        self.generate_user_profiles()
        
        # 3. 훈련/테스트 데이터 분할
        train_data, test_data = train_test_split(
            self.svd_data, test_size=test_size, random_state=42
        )
        
        print(f"훈련 데이터: {len(train_data)}개, 테스트 데이터: {len(test_data)}개")
        
        # 평가 결과 저장
        results = {
            'method1': {'predictions': [], 'actuals': []},
            'method2': {'predictions': [], 'actuals': []}
        }
        
        print("예측 진행 중...")
        for i, (_, row) in enumerate(test_data.iterrows()):
            if i % 10 == 0:
                print(f"  진행률: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
            
            user_id = row['userId']
            restaurant_id = row['restaurantId']
            actual_rating = row['rating']
            
            # 방법 1 예측
            pred1 = self.hybrid_prediction(1, user_id, restaurant_id, alpha)
            results['method1']['predictions'].append(pred1)
            results['method1']['actuals'].append(actual_rating)
            
            # 방법 2 예측
            pred2 = self.hybrid_prediction(2, user_id, restaurant_id, alpha)
            results['method2']['predictions'].append(pred2)
            results['method2']['actuals'].append(actual_rating)
        
        # 평가 메트릭 계산
        print("\n=== 평가 결과 ===")
        
        for method_name, data in results.items():
            predictions = np.array(data['predictions'])
            actuals = np.array(data['actuals'])
            
            # RMSE 계산
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            
            # MAE 계산
            mae = np.mean(np.abs(predictions - actuals))
            
            # 상세 통계
            print(f"\n{method_name.upper()} 결과:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  예측 범위: [{predictions.min():.2f}, {predictions.max():.2f}]")
            print(f"  예측 평균: {predictions.mean():.2f}")
            print(f"  실제 범위: [{actuals.min():.2f}, {actuals.max():.2f}]")
            print(f"  실제 평균: {actuals.mean():.2f}")
        
        return results
    
    def alpha_sensitivity_analysis(self, alpha_values=[0.3, 0.5, 0.7, 0.9]):
        """
        Alpha 값에 따른 성능 변화 분석
        """
        print("\n=== Alpha 민감도 분석 ===")
        
        # 콘텐츠 유사도 매트릭스와 사용자 프로필 미리 준비
        self.prepare_content_similarity_matrix()
        self.generate_user_profiles()
        
        # 테스트 데이터 준비
        _, test_data = train_test_split(self.svd_data, test_size=0.3, random_state=42)
        
        # 샘플 크기를 데이터 크기에 맞게 조정
        sample_size = min(len(test_data), 50)  # 최대 50개, 데이터가 적으면 전체 사용
        test_sample = test_data.sample(sample_size, random_state=42) if len(test_data) > sample_size else test_data
        
        print(f"Alpha 분석용 테스트 샘플: {len(test_sample)}개")
        
        results_summary = []
        
        for alpha in alpha_values:
            print(f"\nAlpha = {alpha} 평가 중...")
            
            method1_errors = []
            method2_errors = []
            
            for _, row in test_sample.iterrows():
                user_id = row['userId']
                restaurant_id = row['restaurantId']
                actual_rating = row['rating']
                
                # 방법 1 예측
                pred1 = self.hybrid_prediction(1, user_id, restaurant_id, alpha)
                method1_errors.append((pred1 - actual_rating) ** 2)
                
                # 방법 2 예측
                pred2 = self.hybrid_prediction(2, user_id, restaurant_id, alpha)
                method2_errors.append((pred2 - actual_rating) ** 2)
            
            # RMSE 계산
            method1_rmse = np.sqrt(np.mean(method1_errors))
            method2_rmse = np.sqrt(np.mean(method2_errors))
            
            results_summary.append({
                'alpha': alpha,
                'method1_rmse': method1_rmse,
                'method2_rmse': method2_rmse,
                'better_method': 'Method1' if method1_rmse < method2_rmse else 'Method2',
                'rmse_diff': abs(method1_rmse - method2_rmse)
            })
            
            print(f"  방법1 RMSE: {method1_rmse:.4f}")
            print(f"  방법2 RMSE: {method2_rmse:.4f}")
            print(f"  더 좋은 방법: {'방법1' if method1_rmse < method2_rmse else '방법2'}")
        
        # 결과 요약
        print("\n=== Alpha 민감도 분석 결과 요약 ===")
        for result in results_summary:
            print(f"Alpha {result['alpha']}: {result['better_method']} 승 "
                  f"(차이: {result['rmse_diff']:.4f})")
        
        return results_summary

    def simple_evaluation(self):
        """
        간단한 평가 - 콘텐츠 기반 필터링만 테스트
        """
        print("\n=== 간단한 평가 (콘텐츠 기반만) ===")
        
        # 디버깅 카운터 초기화
        self._debug_count = 0
        self._debug_count2 = 0
        
        # 준비
        self.prepare_content_similarity_matrix()
        self.generate_user_profiles()
        
        # 작은 샘플로 테스트
        test_sample = self.svd_data.sample(min(10, len(self.svd_data)), random_state=42)
        
        method1_predictions = []
        method2_predictions = []
        actuals = []
        
        print(f"\n콘텐츠 기반 예측 테스트 ({len(test_sample)}개 샘플):")
        for i, (_, row) in enumerate(test_sample.iterrows()):
            user_id = row['userId']
            restaurant_id = row['restaurantId']
            actual_rating = row['rating']
            
            print(f"\n--- 테스트 {i+1} ---")
            print(f"사용자 {user_id} → 식당 {restaurant_id} (실제 평점: {actual_rating})")
            
            # 방법 1: 식당×식당 유사도
            pred1 = self.method1_content_based_prediction(user_id, restaurant_id)
            method1_predictions.append(pred1)
            
            # 방법 2: 사용자 프로필
            pred2 = self.method2_user_profile_prediction(user_id, restaurant_id)
            method2_predictions.append(pred2)
            
            actuals.append(actual_rating)
            
            print(f"예측 결과: 방법1={pred1:.2f}, 방법2={pred2:.2f}")
        
        # 결과 출력
        method1_predictions = np.array(method1_predictions)
        method2_predictions = np.array(method2_predictions)
        actuals = np.array(actuals)
        
        print(f"\n=== 종합 결과 ===")
        print(f"방법1 (식당×식당 유사도):")
        print(f"  예측 범위: [{method1_predictions.min():.2f}, {method1_predictions.max():.2f}]")
        print(f"  예측 평균: {method1_predictions.mean():.2f}")
        print(f"  RMSE: {np.sqrt(np.mean((method1_predictions - actuals) ** 2)):.4f}")
        
        print(f"\n방법2 (사용자 프로필):")
        print(f"  예측 범위: [{method2_predictions.min():.2f}, {method2_predictions.max():.2f}]")
        print(f"  예측 평균: {method2_predictions.mean():.2f}")
        print(f"  RMSE: {np.sqrt(np.mean((method2_predictions - actuals) ** 2)):.4f}")
        
        print(f"\n실제 데이터:")
        print(f"  실제 범위: [{actuals.min():.2f}, {actuals.max():.2f}]")
        print(f"  실제 평균: {actuals.mean():.2f}")
        
        # 기본값 문제 확인
        method2_default_count = sum(1 for pred in method2_predictions if abs(pred - 2.5) < 0.01)
        print(f"\n방법2에서 기본값(2.5) 반환 횟수: {method2_default_count}/{len(method2_predictions)}")
        
        return method1_predictions, method2_predictions, actuals


# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정 (실제 경로로 수정 필요)
    # 현재 스크립트의 디렉토리 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')  # 상위 폴더의 data 폴더
    result_dir = os.path.join(current_dir, '..', 'result')  # 상위 폴더의 result 폴더
    # data path
    restaurants_path = os.path.join(data_dir, 'restaurants.csv')
    content_features_path = os.path.join(data_dir, 'content_features.csv')
    # result path 
    svd_data_path=os.path.join(result_dir, 'svd_data.csv')
    prediction_matrix_path=os.path.join(result_dir, 'prediction_matrix.csv')
    mappings_path = os.path.join(result_dir,'restaurant_mappings.pkl')
    
    evaluator = RecommendationComparison(
        svd_data_path,
        prediction_matrix_path, 
        restaurants_path,
        content_features_path,
        mappings_path
    )
    # 간단한 평가 먼저 실행
    evaluator.simple_evaluation()
    
    
    # 기본 평가
    results = evaluator.evaluate_methods(test_size=0.2, alpha=0.7)
    
    # Alpha 민감도 분석
    sensitivity_results = evaluator.alpha_sensitivity_analysis()