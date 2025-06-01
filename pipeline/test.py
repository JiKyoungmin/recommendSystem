import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

class ContentFilteringComparison:
    """
    완전히 새로 작성한 콘텐츠 기반 필터링 비교 클래스
    방법1: 식당×식당 유사도 기반 추천
    방법2: 사용자-식당 프로필 코사인 유사도 기반 추천
    """
    
    def __init__(self, svd_data_path, restaurants_path, content_features_path, 
                 mappings_path, user_profiles_path=None):
        """
        데이터 로드 및 초기화 - return 문 완전 제거
        """
        print("=== 콘텐츠 기반 필터링 비교 시작 ===")
        
        # 기본 데이터 로드
        self.svd_data = pd.read_csv(svd_data_path)
        self.restaurants = pd.read_csv(restaurants_path)
        self.content_features = pd.read_csv(content_features_path)
        
        # 사용자 프로필 로드
        if user_profiles_path and os.path.exists(user_profiles_path):
            self.user_profiles_df = pd.read_csv(user_profiles_path)
            print(f"✅ 사용자 프로필: {len(self.user_profiles_df)}명")
        else:
            self.user_profiles_df = None
            print("⚠️ 사용자 프로필 없음")
        
        # 매핑 로드
        self._load_mappings(mappings_path)
        
        # 초기화 완료
        print(f"✅ 초기화 완료")
    
    def _load_mappings(self, mappings_path):
        """
        매핑 파일 로드 (내부 메서드)
        """
        print(f"📁 매핑 로드: {os.path.basename(mappings_path)}")
        
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        # 식당 정보 딕셔너리
        self.restaurant_info = self.restaurants.set_index('id').to_dict('index')
        
        # 매핑 딕셔너리 초기화
        self.sequential_to_real_id = {}
        self.id_to_restaurant = {}
        
        if 'all_mappings' in mappings:
            survey_to_id = mappings['all_mappings']['survey_to_id']
            id_to_survey = mappings['all_mappings']['id_to_survey']
            
            # 직접 매핑
            for survey_name, mapped_id in survey_to_id.items():
                self.sequential_to_real_id[mapped_id] = mapped_id
                
                # 가상 ID 처리
                if mapped_id < 1000 and mapped_id not in self.restaurant_info:
                    self.restaurant_info[mapped_id] = {
                        'name': survey_name,
                        'category': '한식',
                        'menu_average': 20000
                    }
            
            # id_to_survey 처리
            for mapped_id_str, survey_name in id_to_survey.items():
                try:
                    mapped_id = int(mapped_id_str)
                    self.id_to_restaurant[mapped_id] = survey_name
                except:
                    self.id_to_restaurant[mapped_id_str] = survey_name
        
        # 매칭 확인
        svd_ids = set(self.svd_data['restaurantId'].unique())
        matched = svd_ids.intersection(set(self.sequential_to_real_id.keys()))
        match_rate = len(matched) / len(svd_ids) if svd_ids else 0
        
        print(f"📊 매칭률: {match_rate*100:.1f}% ({len(matched)}/{len(svd_ids)})")
    
    def prepare_method1_similarity_matrix(self):
        """
        방법1: 식당×식당 유사도 매트릭스 생성
        """
        print("\n=== 방법1: 식당 유사도 매트릭스 ===")
        
        # 특성 추출
        feature_cols = [col for col in self.content_features.columns 
                       if col not in ['id', 'name']]
        content_matrix = self.content_features[feature_cols].values
        
        # 표준화 및 유사도 계산
        scaler = StandardScaler()
        content_scaled = scaler.fit_transform(content_matrix)
        similarity_matrix = cosine_similarity(content_scaled)
        
        # 실제 ID → 연속번호 매핑
        real_to_seq = {v: k for k, v in self.sequential_to_real_id.items()}
        content_real_ids = self.content_features['id'].values
        
        # 유효한 인덱스 필터링
        valid_indices = []
        valid_seq_ids = []
        
        for i, real_id in enumerate(content_real_ids):
            if real_id in real_to_seq:
                valid_indices.append(i)
                valid_seq_ids.append(real_to_seq[real_id])
        
        if len(valid_indices) == 0:
            print("⚠️ 매칭되는 식당이 없습니다")
            self.similarity_matrix = pd.DataFrame()
        else:
            filtered_sim = similarity_matrix[np.ix_(valid_indices, valid_indices)]
            self.similarity_matrix = pd.DataFrame(
                filtered_sim, index=valid_seq_ids, columns=valid_seq_ids
            )
            print(f"✅ 유사도 매트릭스: {len(valid_seq_ids)}개 식당")
    
    def prepare_method2_user_restaurant_profiles(self):
        """
        방법2: 사용자-식당 프로필 벡터 생성
        """
        print("\n=== 방법2: 프로필 벡터 생성 ===")
        
        categories = ['한식', '중식', '일식', '양식', '분식', '아시안', '멕시칸', '기타']
        
        # 사용자 프로필 생성
        user_profiles = {}
        
        if self.user_profiles_df is not None:
            print("📁 CSV에서 사용자 프로필 생성")
            for _, row in self.user_profiles_df.iterrows():
                user_id = row['userId']
                preferred_cat = row['category']
                budget = row['budget']
                
                # 카테고리 원-핫
                cat_vector = [1.0 if cat == preferred_cat else 0.0 for cat in categories]
                
                # 예산 정규화
                norm_budget = min(budget / 200000, 1.0)
                
                user_profiles[user_id] = np.array(cat_vector + [norm_budget])
        else:
            print("📊 Rating에서 사용자 프로필 생성")
            for user_id in self.svd_data['userId'].unique():
                user_ratings = self.svd_data[self.svd_data['userId'] == user_id]
                high_rated = user_ratings[user_ratings['rating'] >= 4.0]
                
                # 선호 카테고리 추출
                categories_found = []
                total_spending = 0
                count = 0
                
                for _, rating in high_rated.iterrows():
                    seq_id = rating['restaurantId']
                    if seq_id in self.sequential_to_real_id:
                        real_id = self.sequential_to_real_id[seq_id]
                        if real_id in self.restaurant_info:
                            info = self.restaurant_info[real_id]
                            categories_found.append(info['category'])
                            total_spending += info['menu_average']
                            count += 1
                
                # 최다 카테고리
                if categories_found:
                    from collections import Counter
                    top_cat = Counter(categories_found).most_common(1)[0][0]
                else:
                    top_cat = '한식'
                
                # 평균 예산
                avg_budget = total_spending / count if count > 0 else 20000
                
                # 프로필 벡터
                cat_vector = [1.0 if cat == top_cat else 0.0 for cat in categories]
                norm_budget = min(avg_budget / 200000, 1.0)
                user_profiles[user_id] = np.array(cat_vector + [norm_budget])
        
        # 식당 프로필 생성
        restaurant_profiles = {}
        
        for _, row in self.content_features.iterrows():
            real_id = row['id']
            
            # 카테고리 벡터
            cat_vector = []
            for cat in categories:
                col_name = f'category_{cat}'
                if col_name in row:
                    cat_vector.append(float(row[col_name]))
                else:
                    cat_vector.append(0.0)
            
            # 정규화된 가격
            norm_price = row['menu_average_normalized']
            
            # 프로필 벡터
            restaurant_vector = cat_vector + [norm_price]
            
            # 실제 ID → 연속번호 변환
            real_to_seq = {v: k for k, v in self.sequential_to_real_id.items()}
            if real_id in real_to_seq:
                seq_id = real_to_seq[real_id]
                restaurant_profiles[seq_id] = np.array(restaurant_vector)
        
        self.user_profiles = user_profiles
        self.restaurant_profiles = restaurant_profiles
        
        print(f"✅ 사용자: {len(user_profiles)}개, 식당: {len(restaurant_profiles)}개")
    
    def method1_prediction(self, user_id, restaurant_id):
        """
        방법1: 식당 유사도 기반 예측
        """
        user_ratings = self.svd_data[self.svd_data['userId'] == user_id]
        
        if len(user_ratings) == 0 or self.similarity_matrix.empty:
            return 2.5
        
        weighted_sum = 0
        sim_sum = 0
        
        for _, rating in user_ratings.iterrows():
            rated_id = rating['restaurantId']
            user_rating = rating['rating']
            
            if (rated_id in self.similarity_matrix.index and 
                restaurant_id in self.similarity_matrix.columns and
                rated_id != restaurant_id):
                
                sim = self.similarity_matrix.loc[rated_id, restaurant_id]
                
                if sim > 0.1:
                    weighted_sum += sim * user_rating
                    sim_sum += sim
        
        if sim_sum > 0:
            prediction = weighted_sum / sim_sum
            return max(1.0, min(5.0, prediction))
        else:
            return user_ratings['rating'].mean() if len(user_ratings) > 0 else 2.5
    
    def method2_prediction(self, user_id, restaurant_id):
        """
        방법2: 사용자-식당 프로필 코사인 유사도 예측
        """
        if (user_id not in self.user_profiles or 
            restaurant_id not in self.restaurant_profiles):
            return 2.5
        
        user_vec = self.user_profiles[user_id]
        restaurant_vec = self.restaurant_profiles[restaurant_id]
        
        # 코사인 유사도
        similarity = cosine_similarity([user_vec], [restaurant_vec])[0][0]
        
        # 1-5 평점으로 변환
        base_rating = 1 + (similarity * 4)
        
        # 사용자 편향 고려
        user_ratings = self.svd_data[self.svd_data['userId'] == user_id]['rating']
        if len(user_ratings) > 0:
            user_avg = user_ratings.mean()
            adjusted = 0.7 * base_rating + 0.3 * user_avg
        else:
            adjusted = base_rating
        
        return max(1.0, min(5.0, adjusted))
    
    def compare_methods(self, test_size=0.2):
        """
        두 방법 성능 비교
        """
        print(f"\n=== 두 방법 성능 비교 ===")
        
        # 준비
        self.prepare_method1_similarity_matrix()
        self.prepare_method2_user_restaurant_profiles()
        
        # 테스트 데이터
        _, test_data = train_test_split(self.svd_data, test_size=test_size, random_state=42)
        
        method1_results = {'predictions': [], 'actuals': []}
        method2_results = {'predictions': [], 'actuals': []}
        
        valid_count = 0
        
        print(f"테스트 데이터: {len(test_data)}개")
        
        for i, (_, row) in enumerate(test_data.iterrows()):
            if i % 20 == 0:
                print(f"  진행: {i}/{len(test_data)}")
            
            user_id = row['userId']
            restaurant_id = row['restaurantId']
            actual = row['rating']
            
            # 두 방법 모두 예측 가능한 경우만
            can_method1 = (not self.similarity_matrix.empty and 
                          restaurant_id in self.similarity_matrix.index)
            can_method2 = (user_id in self.user_profiles and 
                          restaurant_id in self.restaurant_profiles)
            
            if can_method1 and can_method2:
                pred1 = self.method1_prediction(user_id, restaurant_id)
                pred2 = self.method2_prediction(user_id, restaurant_id)
                
                method1_results['predictions'].append(pred1)
                method1_results['actuals'].append(actual)
                method2_results['predictions'].append(pred2)
                method2_results['actuals'].append(actual)
                
                valid_count += 1
        
        print(f"✅ 유효 예측: {valid_count}개")
        
        # 결과 계산
        if valid_count > 0:
            results = {}
            
            for name, data in [('방법1_식당유사도', method1_results), 
                              ('방법2_코사인유사도', method2_results)]:
                
                preds = np.array(data['predictions'])
                actuals = np.array(data['actuals'])
                
                rmse = np.sqrt(np.mean((preds - actuals) ** 2))
                mae = np.mean(np.abs(preds - actuals))
                accuracy = np.mean(np.abs(preds - actuals) <= 0.5)
                
                results[name] = {'rmse': rmse, 'mae': mae, 'accuracy': accuracy}
                
                print(f"\n📊 {name}:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE:  {mae:.4f}")
                print(f"  정확도(±0.5): {accuracy:.3f}")
            
            # 승자 결정
            if len(results) == 2:
                names = list(results.keys())
                rmse1 = results[names[0]]['rmse']
                rmse2 = results[names[1]]['rmse']
                
                winner = names[0] if rmse1 < rmse2 else names[1]
                diff = abs(rmse1 - rmse2)
                
                print(f"\n🏆 결론:")
                print(f"  승자: {winner}")
                print(f"  RMSE 차이: {diff:.4f}")
                
                if diff < 0.05:
                    print(f"  → 두 방법이 비슷한 성능")
                else:
                    print(f"  → {winner}이 더 우수함")
        
        else:
            print("❌ 비교할 유효한 데이터가 없습니다")


def run_comparison():
    """
    비교 분석 실행 함수
    """
    print("🚀 === 콘텐츠 기반 필터링 방법 비교 ===")
    
    # 파일 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    result_dir = os.path.join(current_dir, '..', 'result')
    
    # 필요한 파일들
    files = {
        'svd_data': os.path.join(result_dir, 'svd_data.csv'),
        'restaurants': os.path.join(data_dir, 'restaurants.csv'),
        'content_features': os.path.join(data_dir, 'content_features.csv'),
        'mappings': os.path.join(result_dir, 'restaurant_real_id_mappings.pkl'),
        'user_profiles': os.path.join(result_dir, 'user_profiles.csv')
    }
    
    # 파일 확인
    print("📁 파일 확인:")
    missing = []
    for name, path in files.items():
        if os.path.exists(path):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"⚠️ 누락 파일: {missing}")
        print("💡 전처리를 먼저 실행하세요")
        return None
    
    try:
        # 비교 실행
        comparator = ContentFilteringComparison(
            svd_data_path=files['svd_data'],
            restaurants_path=files['restaurants'],
            content_features_path=files['content_features'],
            mappings_path=files['mappings'],
            user_profiles_path=files['user_profiles']
        )
        
        # 성능 비교
        comparator.compare_methods(test_size=0.3)
        
        print("\n🎉 비교 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comparison()
