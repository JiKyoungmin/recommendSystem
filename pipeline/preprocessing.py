
import pandas as pd
import pickle
import os
import json
from surprise import Dataset, Reader
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantDataPreprocessor:
    """식당 평점 데이터 전처리 및 SVD++ 데이터 생성 파이프라인"""
    
    def __init__(self, rating_scale=(1, 5)):
        """
        Args:
            rating_scale: 평점 범위 (기본값: 1-5점)
        """
        self.rating_scale = rating_scale
        self.restaurant_mapping = None
        self.restaurant_reverse_mapping = None
        
    def load_raw_data(self, file_path):
        """
        원본 엑셀 파일을 로드
        
        Args:
            file_path: 엑셀 파일 경로
            
        Returns:
            pd.DataFrame: 원본 데이터
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
                
            data = pd.read_excel(file_path)
            logger.info(f"원본 데이터 로드 완료: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류: {e}")
            raise
    
    def clean_columns(self, data):
        """
        필요없는 컬럼 제거
        
        Args:
            data: 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 정리된 데이터
        """
        try:
            # 제거할 컬럼 인덱스 (타임스탬프, 추가입력여부, 개인정보 관련)
            columns_to_drop = [0, 3, 6, 9, 12, 17, 18]
            cleaned_data = data.drop(data.columns[columns_to_drop], axis=1)
            
            logger.info(f"컬럼 정리 완료: {cleaned_data.shape}")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"컬럼 정리 중 오류: {e}")
            raise
    
    def extract_restaurant_rating_data(self, data):
        """
        식당 이름과 평점 데이터를 분리 추출
        
        Args:
            data: 정리된 데이터프레임
            
        Returns:
            tuple: (식당 이름 데이터, 평점 데이터)
        """
        try:
            # 식당 이름 컬럼 (0, 2, 4, 6, 8번째)
            restaurant_cols = [0, 2, 4, 6, 8]
            # 평점 컬럼 (1, 3, 5, 7, 9번째)  
            rating_cols = [1, 3, 5, 7, 9]
            
            restaurants = data.iloc[:, restaurant_cols]
            ratings = data.iloc[:, rating_cols]
            
            logger.info("식당-평점 데이터 분리 완료")
            return restaurants, ratings
            
        except Exception as e:
            logger.error(f"데이터 분리 중 오류: {e}")
            raise
    
    def convert_to_long_format(self, restaurants, ratings):
        """
        Wide format을 Long format으로 변환
        
        Args:
            restaurants: 식당 이름 데이터
            ratings: 평점 데이터
            
        Returns:
            pd.DataFrame: Long format 데이터
        """
        try:
            user_data = []
            
            for user_id in range(len(restaurants)):
                for i in range(5):  # 5개 식당
                    restaurant_name = restaurants.iloc[user_id, i]
                    rating_score = ratings.iloc[user_id, i]
                    
                    # 식당 이름과 평점이 모두 있는 경우만 추가
                    if pd.notna(restaurant_name) and pd.notna(rating_score):
                        user_data.append({
                            'user_id': user_id,
                            'restaurant_name': restaurant_name,
                            'rating': float(rating_score)
                        })
            
            rating_long = pd.DataFrame(user_data)
            logger.info(f"Long format 변환 완료: {len(rating_long)}개 평점 데이터")
            return rating_long
            
        except Exception as e:
            logger.error(f"Long format 변환 중 오류: {e}")
            raise
    
    def create_mappings(self, rating_long):
        """
        식당 이름-ID 매핑 딕셔너리 생성
        
        Args:
            rating_long: Long format 데이터
            
        Returns:
            pd.DataFrame: 매핑이 추가된 데이터
        """
        try:
            # 식당 이름 정규화 (공백 제거)
            rating_long['restaurant_name'] = rating_long['restaurant_name'].str.strip()
            
            # 유니크한 식당 목록과 ID 매핑
            unique_restaurants = rating_long['restaurant_name'].unique()
            
            # 매핑 딕셔너리 생성
            self.restaurant_mapping = {name: idx for idx, name in enumerate(unique_restaurants)}
            self.restaurant_reverse_mapping = {idx: name for name, idx in self.restaurant_mapping.items()}
            
            # 식당 ID 추가
            rating_long['restaurant_id'] = rating_long['restaurant_name'].map(self.restaurant_mapping)
            
            logger.info(f"매핑 생성 완료: {len(unique_restaurants)}개 식당")
            return rating_long
            
        except Exception as e:
            logger.error(f"매핑 생성 중 오류: {e}")
            raise
    
    def prepare_svd_data(self, rating_long):
        """
        SVD++ 입력용 데이터 준비
        
        Args:
            rating_long: 매핑이 완료된 Long format 데이터
            
        Returns:
            pd.DataFrame: SVD++ 입력용 데이터
        """
        try:
            # 필요한 컬럼만 선택하고 컬럼명 변경
            svd_data = rating_long[['user_id', 'restaurant_id', 'rating']].copy()
            svd_data.columns = ['userId', 'restaurantId', 'rating']
            
            logger.info(f"SVD++ 데이터 준비 완료: {svd_data.shape}")
            return svd_data
            
        except Exception as e:
            logger.error(f"SVD++ 데이터 준비 중 오류: {e}")
            raise
    
    def create_surprise_dataset(self, svd_data):
        """
        Surprise 라이브러리용 데이터셋 생성
        
        Args:
            svd_data: SVD++ 입력용 데이터
            
        Returns:
            surprise.Dataset: Surprise 데이터셋
        """
        try:
            reader = Reader(rating_scale=self.rating_scale)
            surprise_data = Dataset.load_from_df(svd_data, reader)
            
            logger.info("Surprise 데이터셋 생성 완료")
            return surprise_data
            
        except Exception as e:
            logger.error(f"Surprise 데이터셋 생성 중 오류: {e}")
            raise
    
    def save_mappings(self, output_path):
        """
        매핑 딕셔너리를 파일로 저장
        
        Args:
            output_path: 저장할 파일 경로
        """
        try:
            if self.restaurant_mapping is None:
                raise ValueError("매핑 딕셔너리가 생성되지 않았습니다.")
                
            mappings = {
                'restaurant_to_id': self.restaurant_mapping,
                'id_to_restaurant': self.restaurant_reverse_mapping
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(mappings, f)
                
            logger.info(f"매핑 딕셔너리 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"매핑 저장 중 오류: {e}")
            raise
    
    def save_svd_data_csv(self, svd_data, output_path):
        """
        SVD 데이터를 CSV 파일로 저장
        
        Args:
            svd_data: SVD 입력용 데이터프레임
            output_path: 저장할 CSV 파일 경로
        """
        try:
            svd_data.to_csv(output_path, index=False)
            logger.info(f"SVD 데이터 CSV 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"CSV 저장 중 오류: {e}")
            raise
    
    def save_surprise_dataset(self, surprise_dataset, output_path):
        """
        Surprise 데이터셋을 pickle 파일로 저장
        
        Args:
            surprise_dataset: Surprise Dataset 객체
            output_path: 저장할 pickle 파일 경로
        """
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(surprise_dataset, f)
            logger.info(f"Surprise 데이터셋 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"Surprise 데이터셋 저장 중 오류: {e}")
            raise
    
    def load_surprise_dataset(self, dataset_path):
        """
        저장된 Surprise 데이터셋을 불러오기
        
        Args:
            dataset_path: 불러올 pickle 파일 경로
            
        Returns:
            surprise.Dataset: Surprise 데이터셋 객체
        """
        try:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
            
            with open(dataset_path, 'rb') as f:
                surprise_dataset = pickle.load(f)
            
            logger.info(f"Surprise 데이터셋 로드 완료: {dataset_path}")
            return surprise_dataset
            
        except Exception as e:
            logger.error(f"Surprise 데이터셋 로드 중 오류: {e}")
            raise
    
    def load_svd_data_from_csv(self, csv_path):
        """
        CSV 파일에서 SVD 데이터를 불러와서 Surprise 데이터셋 생성
        
        Args:
            csv_path: 불러올 CSV 파일 경로
            
        Returns:
            tuple: (svd_data, surprise_dataset)
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
            
            # CSV에서 SVD 데이터 로드
            svd_data = pd.read_csv(csv_path)
            logger.info(f"CSV에서 SVD 데이터 로드 완료: {svd_data.shape}")
            
            # 데이터 컬럼 검증
            required_columns = ['userId', 'restaurantId', 'rating']
            if not all(col in svd_data.columns for col in required_columns):
                raise ValueError(f"CSV 파일에 필수 컬럼이 없습니다. 필요한 컬럼: {required_columns}")
            
            # Surprise 데이터셋 생성
            surprise_dataset = self.create_surprise_dataset(svd_data)
            
            logger.info("CSV에서 Surprise 데이터셋 생성 완료")
            return svd_data, surprise_dataset
            
        except Exception as e:
            logger.error(f"CSV 로드 중 오류: {e}")
            raise
    
    def get_data_summary(self, svd_data):
        """
        데이터 요약 정보 반환
        
        Args:
            svd_data: SVD++ 데이터
            
        Returns:
            dict: 데이터 요약 정보
        """
        summary = {
            'total_ratings': len(svd_data),
            'num_users': svd_data['userId'].nunique(),
            'num_restaurants': svd_data['restaurantId'].nunique(),
            'rating_distribution': svd_data['rating'].value_counts().to_dict(),
            'sparsity': len(svd_data) / (svd_data['userId'].nunique() * svd_data['restaurantId'].nunique())
        }
        return summary
    
    def load_restaurant_json(self, file_path):
        """
        restaurants.json 파일을 로드
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            list: 식당 데이터 리스트
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                restaurant_data = json.load(f)
                
            logger.info(f"식당 JSON 데이터 로드 완료: {len(restaurant_data)}개 식당")
            return restaurant_data
            
        except Exception as e:
            logger.error(f"JSON 데이터 로드 중 오류: {e}")
            raise

    def extract_restaurant_features(self, restaurant_data):
        """
        JSON 데이터에서 필요한 필드만 추출하여 데이터프레임 생성
        누락된 필드는 None으로 처리하여 데이터 보존
        
        Args:
            restaurant_data: 식당 JSON 데이터 리스트
            
        Returns:
            pd.DataFrame: 식당 정보 데이터프레임
        """
        try:
            restaurants = []
            
            for restaurant in restaurant_data:
                # 필요한 필드만 추출 (없으면 None)
                extracted_data = {
                    'id': restaurant.get('id'),
                    'name': restaurant.get('name'),
                    'category': restaurant.get('category'),
                    'menu_average': restaurant.get('menu_average')
                }
                
                # id나 name이 없는 경우만 스킵 (핵심 식별자)
                if extracted_data['id'] is None or extracted_data['name'] is None:
                    logger.warning(f"핵심 필드(id/name) 누락된 식당 스킵: {restaurant.get('name', 'Unknown')}")
                    continue
                    
                restaurants.append(extracted_data)
            
            restaurant_df = pd.DataFrame(restaurants)
            
            # 데이터 타입 정리
            if not restaurant_df.empty:
                # menu_average는 숫자형으로 변환 (변환 불가능하면 NaN)
                restaurant_df['menu_average'] = pd.to_numeric(restaurant_df['menu_average'], errors='coerce')
            
            logger.info(f"식당 데이터프레임 생성 완료: {len(restaurant_df)}개 식당")
            
            # 누락 데이터 현황 로그
            missing_info = restaurant_df.isnull().sum()
            if missing_info.any():
                logger.info(f"누락 데이터 현황:\n{missing_info}")
            
            return restaurant_df
        
        except Exception as e:
            logger.error(f"식당 데이터 추출 중 오류: {e}")
            raise
    
    def prepare_content_features(self, restaurant_df):
        """
        콘텐츠 기반 필터링을 위한 특성 벡터 준비
        NaN 값 처리 및 범주형 데이터 인코딩
        
        Args:
            restaurant_df: 식당 데이터프레임
            
        Returns:
            pd.DataFrame: 전처리된 특성 매트릭스
        """
        try:
            df = restaurant_df.copy()
            
            # 1. menu_average 결측값 처리 (평균값으로 대체)
            if df['menu_average'].isnull().any():
                mean_price = df['menu_average'].mean()
                df['menu_average'] = df['menu_average'].fillna(mean_price)
                logger.info(f"menu_average 결측값을 평균값({mean_price:.0f})으로 대체")
            
            # 2. category 결측값 처리 (기타로 대체)
            if df['category'].isnull().any():
                df['category'] = df['category'].fillna('기타')
                logger.info("category 결측값을 '기타'로 대체")
            
            # 3. category를 원-핫 인코딩
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            
            # 4. menu_average 정규화 (0-1 스케일)
            df['menu_average_normalized'] = (df['menu_average'] - df['menu_average'].min()) / \
                                        (df['menu_average'].max() - df['menu_average'].min())
            
            # 5. 최종 특성 매트릭스 구성
            feature_matrix = pd.concat([
                df[['id', 'name']],  # 식별자
                category_dummies,    # 카테고리 원-핫 인코딩
                df[['menu_average_normalized']]  # 정규화된 가격
            ], axis=1)
            
            logger.info(f"콘텐츠 특성 매트릭스 생성 완료: {feature_matrix.shape}")
            return feature_matrix
            
        except Exception as e:
            logger.error(f"콘텐츠 특성 준비 중 오류: {e}")
            raise

    def save_restaurant_csv(self, restaurant_df, output_path):
        """
        식당 데이터프레임을 CSV 파일로 저장
        
        Args:
            restaurant_df: 식당 데이터프레임
            output_path: 저장할 CSV 파일 경로
        """
        try:
            restaurant_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"식당 데이터 CSV 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"식당 CSV 저장 중 오류: {e}")
            raise

    def process_restaurant_data(self, json_file_path, csv_output_path):
        """
        식당 JSON 데이터 전처리 파이프라인
        JSON 파일에서 필요한 필드만 추출하여 CSV로 저장
        
        Args:
            json_file_path: 입력 JSON 파일 경로
            csv_output_path: 출력 CSV 파일 경로
            save_content_features: 콘텐츠 특성 매트릭스 저장 여부
            
        Returns:
            pd.DataFrame: 처리된 식당 데이터프레임
        """
        try:
            logger.info("식당 데이터 전처리 파이프라인 시작")
            
            # 1. JSON 파일 로드
            restaurant_data = self.load_restaurant_json(json_file_path)
            
            # 2. 필요한 필드 추출하여 데이터프레임 생성
            restaurant_df = self.extract_restaurant_features(restaurant_data)
            
            # 3. CSV 파일로 저장
            self.save_restaurant_csv(restaurant_df, csv_output_path)
            
            logger.info("식당 데이터 전처리 파이프라인 완료")
            return restaurant_df
            
        except Exception as e:
            logger.error(f"식당 데이터 전처리 중 오류: {e}")
            raise
    def save_content_features_csv(self, content_features_df, output_path):
        """
        콘텐츠 기반 필터링용 특성 매트릭스를 CSV 파일로 저장
        
        Args:
            content_features_df: 콘텐츠 특성 매트릭스 데이터프레임
            output_path: 저장할 CSV 파일 경로
        """
        try:
            content_features_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"콘텐츠 특성 매트릭스 CSV 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"콘텐츠 특성 CSV 저장 중 오류: {e}")
            raise

    def process_content_features_pipeline(self, restaurant_df, content_csv_output_path):
        """
        콘텐츠 기반 필터링용 특성 처리 파이프라인
        특성 매트릭스 생성 후 CSV로 저장
        
        Args:
            restaurant_df: 식당 데이터프레임
            content_csv_output_path: 콘텐츠 특성 CSV 저장 경로
            
        Returns:
            pd.DataFrame: 처리된 콘텐츠 특성 매트릭스
        """
        try:
            logger.info("콘텐츠 특성 처리 파이프라인 시작")
            
            # 1. 콘텐츠 특성 매트릭스 생성
            content_features = self.prepare_content_features(restaurant_df)
            
            # 2. CSV 파일로 저장
            self.save_content_features_csv(content_features, content_csv_output_path)
            
            logger.info("콘텐츠 특성 처리 파이프라인 완료")
            return content_features
            
        except Exception as e:
            logger.error(f"콘텐츠 특성 처리 중 오류: {e}")
            raise
    
    def preprocessing_pipeline(self, input_file_path, mappings_output_path='restaurant_mappings.pkl', 
                          csv_output_path='svd_data.csv', surprise_output_path='surprise_dataset.pkl', 
                          save_csv=True, save_surprise=True):
        """
        전체 전처리 파이프라인 실행
        
        Args:
            input_file_path: 입력 엑셀 파일 경로
            mappings_output_path: 매핑 딕셔너리 저장 경로
            csv_output_path: SVD 데이터 CSV 저장 경로
            surprise_output_path: Surprise 데이터셋 저장 경로
            save_csv: CSV 파일로 저장 여부 (기본값: True)
            save_surprise: Surprise 데이터셋 저장 여부 (기본값: True)
        """
        try:
            logger.info("전처리 파이프라인 시작")
            
            # 1. 원본 데이터 로드
            raw_data = self.load_raw_data(input_file_path)
            
            # 2. 컬럼 정리
            cleaned_data = self.clean_columns(raw_data)
            
            # 3. 식당-평점 데이터 분리
            restaurants, ratings = self.extract_restaurant_rating_data(cleaned_data)
            
            # 4. Long format 변환
            rating_long = self.convert_to_long_format(restaurants, ratings)
            
            # 5. 매핑 생성
            rating_long = self.create_mappings(rating_long)
            
            # 6. SVD++ 데이터 준비
            svd_data = self.prepare_svd_data(rating_long)
            
            # 7. Surprise 데이터셋 생성
            surprise_dataset = self.create_surprise_dataset(svd_data)
            
            # 8. 매핑 저장
            self.save_mappings(mappings_output_path)
            
            # 9. SVD 데이터 CSV 저장
            if save_csv:
                self.save_svd_data_csv(svd_data, csv_output_path)
            
            # 10. Surprise 데이터셋 저장
            if save_surprise:
                self.save_surprise_dataset(surprise_dataset, surprise_output_path)
            
            # 11. 데이터 요약
            data_summary = self.get_data_summary(svd_data)
            
            logger.info("전처리 파이프라인 완료")
            
            return svd_data, surprise_dataset, data_summary
        
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류: {e}")
            raise

# 사용 예시 수정
if __name__ == "__main__":
    import os
    
    # 현재 스크립트의 디렉토리 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')  # 상위 폴더의 data 폴더
    result_dir = os.path.join(current_dir, '..', 'result')  # 상위 폴더의 result 폴더
    
    # 결과 폴더가 없으면 생성
    os.makedirs(result_dir, exist_ok=True)
    
    # data path
    rating_file_path=os.path.join(data_dir, 'rating.xlsx')
    restaurant_json_path = os.path.join(data_dir, 'restaurants.json')
    restaurant_csv_path = os.path.join(data_dir, 'restaurants.csv')
    content_features_csv_path = os.path.join(data_dir, 'content_features.csv')
    # result path 
    mappings_output_path=os.path.join(result_dir, 'restaurant_mappings.pkl')
    csv_output_path=os.path.join(result_dir, 'svd_data.csv')
    surprise_output_path=os.path.join(result_dir, 'surprise_dataset.pkl')

    # 전처리기 초기화
    preprocessor = RestaurantDataPreprocessor(rating_scale=(1, 5))
    
    try:
        # 1. 식당 데이터 전처리
        print("=== 식당 데이터 전처리 시작 ===")
        restaurant_df = preprocessor.process_restaurant_data(
            restaurant_json_path,
            restaurant_csv_path
        )
        print(f"식당 데이터 처리 완료: {len(restaurant_df)}개 식당")
        print("처리된 식당 데이터 샘플:")
        print(restaurant_df.head())
        
        # 2. 콘텐츠 기반 필터링용 특성 처리 (추가)
        print("\n=== 콘텐츠 특성 매트릭스 생성 시작 ===")
        content_features = preprocessor.process_content_features_pipeline(
            restaurant_df,
            content_features_csv_path
        )
        print(f"콘텐츠 특성 매트릭스 생성 완료: {content_features.shape}")
        print("특성 컬럼:", content_features.columns.tolist())
        print("콘텐츠 특성 매트릭스 샘플:")
        print(content_features.head())

        # 3. 평점 데이터 전처리
        print("\n=== 평점 데이터 전처리 시작 ===")
        svd_data, surprise_dataset, summary = preprocessor.preprocessing_pipeline(
            rating_file_path,
            mappings_output_path,
            csv_output_path,
            surprise_output_path,
            save_csv=True,
            save_surprise=True
        )
        
        print("평점 데이터 전처리 완료!")
        print(f"데이터 요약: {summary}")
        
    except Exception as e:
        print(f"전처리 실패: {e}")

#빠른 로드 (Surprise Dataset 직접 불러오기)
    try:
        print("Surprise 데이터셋 직접 로드 테스트...")
        
        # 새로운 전처리기 인스턴스 생성
        new_preprocessor = RestaurantDataPreprocessor(rating_scale=(1, 5))
        
        # Surprise 데이터셋 직접 로드 (가장 빠름)
        surprise_dataset = new_preprocessor.load_surprise_dataset(surprise_output_path)
        
        print("✅ Surprise 데이터셋 직접 로드 성공!")
        print("⚡ 바로 SVD++ 학습 가능!")
        
    except Exception as e:
        print(f"직접 로드 실패: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # CSV 로드 (호환성용)
    try:
        print("CSV에서 데이터 불러오기 테스트...")
        
        # CSV에서 데이터 로드 (느리지만 호환성 좋음)
        loaded_svd_data, loaded_surprise_dataset = new_preprocessor.load_svd_data_from_csv(csv_output_path)
        
        print("✅ CSV 로드 성공!")
        print(f"로드된 데이터 형태: {loaded_svd_data.shape}")
        print("첫 5개 데이터:")
        print(loaded_svd_data.head())
        
    except Exception as e:
        print(f"CSV 로드 실패: {e}")
