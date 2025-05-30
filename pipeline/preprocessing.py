
import pandas as pd
import pickle
import os
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
    
    input_file_path=os.path.join(data_dir, 'rating.xlsx')
    mappings_output_path=os.path.join(result_dir, 'restaurant_mappings.pkl')
    csv_output_path=os.path.join(result_dir, 'svd_data.csv')
    surprise_output_path=os.path.join(result_dir, 'surprise_dataset.pkl')

    # 전처리기 초기화
    preprocessor = RestaurantDataPreprocessor(rating_scale=(1, 5))
    
    try:
        svd_data, surprise_dataset, summary = preprocessor.preprocessing_pipeline(
            input_file_path,
            mappings_output_path,
            csv_output_path,
            surprise_output_path,  # 추가
            save_csv=True,
            save_surprise=True  # 추가
        )
        
        print("전처리 완료!")
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
