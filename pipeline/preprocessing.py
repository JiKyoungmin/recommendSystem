import pandas as pd
import pickle
import os
import json
import re
from surprise import Dataset, Reader
import logging
from difflib import SequenceMatcher

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRestaurantDataPreprocessor:
    """개선된 식당 평점 데이터 전처리 - 실제 ID 기준 매핑"""
    
    def __init__(self, rating_scale=(1, 5)):
        """
        Args:
            rating_scale: 평점 범위 (기본값: 1-5점)
        """
        self.rating_scale = rating_scale
        self.restaurant_mapping = None
        self.restaurant_reverse_mapping = None
        self.restaurants_df = None
        
    def load_restaurants_data(self, restaurants_csv_path):
        """
        restaurants.csv 로드 (크롤링된 실제 식당 데이터)
        """
        try:
            if not os.path.exists(restaurants_csv_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {restaurants_csv_path}")
                
            self.restaurants_df = pd.read_csv(restaurants_csv_path)
            logger.info(f"식당 데이터 로드 완료: {len(self.restaurants_df)}개 식당")
            
            # 데이터 정보 출력
            logger.info(f"식당 데이터 컬럼: {list(self.restaurants_df.columns)}")
            logger.info(f"샘플 식당명: {self.restaurants_df['name'].head(3).tolist()}")
            
            return self.restaurants_df
            
        except Exception as e:
            logger.error(f"식당 데이터 로드 중 오류: {e}")
            raise
    
    def similarity_score(self, str1, str2):
        """
        두 문자열 간의 유사도 계산
        """
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def normalize_restaurant_name(self, name):
        """
        식당명 정규화 (매칭 정확도 향상)
        """
        if pd.isna(name):
            return ""
        
        # 1. 소문자 변환
        normalized = name.lower()
        
        # 2. 특수문자 및 공백 제거
        normalized = re.sub(r'[^\w가-힣]', '', normalized)
        
        # 3. 흔한 접미사 제거
        suffixes = ['본점', '직영점', '점', '지점', '매장', '역점', '점포']
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        return normalized.strip()
    
    def find_matching_restaurant_id(self, survey_name, min_similarity=0.6):
        """
        설문 식당명에 매칭되는 restaurants.csv의 실제 ID 찾기
        
        Args:
            survey_name: 설문에서 입력받은 식당명
            min_similarity: 최소 유사도 임계값
            
        Returns:
            int: 매칭된 식당의 실제 ID, 매칭 실패시 None
        """
        if pd.isna(survey_name) or survey_name.strip() == "":
            return None
        
        survey_normalized = self.normalize_restaurant_name(survey_name)
        
        # 1단계: 정확한 매칭 시도
        for _, restaurant_row in self.restaurants_df.iterrows():
            restaurant_normalized = self.normalize_restaurant_name(restaurant_row['name'])
            
            if survey_normalized == restaurant_normalized:
                logger.debug(f"정확 매칭: '{survey_name}' → '{restaurant_row['name']}' (ID: {restaurant_row['id']})")
                return restaurant_row['id']
        
        # 2단계: 포함 관계 매칭
        best_match = None
        best_score = 0
        best_restaurant_name = ""
        
        for _, restaurant_row in self.restaurants_df.iterrows():
            restaurant_normalized = self.normalize_restaurant_name(restaurant_row['name'])
            
            # 설문명이 식당명에 포함되거나 그 반대
            if len(survey_normalized) >= 2:
                if survey_normalized in restaurant_normalized:
                    score = len(survey_normalized) / len(restaurant_normalized)
                    if score > best_score:
                        best_score = score
                        best_match = restaurant_row['id']
                        best_restaurant_name = restaurant_row['name']
                
                elif restaurant_normalized in survey_normalized:
                    score = len(restaurant_normalized) / len(survey_normalized)
                    if score > best_score and score > 0.4:  # 최소 40% 일치
                        best_score = score
                        best_match = restaurant_row['id']
                        best_restaurant_name = restaurant_row['name']
        
        # 3단계: 유사도 기반 매칭
        if best_match is None:
            for _, restaurant_row in self.restaurants_df.iterrows():
                similarity = self.similarity_score(survey_name, restaurant_row['name'])
                
                if similarity > best_score and similarity >= min_similarity:
                    best_score = similarity
                    best_match = restaurant_row['id']
                    best_restaurant_name = restaurant_row['name']
        
        if best_match:
            logger.debug(f"유사 매칭: '{survey_name}' → '{best_restaurant_name}' (ID: {best_match}, 유사도: {best_score:.3f})")
            return best_match
        
        logger.warning(f"매칭 실패: '{survey_name}' - 유사한 식당을 찾을 수 없습니다")
        return None
    
    def load_raw_data(self, file_path):
        """
        원본 엑셀 파일을 로드
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
                            'restaurant_name': restaurant_name.strip(),
                            'rating': float(rating_score)
                        })
            
            rating_long = pd.DataFrame(user_data)
            logger.info(f"Long format 변환 완료: {len(rating_long)}개 평점 데이터")
            return rating_long
            
        except Exception as e:
            logger.error(f"Long format 변환 중 오류: {e}")
            raise
    
    def create_real_id_mappings(self, rating_long):
        """
        실제 ID 기준 매핑 생성 + 매칭 실패 시 0부터 시작하는 가상 ID 부여
        """
        try:
            logger.info("실제 ID 기준 매핑 생성 시작")
            
            if self.restaurants_df is None:
                raise ValueError("restaurants.csv를 먼저 로드해야 합니다")
            
            # 설문에서 나온 유니크한 식당명들
            unique_survey_restaurants = rating_long['restaurant_name'].unique()
            logger.info(f"설문에서 추출된 식당 수: {len(unique_survey_restaurants)}개")
            
            # 실제 ID 매핑 딕셔너리
            survey_name_to_real_id = {}
            real_id_to_survey_name = {}
            
            # 가상 ID 매핑 딕셔너리 (0부터 시작하는 양수)
            survey_name_to_virtual_id = {}
            virtual_id_to_survey_name = {}
            
            # 매칭 통계
            exact_matches = 0
            fuzzy_matches = 0
            virtual_id_assigned = 0
            
            # 가상 ID 시작값 (0부터 시작)
            virtual_id_counter = 0
            
            # 각 설문 식당명에 대해 실제 ID 찾기
            for survey_name in unique_survey_restaurants:
                real_id = self.find_matching_restaurant_id(survey_name)
                
                if real_id:
                    # 실제 식당과 매칭된 경우 - 크롤링된 실제 ID 사용
                    survey_name_to_real_id[survey_name] = real_id
                    real_id_to_survey_name[real_id] = survey_name
                    
                    # 매칭 타입 확인
                    normalized_survey = self.normalize_restaurant_name(survey_name)
                    matched_restaurant = self.restaurants_df[self.restaurants_df['id'] == real_id].iloc[0]
                    normalized_matched = self.normalize_restaurant_name(matched_restaurant['name'])
                    
                    if normalized_survey == normalized_matched:
                        exact_matches += 1
                    else:
                        fuzzy_matches += 1
                else:
                    # 매칭 실패 시 0부터 시작하는 가상 ID 부여
                    survey_name_to_virtual_id[survey_name] = virtual_id_counter
                    virtual_id_to_survey_name[virtual_id_counter] = survey_name
                    virtual_id_assigned += 1
                    
                    logger.info(f"가상 ID 부여: '{survey_name}' → ID: {virtual_id_counter}")
                    virtual_id_counter += 1
            
            # 매칭 결과 통계
            total_survey_restaurants = len(unique_survey_restaurants)
            total_real_matched = len(survey_name_to_real_id)
            total_processed = total_real_matched + virtual_id_assigned
            
            logger.info(f"매핑 결과 통계:")
            logger.info(f"  전체 설문 식당: {total_survey_restaurants}개")
            logger.info(f"  정확 매칭 (실제 ID): {exact_matches}개")
            logger.info(f"  유사 매칭 (실제 ID): {fuzzy_matches}개")
            logger.info(f"  가상 ID 부여 (0부터): {virtual_id_assigned}개")
            logger.info(f"  전체 처리율: {(total_processed/total_survey_restaurants)*100:.1f}%")
            
            # 전체 매핑 딕셔너리 생성 (실제 ID + 가상 ID)
            all_survey_to_id = {**survey_name_to_real_id, **survey_name_to_virtual_id}
            all_id_to_survey = {**real_id_to_survey_name, **virtual_id_to_survey_name}
            
            # 모든 평점 데이터에 ID 매핑 (데이터 손실 없음)
            rating_data_with_ids = rating_long.copy()
            rating_data_with_ids['restaurant_id'] = rating_data_with_ids['restaurant_name'].map(all_survey_to_id)
            
            # 매핑 정보 저장 (실제 ID와 가상 ID 구분해서 저장)
            self.restaurant_mapping = {
                'real_id_mappings': {
                    'survey_to_real_id': survey_name_to_real_id,
                    'real_id_to_survey': real_id_to_survey_name
                },
                'virtual_id_mappings': {
                    'survey_to_virtual_id': survey_name_to_virtual_id,
                    'virtual_id_to_survey': virtual_id_to_survey_name
                },
                'all_mappings': {
                    'survey_to_id': all_survey_to_id,
                    'id_to_survey': all_id_to_survey
                },
                'statistics': {
                    'total_restaurants': total_survey_restaurants,
                    'real_matched': total_real_matched,
                    'virtual_restaurants': virtual_id_assigned,
                    'exact_matches': exact_matches,
                    'fuzzy_matches': fuzzy_matches,
                    'virtual_id_range': f"0 ~ {virtual_id_counter-1}" if virtual_id_counter > 0 else "없음"
                }
            }
            
            logger.info(f"ID 매핑 완료: {len(rating_data_with_ids)}개 평점 데이터 생성 (데이터 손실 없음)")
            logger.info(f"실제 식당: {len(survey_name_to_real_id)}개, 가상 식당: {len(survey_name_to_virtual_id)}개 (ID: 0~{virtual_id_counter-1})")
            
            return rating_data_with_ids
            
        except Exception as e:
            logger.error(f"ID 매핑 생성 중 오류: {e}")
            raise

    def prepare_svd_data(self, rating_data_with_real_ids):
        """
        SVD++ 입력용 데이터 준비 (실제 ID 사용)
        """
        try:
            # 필요한 컬럼만 선택하고 컬럼명 변경
            svd_data = rating_data_with_real_ids[['user_id', 'restaurant_id', 'rating']].copy()
            svd_data.columns = ['userId', 'restaurantId', 'rating']
            
            logger.info(f"SVD++ 데이터 준비 완료: {svd_data.shape}")
            logger.info(f"사용자 수: {svd_data['userId'].nunique()}명")
            logger.info(f"식당 수: {svd_data['restaurantId'].nunique()}개")
            logger.info(f"평점 데이터: {len(svd_data)}개")
            
            return svd_data
            
        except Exception as e:
            logger.error(f"SVD++ 데이터 준비 중 오류: {e}")
            raise
    
    def create_surprise_dataset(self, svd_data):
        """
        Surprise 라이브러리용 데이터셋 생성
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
        실제 ID + 가상 ID 매핑 딕셔너리를 파일로 저장
        """
        try:
            if self.restaurant_mapping is None:
                raise ValueError("매핑 딕셔너리가 생성되지 않았습니다.")
            
            with open(output_path, 'wb') as f:
                pickle.dump(self.restaurant_mapping, f)
                
            logger.info(f"전체 매핑 딕셔너리 저장 완료: {output_path}")
            logger.info(f"  - 실제 ID 매핑: {len(self.restaurant_mapping['real_id_mappings']['survey_to_real_id'])}개")
            logger.info(f"  - 가상 ID 매핑: {len(self.restaurant_mapping['virtual_id_mappings']['survey_to_virtual_id'])}개")
            
        except Exception as e:
            logger.error(f"매핑 저장 중 오류: {e}")
            raise
    
    def save_svd_data_csv(self, svd_data, output_path):
        """
        SVD 데이터를 CSV 파일로 저장
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
        """
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(surprise_dataset, f)
            logger.info(f"Surprise 데이터셋 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"Surprise 데이터셋 저장 중 오류: {e}")
            raise
    
    def get_data_summary(self, svd_data):
        """
        데이터 요약 정보 반환 (가상 ID는 0부터 시작)
        """
        if self.restaurant_mapping is None:
            return {
                'total_ratings': len(svd_data),
                'num_users': svd_data['userId'].nunique(),
                'num_restaurants': svd_data['restaurantId'].nunique(),
                'rating_distribution': svd_data['rating'].value_counts().to_dict(),
                'sparsity': len(svd_data) / (svd_data['userId'].nunique() * svd_data['restaurantId'].nunique())
            }
        
        # 매핑 통계에서 정보 가져오기
        stats = self.restaurant_mapping['statistics']
        
        summary = {
            'total_ratings': len(svd_data),
            'num_users': svd_data['userId'].nunique(),
            'num_restaurants': svd_data['restaurantId'].nunique(),
            'real_restaurants': stats['real_matched'],
            'virtual_restaurants': stats['virtual_restaurants'],
            'exact_matches': stats['exact_matches'],
            'fuzzy_matches': stats['fuzzy_matches'],
            'rating_distribution': svd_data['rating'].value_counts().to_dict(),
            'sparsity': len(svd_data) / (svd_data['userId'].nunique() * svd_data['restaurantId'].nunique()),
            'real_match_rate': (stats['real_matched'] / stats['total_restaurants']) * 100 if stats['total_restaurants'] > 0 else 0,
            'virtual_id_range': stats['virtual_id_range']
        }
        return summary
    
    def improved_preprocessing_pipeline(self, rating_file_path, restaurants_csv_path,
                                      mappings_output_path='restaurant_real_id_mappings.pkl', 
                                      csv_output_path='svd_data.csv', 
                                      surprise_output_path='surprise_dataset.pkl', 
                                      save_csv=True, save_surprise=True):
        """
        개선된 전체 전처리 파이프라인 실행 (실제 ID 기준)
        """
        try:
            logger.info("🚀 개선된 전처리 파이프라인 시작 (실제 ID 기준)")
            
            # 1. 식당 데이터 로드 (restaurants.csv)
            self.load_restaurants_data(restaurants_csv_path)
            
            # 2. 원본 설문 데이터 로드
            raw_data = self.load_raw_data(rating_file_path)
            
            # 3. 컬럼 정리
            cleaned_data = self.clean_columns(raw_data)
            
            # 4. 식당-평점 데이터 분리
            restaurants, ratings = self.extract_restaurant_rating_data(cleaned_data)
            
            # 5. Long format 변환
            rating_long = self.convert_to_long_format(restaurants, ratings)
            
            # 6. 실제 ID 기준 매핑 생성 (핵심 개선!)
            rating_data_with_real_ids = self.create_real_id_mappings(rating_long)
            
            # 7. SVD++ 데이터 준비
            svd_data = self.prepare_svd_data(rating_data_with_real_ids)
            
            # 8. Surprise 데이터셋 생성
            surprise_dataset = self.create_surprise_dataset(svd_data)
            
            # 9. 실제 ID 매핑 저장
            self.save_mappings(mappings_output_path)
            
            # 10. SVD 데이터 CSV 저장
            if save_csv:
                self.save_svd_data_csv(svd_data, csv_output_path)
            
            # 11. Surprise 데이터셋 저장
            if save_surprise:
                self.save_surprise_dataset(surprise_dataset, surprise_output_path)
            
            # 12. 데이터 요약
            data_summary = self.get_data_summary(svd_data)

            logger.info("✅ 개선된 전처리 파이프라인 완료")
            logger.info(f"📊 최종 결과:")
            logger.info(f"  - 실제 식당: {data_summary['real_restaurants']}개")
            logger.info(f"  - 가상 식당: {data_summary['virtual_restaurants']}개")
            logger.info(f"  - 전체 식당: {data_summary['num_restaurants']}개")
            logger.info(f"  - 사용자: {data_summary['num_users']}명")
            logger.info(f"  - 평점 데이터: {data_summary['total_ratings']}개")
            logger.info(f"  - 실제 식당 매칭률: {data_summary['real_match_rate']:.1f}%")
            logger.info(f"  - 가상 ID 범위: {data_summary['virtual_id_range']}")
            logger.info(f"  - 데이터 밀도: {data_summary['sparsity']:.4f}")

            return svd_data, surprise_dataset, data_summary
        
        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 중 오류: {e}")
            raise
            

# 사용 예시
if __name__ == "__main__":
    import os
    
    # 현재 스크립트의 디렉토리 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    result_dir = os.path.join(current_dir, '..', 'result')
    
    # 결과 폴더가 없으면 생성
    os.makedirs(result_dir, exist_ok=True)
    
    # 파일 경로
    rating_file_path = os.path.join(data_dir, 'rating.xlsx')
    restaurants_csv_path = os.path.join(data_dir, 'restaurants.csv')
    mappings_output_path = os.path.join(result_dir, 'restaurant_real_id_mappings.pkl')
    csv_output_path = os.path.join(result_dir, 'svd_data.csv')
    surprise_output_path = os.path.join(result_dir, 'surprise_dataset.pkl')

    # 개선된 전처리기 초기화
    preprocessor = ImprovedRestaurantDataPreprocessor(rating_scale=(1, 5))
    
    try:
        print("=== 전처리 파이프라인 실행 ===")
        
        # 전체 파이프라인 실행
        svd_data, surprise_dataset, summary = preprocessor.improved_preprocessing_pipeline(
            rating_file_path=rating_file_path,
            restaurants_csv_path=restaurants_csv_path,
            mappings_output_path=mappings_output_path,
            csv_output_path=csv_output_path,
            surprise_output_path=surprise_output_path,
            save_csv=True,
            save_surprise=True
        )
        
        print("🎉 전처리 완료!")
        print(f"📈 실제 식당 매칭률: {summary['real_match_rate']:.1f}%")
        print(f"📊 실제 식당: {summary['real_restaurants']}개")
        print(f"📊 가상 식당: {summary['virtual_restaurants']}개")
        print(f"📊 가상 ID 범위: {summary['virtual_id_range']}")
        print(f"📊 전체 데이터: {summary}")
        
        # 샘플 데이터 확인
        print(f"\n🔍 생성된 SVD 데이터 샘플:")
        print(svd_data.head())
        
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")