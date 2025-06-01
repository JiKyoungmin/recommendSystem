import pandas as pd
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfileExtractor:
    """
    rating.xlsx에서 사용자 프로필(선호 카테고리, 예산) 추출
    """
    
    def __init__(self):
        self.raw_data = None
        self.user_profiles = None
        
    def load_and_analyze_data(self, file_path):
        """
        rating.xlsx 파일을 로드하고 컬럼 구조 분석
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
                
            self.raw_data = pd.read_excel(file_path)
            logger.info(f"원본 데이터 로드 완료: {self.raw_data.shape}")
            
            # 컬럼 정보 출력
            print("=== rating.xlsx 컬럼 구조 분석 ===")
            for i, col in enumerate(self.raw_data.columns):
                sample_value = self.raw_data.iloc[0, i] if len(self.raw_data) > 0 else "N/A"
                print(f"컬럼 {i}: {col} (샘플: {sample_value})")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류: {e}")
            raise
    
    def extract_user_profiles_auto(self, category_col_idx=None, budget_col_idx=None):
        """
        사용자 프로필 자동 추출 (컬럼 인덱스 지정)
        
        Args:
            category_col_idx: 선호 카테고리 컬럼 인덱스
            budget_col_idx: 예산 컬럼 인덱스
        """
        try:
            if self.raw_data is None:
                raise ValueError("먼저 데이터를 로드해야 합니다.")
            
            # 자동으로 컬럼 찾기 시도
            if category_col_idx is None or budget_col_idx is None:
                logger.info("컬럼 인덱스가 지정되지 않음. 자동 탐지 시도...")
                category_col_idx, budget_col_idx = self._auto_detect_profile_columns()
            
            if category_col_idx is None or budget_col_idx is None:
                raise ValueError("선호 카테고리 또는 예산 컬럼을 찾을 수 없습니다. 수동으로 컬럼 인덱스를 지정해주세요.")
            
            # 사용자 프로필 데이터 추출
            user_profiles_data = []
            
            for user_id in range(len(self.raw_data)):
                # 선호 카테고리 추출
                category = self.raw_data.iloc[user_id, category_col_idx]
                if pd.isna(category):
                    category = "한식"  # 기본값
                
                # 예산 추출
                budget = self.raw_data.iloc[user_id, budget_col_idx]
                if pd.isna(budget):
                    budget = 20000  # 기본값 (2만원)
                else:
                    # 숫자로 변환 시도
                    try:
                        budget = float(str(budget).replace(',', '').replace('원', ''))
                    except:
                        budget = 20000  # 변환 실패시 기본값
                
                user_profiles_data.append({
                    'userId': user_id,
                    'category': category,
                    'budget': int(budget)
                })
            
            self.user_profiles = pd.DataFrame(user_profiles_data)
            
            logger.info(f"사용자 프로필 추출 완료: {len(self.user_profiles)}명")
            logger.info(f"사용된 컬럼 - 카테고리: {category_col_idx}, 예산: {budget_col_idx}")
            
            # 추출 결과 요약
            print(f"\n=== 사용자 프로필 추출 결과 ===")
            print(f"총 사용자 수: {len(self.user_profiles)}명")
            print(f"카테고리 분포:")
            print(self.user_profiles['category'].value_counts())
            print(f"\n예산 통계:")
            print(f"  평균: {self.user_profiles['budget'].mean():,.0f}원")
            print(f"  중앙값: {self.user_profiles['budget'].median():,.0f}원")
            print(f"  최소: {self.user_profiles['budget'].min():,.0f}원")
            print(f"  최대: {self.user_profiles['budget'].max():,.0f}원")
            
            return self.user_profiles
            
        except Exception as e:
            logger.error(f"사용자 프로필 추출 중 오류: {e}")
            raise
    
    def _auto_detect_profile_columns(self):
        """
        선호 카테고리와 예산 컬럼을 자동으로 탐지
        """
        category_col_idx = None
        budget_col_idx = None
        
        # 컬럼명 기반 탐지
        for i, col_name in enumerate(self.raw_data.columns):
            col_name_lower = str(col_name).lower()
            
            # 카테고리 관련 키워드
            if any(keyword in col_name_lower for keyword in ['카테고리', 'category', '음식', '선호', '좋아']):
                category_col_idx = i
                logger.info(f"카테고리 컬럼 발견: 인덱스 {i} ({col_name})")
            
            # 예산 관련 키워드
            elif any(keyword in col_name_lower for keyword in ['예산', 'budget', '식비', '금액', '가격', '원']):
                budget_col_idx = i
                logger.info(f"예산 컬럼 발견: 인덱스 {i} ({col_name})")
        
        # 컬럼명으로 찾지 못한 경우, 데이터 패턴 기반 탐지
        if category_col_idx is None:
            # 마지막 몇 개 컬럼에서 텍스트 데이터 찾기
            for i in range(len(self.raw_data.columns)-5, len(self.raw_data.columns)):
                if i < 0:
                    continue
                sample_values = self.raw_data.iloc[:5, i].dropna()
                if len(sample_values) > 0:
                    # 텍스트 데이터이고 음식 카테고리 같은 패턴인지 확인
                    first_value = str(sample_values.iloc[0])
                    if any(food_type in first_value for food_type in ['한식', '중식', '일식', '양식', '분식', '아시안']):
                        category_col_idx = i
                        logger.info(f"패턴 기반 카테고리 컬럼 발견: 인덱스 {i}")
                        break
        
        if budget_col_idx is None:
            # 마지막 몇 개 컬럼에서 숫자 데이터 찾기
            for i in range(len(self.raw_data.columns)-5, len(self.raw_data.columns)):
                if i < 0 or i == category_col_idx:
                    continue
                sample_values = self.raw_data.iloc[:5, i].dropna()
                if len(sample_values) > 0:
                    try:
                        # 숫자로 변환 가능하고 예산 범위인지 확인
                        numeric_values = []
                        for val in sample_values:
                            try:
                                num_val = float(str(val).replace(',', '').replace('원', ''))
                                if 1000 <= num_val <= 200000:  # 1천원~20만원 범위
                                    numeric_values.append(num_val)
                            except:
                                continue
                        
                        if len(numeric_values) >= 2:  # 최소 2개 이상 유효한 예산 데이터
                            budget_col_idx = i
                            logger.info(f"패턴 기반 예산 컬럼 발견: 인덱스 {i}")
                            break
                    except:
                        continue
        
        return category_col_idx, budget_col_idx
    
    def extract_user_profiles_manual(self, category_col_idx, budget_col_idx):
        """
        수동으로 컬럼 인덱스를 지정하여 사용자 프로필 추출
        
        Args:
            category_col_idx: 선호 카테고리 컬럼 인덱스
            budget_col_idx: 예산 컬럼 인덱스
        """
        return self.extract_user_profiles_auto(category_col_idx, budget_col_idx)
    
    def save_user_profiles_csv(self, output_path):
        """
        사용자 프로필을 CSV 파일로 저장
        """
        try:
            if self.user_profiles is None:
                raise ValueError("사용자 프로필이 추출되지 않았습니다.")
            
            self.user_profiles.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"사용자 프로필 CSV 저장 완료: {output_path}")
            
            # 저장된 파일 확인
            print(f"\n=== 저장된 파일 내용 미리보기 ===")
            print(self.user_profiles.head(10))
            
        except Exception as e:
            logger.error(f"CSV 저장 중 오류: {e}")
            raise
    
    def get_user_profile_summary(self):
        """
        사용자 프로필 요약 정보 반환
        """
        if self.user_profiles is None:
            return None
        
        summary = {
            'total_users': len(self.user_profiles),
            'category_distribution': self.user_profiles['category'].value_counts().to_dict(),
            'budget_stats': {
                'mean': float(self.user_profiles['budget'].mean()),
                'median': float(self.user_profiles['budget'].median()),
                'min': int(self.user_profiles['budget'].min()),
                'max': int(self.user_profiles['budget'].max()),
                'std': float(self.user_profiles['budget'].std())
            }
        }
        return summary


# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    result_dir = os.path.join(current_dir, '..', 'result')
    
    # 결과 폴더가 없으면 생성
    os.makedirs(result_dir, exist_ok=True)
    
    rating_file_path = os.path.join(data_dir, 'rating.xlsx')
    user_profiles_output_path = os.path.join(result_dir, 'user_profiles.csv')
    
    # 사용자 프로필 추출기 초기화
    extractor = UserProfileExtractor()
    
    try:
        print("=== 사용자 프로필 추출 시작 ===")
        
        # 1. 데이터 로드 및 컬럼 구조 분석
        extractor.load_and_analyze_data(rating_file_path)
        
        # 2. 사용자 프로필 추출 (자동 탐지)
        # user_profiles = extractor.extract_user_profiles_auto()
        
        # 만약 자동 탐지가 실패하면 수동으로 컬럼 인덱스 지정
        user_profiles = extractor.extract_user_profiles_manual(
            category_col_idx=16,  # 예시: 16번째 컬럼이 선호 카테고리
            budget_col_idx=17     # 예시: 17번째 컬럼이 예산
        )
        
        # 3. CSV 파일로 저장
        extractor.save_user_profiles_csv(user_profiles_output_path)
        
        # 4. 요약 정보 출력
        summary = extractor.get_user_profile_summary()
        print(f"\n=== 최종 요약 ===")
        print(f"사용자 프로필 추출 완료: {summary['total_users']}명")
        print(f"평균 예산: {summary['budget_stats']['mean']:,.0f}원")
        
        print("🎉 사용자 프로필 추출 완료!")
        
    except Exception as e:
        print(f"❌ 사용자 프로필 추출 실패: {e}")
        print("\n💡 해결 방법:")
        print("1. rating.xlsx 파일이 data 폴더에 있는지 확인")
        print("2. 컬럼 구조를 확인하고 수동으로 컬럼 인덱스 지정")
        print("3. 예시: extractor.extract_user_profiles_manual(category_col_idx=16, budget_col_idx=17)")