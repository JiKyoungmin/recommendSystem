import pandas as pd
import os
import logging
from difflib import SequenceMatcher

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantsValidator:
    """
    restaurants.csv에 없는 식당 이름을 찾고 로그로 출력하는 클래스
    """
    
    def __init__(self, restaurants_csv_path):
        """
        초기화 - restaurants.csv 파일을 로드
        
        Args:
            restaurants_csv_path: restaurants.csv 파일 경로
        """
        self.restaurants_csv_path = restaurants_csv_path
        self.restaurants_df = None
        self.restaurant_names = set()
        self.load_restaurants_data()
    
    def load_restaurants_data(self):
        """
        restaurants.csv 데이터 로드 및 식당명 세트 생성
        """
        try:
            if not os.path.exists(self.restaurants_csv_path):
                raise FileNotFoundError(f"restaurants.csv 파일을 찾을 수 없습니다: {self.restaurants_csv_path}")
            
            self.restaurants_df = pd.read_csv(self.restaurants_csv_path)
            
            # 식당명을 소문자로 변환하여 세트에 저장 (대소문자 무시하고 비교)
            self.restaurant_names = set(self.restaurants_df['name'].str.lower())
            
            logger.info(f"✅ restaurants.csv 로드 완료: {len(self.restaurants_df)}개 식당")
            logger.info(f"📋 등록된 식당명 샘플: {list(self.restaurant_names)[:5]}")
            
        except Exception as e:
            logger.error(f"❌ restaurants.csv 로드 실패: {e}")
            raise
    
    def normalize_name(self, name):
        """
        식당명 정규화 (공백, 특수문자 제거)
        
        Args:
            name: 원본 식당명
            
        Returns:
            str: 정규화된 식당명
        """
        if pd.isna(name):
            return ""
        
        # 소문자 변환 및 공백 제거
        normalized = str(name).lower().strip()
        
        # 특수문자 제거 (한글, 영문, 숫자만 유지)
        import re
        normalized = re.sub(r'[^\w가-힣]', '', normalized)
        
        return normalized
    
    def find_similar_restaurant(self, missing_name, threshold=0.6):
        """
        누락된 식당명과 유사한 등록된 식당명 찾기
        
        Args:
            missing_name: 누락된 식당명
            threshold: 유사도 임계값 (0.0~1.0)
            
        Returns:
            list: 유사한 식당명 리스트 [(식당명, 유사도)]
        """
        missing_normalized = self.normalize_name(missing_name)
        similar_restaurants = []
        
        for registered_name in self.restaurants_df['name']:
            registered_normalized = self.normalize_name(registered_name)
            
            # 유사도 계산
            similarity = SequenceMatcher(None, missing_normalized, registered_normalized).ratio()
            
            if similarity >= threshold:
                similar_restaurants.append((registered_name, similarity))
        
        # 유사도 순으로 정렬
        similar_restaurants.sort(key=lambda x: x[1], reverse=True)
        
        return similar_restaurants[:3]  # 상위 3개만 반환
    
    def check_missing_restaurants(self, survey_restaurant_names, show_suggestions=True):
        """
        설문에서 입력된 식당명 중 restaurants.csv에 없는 것들을 찾아서 로그 출력
        
        Args:
            survey_restaurant_names: 설문에서 입력된 식당명 리스트
            show_suggestions: True면 유사한 식당명 제안도 표시
            
        Returns:
            dict: 누락된 식당 정보 {'missing': [...], 'found': [...]}
        """
        logger.info("🔍 restaurants.csv에 없는 식당명 검사 시작")
        
        missing_restaurants = []
        found_restaurants = []
        
        # 중복 제거된 유니크한 식당명 리스트
        unique_survey_names = list(set([name for name in survey_restaurant_names if pd.notna(name) and name.strip()]))
        
        logger.info(f"📊 검사 대상 식당: {len(unique_survey_names)}개 (중복 제거 후)")
        
        for survey_name in unique_survey_names:
            survey_name_lower = survey_name.lower().strip()
            
            if survey_name_lower in self.restaurant_names:
                found_restaurants.append(survey_name)
            else:
                missing_restaurants.append(survey_name)
        
        # 결과 로그 출력
        logger.info(f"✅ restaurants.csv에 있는 식당: {len(found_restaurants)}개")
        logger.info(f"❌ restaurants.csv에 없는 식당: {len(missing_restaurants)}개")
        
        if missing_restaurants:
            logger.warning("🚨 === restaurants.csv에 없는 식당 목록 ===")
            
            for i, missing_name in enumerate(missing_restaurants, 1):
                logger.warning(f"  {i}. '{missing_name}'")
                
                # 유사한 식당명 제안
                if show_suggestions:
                    similar_restaurants = self.find_similar_restaurant(missing_name)
                    if similar_restaurants:
                        logger.info(f"     💡 유사한 식당 제안:")
                        for suggested_name, similarity in similar_restaurants:
                            logger.info(f"       - '{suggested_name}' (유사도: {similarity:.3f})")
                    else:
                        logger.info(f"     💡 유사한 식당을 찾을 수 없습니다")
                
                logger.info("")  # 빈 줄 추가
        
        else:
            logger.info("🎉 모든 설문 식당이 restaurants.csv에 등록되어 있습니다!")
        
        return {
            'missing': missing_restaurants,
            'found': found_restaurants,
            'total_survey': len(unique_survey_names),
            'missing_count': len(missing_restaurants),
            'found_count': len(found_restaurants),
            'coverage_rate': len(found_restaurants) / len(unique_survey_names) * 100 if unique_survey_names else 0
        }
    
    def check_rating_xlsx_restaurants(self, rating_xlsx_path):
        """
        rating.xlsx 파일에서 식당명을 추출하여 누락된 식당 검사
        
        Args:
            rating_xlsx_path: rating.xlsx 파일 경로
            
        Returns:
            dict: 누락된 식당 정보
        """
        try:
            # rating.xlsx 로드
            rating_data = pd.read_excel(rating_xlsx_path)
            
            # 식당명이 있는 컬럼들 추출 (0, 2, 4, 6, 8번째 컬럼)
            restaurant_columns = [0, 2, 4, 6, 8]
            
            # 불필요한 컬럼 제거 후 식당명 컬럼 선택
            # 컬럼 정리가 필요한 경우 (타임스탬프, 추가입력여부, 개인정보 관련 컬럼 제거)
            columns_to_drop = [0, 3, 6, 9, 12, 17, 18]
            cleaned_data = rating_data.drop(rating_data.columns[columns_to_drop], axis=1)
            
            # 정리된 데이터에서 식당명 컬럼 추출
            restaurants_in_survey = cleaned_data.iloc[:, [0, 2, 4, 6, 8]]
            
            # 모든 식당명을 하나의 리스트로 수집
            all_survey_restaurants = []
            for col in restaurants_in_survey.columns:
                restaurant_names = restaurants_in_survey[col].dropna().tolist()
                all_survey_restaurants.extend(restaurant_names)
            
            logger.info(f"📊 rating.xlsx에서 추출된 식당명: {len(all_survey_restaurants)}개 (중복 포함)")
            
            # 누락된 식당 검사
            result = self.check_missing_restaurants(all_survey_restaurants, show_suggestions=True)
            
            # 추가 통계 정보
            logger.info(f"📈 === 최종 통계 ===")
            logger.info(f"  전체 설문 식당: {result['total_survey']}개")
            logger.info(f"  등록된 식당: {result['found_count']}개")
            logger.info(f"  누락된 식당: {result['missing_count']}개")
            logger.info(f"  커버리지: {result['coverage_rate']:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ rating.xlsx 검사 중 오류: {e}")
            return None
    
    def generate_missing_restaurants_report(self, result, output_path=None):
        """
        누락된 식당 정보를 CSV 파일로 저장
        
        Args:
            result: check_missing_restaurants 결과
            output_path: 저장할 파일 경로 (None이면 자동 생성)
        """
        if not result or not result['missing']:
            logger.info("누락된 식당이 없어서 리포트를 생성하지 않습니다")
            return
        
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'missing_restaurants_report_{timestamp}.csv'
        
        # 누락된 식당 리스트를 DataFrame으로 변환
        missing_df = pd.DataFrame({
            'missing_restaurant_name': result['missing'],
            'status': '누락',
            'suggested_action': '크롤링 필요 또는 수동 추가'
        })
        
        missing_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"📄 누락된 식당 리포트 저장: {output_path}")


def validate_restaurants_from_rating_file(rating_xlsx_path, restaurants_csv_path, generate_report=True):
    """
    rating.xlsx와 restaurants.csv를 비교하여 누락된 식당을 찾는 메인 함수
    
    Args:
        rating_xlsx_path: rating.xlsx 파일 경로
        restaurants_csv_path: restaurants.csv 파일 경로
        generate_report: True면 누락된 식당 리포트 생성
        
    Returns:
        dict: 검사 결과
    """
    try:
        # 검증기 초기화
        validator = RestaurantsValidator(restaurants_csv_path)
        
        # rating.xlsx에서 식당명 추출 및 검사
        result = validator.check_rating_xlsx_restaurants(rating_xlsx_path)
        
        # 리포트 생성
        if generate_report and result:
            validator.generate_missing_restaurants_report(result)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 식당 검증 중 오류: {e}")
        return None


# 사용 예시
if __name__ == "__main__":
    import os
    
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    
    rating_xlsx_path = os.path.join(data_dir, 'rating.xlsx')
    restaurants_csv_path = os.path.join(data_dir, 'restaurants.csv')
    
    print("🔍 === 식당 데이터 검증 시작 ===")
    
    # 파일 존재 여부 확인
    if not os.path.exists(rating_xlsx_path):
        print(f"❌ rating.xlsx 파일을 찾을 수 없습니다: {rating_xlsx_path}")
    elif not os.path.exists(restaurants_csv_path):
        print(f"❌ restaurants.csv 파일을 찾을 수 없습니다: {restaurants_csv_path}")
    else:
        # 검증 실행
        result = validate_restaurants_from_rating_file(
            rating_xlsx_path=rating_xlsx_path,
            restaurants_csv_path=restaurants_csv_path,
            generate_report=True
        )
        
        if result:
            print(f"\n🎯 === 검증 완료 ===")
            print(f"커버리지: {result['coverage_rate']:.1f}%")
            
            if result['missing_count'] > 0:
                print(f"⚠️ {result['missing_count']}개 식당이 restaurants.csv에 없습니다")
                print("💡 누락된 식당들을 크롤링하거나 수동으로 추가해주세요")
            else:
                print("✅ 모든 설문 식당이 등록되어 있습니다!")
        else:
            print("❌ 검증 실패")