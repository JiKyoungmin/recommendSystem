import os

class Config:
    # 서버 설정
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    
    # 파일 경로
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULT_DIR = os.path.join(BASE_DIR, 'result')
    
    # 추천 설정
    DEFAULT_TOP_N = 10
    SVD_ALPHA = 0.7