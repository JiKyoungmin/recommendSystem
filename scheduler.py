import schedule
import time
import logging
import threading
from datetime import datetime
from pipeline.incremental_update import IncrementalUpdatePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelUpdateScheduler:
    """
    모델 업데이트 자동 스케줄러
    """
    
    def __init__(self):
        self.pipeline = IncrementalUpdatePipeline()
        self.is_running = False
        self.scheduler_thread = None
    
    def scheduled_update(self):
        """
        스케줄된 업데이트 작업
        """
        logger.info(f"🕐 스케줄된 모델 업데이트 시작: {datetime.now()}")
        
        try:
            success = self.pipeline.run_incremental_update()
            
            if success:
                logger.info("✅ 스케줄된 업데이트 성공")
            else:
                logger.error("❌ 스케줄된 업데이트 실패")
                
        except Exception as e:
            logger.error(f"❌ 스케줄된 업데이트 중 예외 발생: {str(e)}")
    
    def start_scheduler(self):
        """
        스케줄러 시작
        """
        if self.is_running:
            logger.warning("스케줄러가 이미 실행 중입니다")
            return
        
        # 매주 일요일 새벽 2시에 업데이트 실행
        schedule.every().sunday.at("02:00").do(self.scheduled_update)
        
        # 매일 오전 6시에도 피드백 데이터 확인 (선택사항)
        # schedule.every().day.at("06:00").do(self.check_and_update)
        
        self.is_running = True
        
        def run_scheduler():
            logger.info("📅 모델 업데이트 스케줄러 시작")
            logger.info("⏰ 매주 일요일 오전 2시에 자동 업데이트됩니다")
            
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 스케줄 확인
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("스케줄러가 백그라운드에서 시작되었습니다")
    
    def stop_scheduler(self):
        """
        스케줄러 중지
        """
        if not self.is_running:
            logger.warning("스케줄러가 실행 중이 아닙니다")
            return
        
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("📅 스케줄러가 중지되었습니다")
    
    def check_and_update(self):
        """
        피드백 데이터가 충분히 쌓였으면 즉시 업데이트 (선택적 기능)
        """
        import os
        
        feedback_file = os.path.join('data', 'feedback_queue.jsonl')
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_count = sum(1 for line in f if line.strip())
            
            # 피드백이 50개 이상 쌓이면 즉시 업데이트
            if feedback_count >= 50:
                logger.info(f"피드백 {feedback_count}개 누적 - 즉시 업데이트 실행")
                self.scheduled_update()
    
    def get_next_update_time(self):
        """
        다음 업데이트 예정 시간 반환
        """
        jobs = schedule.get_jobs()
        if jobs:
            next_run = min(job.next_run for job in jobs)
            return next_run.isoformat()
        return "스케줄되지 않음"


# 글로벌 스케줄러 인스턴스
scheduler_instance = None

def start_auto_scheduler():
    """
    자동 스케줄러 시작 함수
    """
    global scheduler_instance
    
    if scheduler_instance is None:
        scheduler_instance = ModelUpdateScheduler()
    
    scheduler_instance.start_scheduler()
    return scheduler_instance

def stop_auto_scheduler():
    """
    자동 스케줄러 중지 함수
    """
    global scheduler_instance
    
    if scheduler_instance:
        scheduler_instance.stop_scheduler()

if __name__ == "__main__":
    # 직접 실행 시 스케줄러 시작
    scheduler = start_auto_scheduler()
    
    try:
        # 메인 스레드에서 대기
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("사용자 중단 요청")
        stop_auto_scheduler()