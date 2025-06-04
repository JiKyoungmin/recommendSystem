import json
import os
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdaptiveWeightManager:
    """
    사용자 피드백을 기반으로 하이브리드 가중치를 동적 조절
    """
    
    def __init__(self, weights_file_path='result/adaptive_weights.json'):
        self.weights_file_path = weights_file_path
        self.default_alpha = 0.5  # 기본 SVD++ 가중치
        self.default_category_boost = 0.7  # 기본 카테고리 부스트
        
        # 사용자별 가중치 저장
        self.user_weights = self.load_weights()
        
        # 피드백 기반 학습률
        self.learning_rate = 0.05
        self.min_feedback_count = 3  # 최소 피드백 수
        
    def load_weights(self):
        """저장된 가중치 로드"""
        if os.path.exists(self.weights_file_path):
            with open(self.weights_file_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_weights(self):
        """가중치 저장"""
        os.makedirs(os.path.dirname(self.weights_file_path), exist_ok=True)
        with open(self.weights_file_path, 'w') as f:
            json.dump(self.user_weights, f, indent=2)
    
    def get_user_weights(self, user_id):
        """사용자별 최적화된 가중치 반환"""
        user_id_str = str(user_id)
        
        if user_id_str in self.user_weights:
            weights = self.user_weights[user_id_str]
            return {
                'alpha': weights.get('alpha', self.default_alpha),
                'category_boost': weights.get('category_boost', self.default_category_boost),
                'feedback_count': weights.get('feedback_count', 0),
                'last_updated': weights.get('last_updated', 'Never')
            }
        else:
            # 신규 사용자는 기본값 사용
            return {
                'alpha': self.default_alpha,
                'category_boost': self.default_category_boost,
                'feedback_count': 0,
                'last_updated': 'Never'
            }
    
    def update_weights_from_feedback(self, user_id, feedback_score, recommendation_method='hybrid'):
        """
        피드백 점수를 기반으로 가중치 업데이트
        
        Args:
            user_id: 사용자 ID
            feedback_score: 피드백 점수 (1-5)
            recommendation_method: 사용된 추천 방법 ('svd', 'content', 'hybrid')
        """
        user_id_str = str(user_id)
        current_weights = self.get_user_weights(user_id)
        
        # 피드백 점수 정규화 (-1 ~ +1)
        normalized_feedback = (feedback_score - 3) / 2  # 3점 기준, 5점=+1, 1점=-1
        
        # 현재 알파값
        current_alpha = current_weights['alpha']
        current_category_boost = current_weights['category_boost']
        
        # 가중치 조정 로직
        if recommendation_method == 'hybrid':
            # 피드백이 좋으면(4-5점) 현재 비율 유지/강화
            # 피드백이 나쁘면(1-2점) 비율 조정
            
            if feedback_score >= 4:
                # 좋은 피드백: 현재 가중치를 약간 강화
                alpha_adjustment = 0
            elif feedback_score <= 2:
                # 나쁜 피드백: 가중치 균형 조정
                # SVD++가 너무 높으면 줄이고, 너무 낮으면 올림
                if current_alpha > 0.8:
                    alpha_adjustment = -self.learning_rate
                elif current_alpha < 0.5:
                    alpha_adjustment = self.learning_rate
                else:
                    alpha_adjustment = np.random.choice([-self.learning_rate, self.learning_rate])
            else:
                # 보통 피드백(3점): 작은 무작위 조정
                alpha_adjustment = np.random.normal(0, self.learning_rate/2)
            
            # 새로운 알파값 계산 (0.3 ~ 0.9 범위 제한)
            new_alpha = np.clip(current_alpha + alpha_adjustment, 0.3, 0.9)
            
            # 카테고리 부스트도 유사하게 조정
            if feedback_score >= 4:
                category_adjustment = self.learning_rate * 0.5  # 좋으면 카테고리 가중치 증가
            elif feedback_score <= 2:
                category_adjustment = -self.learning_rate * 0.5  # 나쁘면 감소
            else:
                category_adjustment = 0
            
            new_category_boost = np.clip(current_category_boost + category_adjustment, 0.1, 0.5)
            
        else:
            # 단일 방법 사용시는 카테고리 부스트만 조정
            new_alpha = current_alpha
            new_category_boost = current_category_boost
        
        # 가중치 업데이트
        self.user_weights[user_id_str] = {
            'alpha': round(new_alpha, 3),
            'category_boost': round(new_category_boost, 3),
            'feedback_count': current_weights['feedback_count'] + 1,
            'last_updated': datetime.now().isoformat(),
            'last_feedback_score': feedback_score
        }
        
        # 저장
        self.save_weights()
        
        logger.info(f"사용자 {user_id} 가중치 업데이트: α={new_alpha:.3f}, boost={new_category_boost:.3f} (피드백: {feedback_score}점)")
        
        return self.user_weights[user_id_str]