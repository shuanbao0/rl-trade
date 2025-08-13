"""
测试专家反馈接口模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.rlhf_components.expert_feedback_interface import ExpertFeedbackInterface


class TestExpertFeedbackInterface:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.feedback_interface = ExpertFeedbackInterface(
            num_experts=3,
            feedback_aggregation='weighted_average',
            confidence_threshold=0.7
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.feedback_interface.num_experts == 3
        assert self.feedback_interface.feedback_aggregation == 'weighted_average'
        assert self.feedback_interface.confidence_threshold == 0.7
    
    def test_collect_feedback(self):
        """测试收集反馈"""
        mock_action = {'buy': 0.6, 'sell': 0.2, 'hold': 0.2}
        mock_state = np.random.rand(50)
        
        feedback = self.feedback_interface.collect_feedback(mock_action, mock_state)
        
        assert isinstance(feedback, dict)
        assert 'expert_ratings' in feedback
        assert 'consensus' in feedback
        assert 'confidence' in feedback
    
    def test_aggregate_expert_opinions(self):
        """测试聚合专家意见"""
        expert_opinions = [
            {'rating': 0.8, 'confidence': 0.9, 'weight': 1.0},
            {'rating': 0.6, 'confidence': 0.7, 'weight': 0.8},
            {'rating': 0.9, 'confidence': 0.95, 'weight': 1.2}
        ]
        
        aggregated = self.feedback_interface.aggregate_opinions(expert_opinions)
        
        assert isinstance(aggregated, dict)
        assert 'final_rating' in aggregated
        assert 'consensus_score' in aggregated
    
    def test_feedback_validation(self):
        """测试反馈验证"""
        valid_feedback = {
            'expert_id': 1,
            'rating': 0.8,
            'confidence': 0.9,
            'timestamp': '2024-01-01T10:00:00'
        }
        
        invalid_feedback = {
            'expert_id': 1,
            'rating': 1.5,  # 超出范围
            'confidence': 0.9
        }
        
        assert self.feedback_interface.validate_feedback(valid_feedback) == True
        assert self.feedback_interface.validate_feedback(invalid_feedback) == False