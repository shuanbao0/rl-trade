"""
测试偏好学习模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.rlhf_components.preference_learning import PreferenceLearningTrainer


class TestPreferenceLearningTrainer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.preference_model = PreferenceLearningTrainer(
            feature_dim=128,
            hidden_dims=[256, 128, 64],
            learning_rate=0.001,
            batch_size=32
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.preference_model.feature_dim == 128
        assert self.preference_model.hidden_dims == [256, 128, 64]
        assert self.preference_model.learning_rate == 0.001
        assert self.preference_model.batch_size == 32
    
    def test_preference_prediction(self):
        """测试偏好预测"""
        action_a = np.random.rand(128)
        action_b = np.random.rand(128)
        
        preference_score = self.preference_model.predict_preference(action_a, action_b)
        
        assert isinstance(preference_score, (int, float))
        assert 0 <= preference_score <= 1
    
    def test_model_training(self):
        """测试模型训练"""
        preference_pairs = []
        for _ in range(10):
            action_a = np.random.rand(128)
            action_b = np.random.rand(128) 
            preference = np.random.choice(['a', 'b'])
            
            preference_pairs.append({
                'action_a': action_a,
                'action_b': action_b,
                'preference': preference
            })
        
        training_loss = self.preference_model.train(preference_pairs, epochs=5)
        
        assert isinstance(training_loss, list)
        assert len(training_loss) == 5
        assert all(isinstance(loss, (int, float)) for loss in training_loss)
    
    def test_preference_data_validation(self):
        """测试偏好数据验证"""
        valid_pair = {
            'action_a': np.random.rand(128),
            'action_b': np.random.rand(128),
            'preference': 'a',
            'confidence': 0.8
        }
        
        invalid_pair = {
            'action_a': np.random.rand(64),  # 错误维度
            'action_b': np.random.rand(128),
            'preference': 'invalid'  # 无效偏好
        }
        
        assert self.preference_model.validate_preference_pair(valid_pair) == True
        assert self.preference_model.validate_preference_pair(invalid_pair) == False