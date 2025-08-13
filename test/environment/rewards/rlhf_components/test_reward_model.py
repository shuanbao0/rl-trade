"""
测试奖励模型模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.rlhf_components.reward_model import RewardModelTrainer


class TestRewardModelTrainer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_model = RewardModelTrainer(
            state_dim=50,
            action_dim=3,
            hidden_dims=[128, 64, 32],
            learning_rate=0.001
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_model.state_dim == 50
        assert self.reward_model.action_dim == 3
        assert self.reward_model.hidden_dims == [128, 64, 32]
        assert self.reward_model.learning_rate == 0.001
    
    def test_reward_prediction(self):
        """测试奖励预测"""
        mock_state = np.random.rand(50)
        mock_action = np.random.rand(3)
        
        predicted_reward = self.reward_model.predict_reward(mock_state, mock_action)
        
        assert isinstance(predicted_reward, (int, float))
        assert not np.isnan(predicted_reward)
    
    def test_model_update(self):
        """测试模型更新"""
        training_data = []
        for _ in range(20):
            state = np.random.rand(50)
            action = np.random.rand(3)
            human_rating = np.random.uniform(-1, 1)
            
            training_data.append({
                'state': state,
                'action': action,
                'human_rating': human_rating,
                'confidence': np.random.uniform(0.5, 1.0)
            })
        
        training_loss = self.reward_model.update_model(training_data, epochs=10)
        
        assert isinstance(training_loss, list)
        assert len(training_loss) == 10
        assert all(isinstance(loss, (int, float)) for loss in training_loss)
    
    def test_uncertainty_estimation(self):
        """测试不确定性估计"""
        mock_state = np.random.rand(50)
        mock_action = np.random.rand(3)
        
        reward, uncertainty = self.reward_model.predict_with_uncertainty(mock_state, mock_action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(uncertainty, (int, float))
        assert uncertainty >= 0
    
    def test_model_evaluation(self):
        """测试模型评估"""
        test_data = []
        for _ in range(10):
            state = np.random.rand(50)
            action = np.random.rand(3)
            true_rating = np.random.uniform(-1, 1)
            
            test_data.append({
                'state': state,
                'action': action,
                'true_rating': true_rating
            })
        
        evaluation_metrics = self.reward_model.evaluate(test_data)
        
        assert isinstance(evaluation_metrics, dict)
        assert 'mse' in evaluation_metrics
        assert 'mae' in evaluation_metrics
        assert 'correlation' in evaluation_metrics