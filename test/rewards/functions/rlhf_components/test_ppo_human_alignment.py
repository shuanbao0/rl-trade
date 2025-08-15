"""
测试PPO人类对齐模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.rlhf_components.ppo_human_alignment import PPOWithHumanAlignment


class TestPPOWithHumanAlignment:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.ppo_alignment = PPOWithHumanAlignment(
            state_dim=50,
            action_dim=3,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.ppo_alignment.state_dim == 50
        assert self.ppo_alignment.action_dim == 3
        assert self.ppo_alignment.lr_actor == 0.0003
        assert self.ppo_alignment.lr_critic == 0.001
        assert self.ppo_alignment.gamma == 0.99
        assert self.ppo_alignment.eps_clip == 0.2
        assert self.ppo_alignment.k_epochs == 4
    
    def test_action_selection(self):
        """测试动作选择"""
        mock_state = np.random.rand(50)
        
        action, action_logprob = self.ppo_alignment.select_action(mock_state)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert isinstance(action_logprob, (int, float))
    
    def test_human_feedback_integration(self):
        """测试人类反馈集成"""
        trajectory = {
            'states': [np.random.rand(50) for _ in range(10)],
            'actions': [np.random.rand(3) for _ in range(10)],
            'rewards': [np.random.uniform(-1, 1) for _ in range(10)],
            'human_feedback': [np.random.uniform(-1, 1) for _ in range(10)]
        }
        
        alignment_loss = self.ppo_alignment.integrate_human_feedback(trajectory)
        
        assert isinstance(alignment_loss, (int, float))
        assert alignment_loss >= 0
    
    def test_ppo_update(self):
        """测试PPO更新"""
        batch_data = {
            'states': np.random.rand(32, 50),
            'actions': np.random.rand(32, 3),
            'old_logprobs': np.random.rand(32),
            'rewards': np.random.rand(32),
            'advantages': np.random.rand(32),
            'returns': np.random.rand(32),
            'human_rewards': np.random.rand(32)
        }
        
        update_metrics = self.ppo_alignment.update(batch_data)
        
        assert isinstance(update_metrics, dict)
        assert 'actor_loss' in update_metrics
        assert 'critic_loss' in update_metrics
        assert 'alignment_loss' in update_metrics
    
    def test_advantage_calculation(self):
        """测试优势计算"""
        rewards = np.array([0.1, 0.2, -0.1, 0.3, 0.0])
        values = np.array([0.05, 0.15, -0.05, 0.25, 0.1])
        done_mask = np.array([0, 0, 0, 0, 1])
        
        advantages = self.ppo_alignment.compute_advantages(rewards, values, done_mask)
        
        assert isinstance(advantages, np.ndarray)
        assert advantages.shape == rewards.shape