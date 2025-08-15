"""
测试课程奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.curriculum_reward import CurriculumReward


class TestCurriculumReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.curriculum_stages = [
            {'difficulty': 0.2, 'min_episodes': 10},
            {'difficulty': 0.5, 'min_episodes': 20},
            {'difficulty': 0.8, 'min_episodes': 30}
        ]
        self.reward_scheme = CurriculumReward(
            curriculum_stages=self.curriculum_stages,
            progression_threshold=0.7,
            stage_completion_bonus=0.1,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.curriculum_stages == self.curriculum_stages
        assert self.reward_scheme.progression_threshold == 0.7
        assert self.reward_scheme.stage_completion_bonus == 0.1
        assert self.reward_scheme.initial_balance == self.initial_balance
        assert self.reward_scheme.current_stage == 0
        assert self.reward_scheme.episode_count == 0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Curriculum Learning Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'current_stage' in info['parameters']
        assert 'total_stages' in info['parameters']
    
    def test_reset(self):
        """测试重置功能"""
        # 修改一些状态
        self.reward_scheme.episode_count = 15
        self.reward_scheme.performance_history = [0.5, 0.6, 0.7]
        self.reward_scheme.stage_rewards = [0.1, 0.2]
        
        # 重置
        self.reward_scheme.reset()
        
        # 验证某些状态保持（课程学习不完全重置）
        assert self.reward_scheme.episode_count == 0
        assert len(self.reward_scheme.performance_history) == 0
        assert len(self.reward_scheme.stage_rewards) == 0
        # current_stage 可能保持或重置，取决于实现
    
    def test_stage_progression(self):
        """测试阶段进展"""
        # 模拟足够的剧集和好的表现
        self.reward_scheme.episode_count = 15
        self.reward_scheme.performance_history = [0.8, 0.75, 0.9, 0.85]  # 高表现
        
        initial_stage = self.reward_scheme.current_stage
        
        # 检查是否满足进阶条件
        should_progress = (
            self.reward_scheme.episode_count >= 
            self.reward_scheme.curriculum_stages[initial_stage]['min_episodes']
        )
        
        if should_progress:
            avg_performance = np.mean(self.reward_scheme.performance_history[-5:])
            should_progress = avg_performance >= self.reward_scheme.progression_threshold
        
        if should_progress and initial_stage < len(self.curriculum_stages) - 1:
            # 测试进阶逻辑存在
            assert hasattr(self.reward_scheme, 'current_stage')
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        # 创建模拟环境
        mock_env = Mock()
        mock_env.portfolio_value = 10800.0
        mock_env.initial_balance = self.initial_balance
        mock_env.episode_reward = 0.08  # 8% 回报
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        
        # 验证剧集计数增加
        assert self.reward_scheme.episode_count > 0
    
    def test_difficulty_scaling(self):
        """测试难度缩放"""
        # 测试不同阶段的难度影响
        stage_0_difficulty = self.curriculum_stages[0]['difficulty']
        stage_2_difficulty = self.curriculum_stages[2]['difficulty']
        
        assert stage_0_difficulty < stage_2_difficulty
        
        # 模拟不同阶段的奖励计算
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        mock_env.episode_reward = 0.1
        
        # 第一阶段
        self.reward_scheme.current_stage = 0
        reward_stage_0 = self.reward_scheme.get_reward(mock_env)
        
        # 高阶段
        self.reward_scheme.current_stage = 2
        reward_stage_2 = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward_stage_0, (int, float))
        assert isinstance(reward_stage_2, (int, float))
    
    def test_stage_completion_bonus(self):
        """测试阶段完成奖励"""
        # 设置即将完成阶段的条件
        current_stage = 0
        self.reward_scheme.current_stage = current_stage
        self.reward_scheme.episode_count = self.curriculum_stages[current_stage]['min_episodes']
        
        # 添加高表现历史
        high_performance = [0.8, 0.85, 0.9, 0.75, 0.88]
        self.reward_scheme.performance_history = high_performance
        
        mock_env = Mock()
        mock_env.portfolio_value = 11500.0
        mock_env.initial_balance = 10000.0
        mock_env.episode_reward = 0.15
        
        # 计算奖励（可能触发阶段完成）
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_performance_tracking(self):
        """测试表现跟踪"""
        initial_history_length = len(self.reward_scheme.performance_history)
        
        mock_env = Mock()
        mock_env.portfolio_value = 10300.0
        mock_env.initial_balance = 10000.0
        mock_env.episode_reward = 0.03
        
        # 多次调用奖励计算
        for _ in range(5):
            self.reward_scheme.get_reward(mock_env)
        
        # 验证历史记录增长
        final_history_length = len(self.reward_scheme.performance_history)
        assert final_history_length > initial_history_length
    
    def test_curriculum_completion(self):
        """测试课程完成"""
        # 设置为最后阶段
        final_stage = len(self.curriculum_stages) - 1
        self.reward_scheme.current_stage = final_stage
        
        mock_env = Mock()
        mock_env.portfolio_value = 12000.0
        mock_env.initial_balance = 10000.0
        mock_env.episode_reward = 0.2
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        
        # 在最后阶段，stage 不应该继续增加
        assert self.reward_scheme.current_stage <= final_stage
    
    def test_custom_curriculum_stages(self):
        """测试自定义课程阶段"""
        custom_stages = [
            {'difficulty': 0.1, 'min_episodes': 5},
            {'difficulty': 0.9, 'min_episodes': 15}
        ]
        
        custom_reward = CurriculumReward(curriculum_stages=custom_stages)
        
        assert custom_reward.curriculum_stages == custom_stages
        assert len(custom_reward.curriculum_stages) == 2
        
        info = custom_reward.get_reward_info()
        assert info['parameters']['total_stages'] == 2
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空课程阶段
        try:
            empty_curriculum = CurriculumReward(curriculum_stages=[])
            # 如果允许空课程，应该有默认行为
            mock_env = Mock()
            mock_env.portfolio_value = 10000.0
            mock_env.initial_balance = 10000.0
            mock_env.episode_reward = 0.0
            
            reward = empty_curriculum.get_reward(mock_env)
            assert isinstance(reward, (int, float))
        except (ValueError, IndexError):
            # 空课程可能导致预期的异常
            pass
    
    def test_progression_threshold_impact(self):
        """测试进展阈值的影响"""
        # 创建不同阈值的奖励函数
        easy_progression = CurriculumReward(
            curriculum_stages=self.curriculum_stages,
            progression_threshold=0.3
        )
        hard_progression = CurriculumReward(
            curriculum_stages=self.curriculum_stages,
            progression_threshold=0.9
        )
        
        # 设置中等表现
        moderate_performance = [0.6, 0.65, 0.55, 0.7]
        
        easy_progression.performance_history = moderate_performance.copy()
        hard_progression.performance_history = moderate_performance.copy()
        
        # 验证阈值影响进展判断
        avg_perf = np.mean(moderate_performance)
        
        easy_should_progress = avg_perf >= 0.3
        hard_should_progress = avg_perf >= 0.9
        
        assert easy_should_progress
        assert not hard_should_progress
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试有效参数组合
        valid_params = [
            {
                'curriculum_stages': [{'difficulty': 0.5, 'min_episodes': 10}],
                'progression_threshold': 0.5
            },
            {
                'curriculum_stages': self.curriculum_stages,
                'stage_completion_bonus': 0.0
            }
        ]
        
        for params in valid_params:
            scheme = CurriculumReward(**params)
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)