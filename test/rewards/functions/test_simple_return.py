"""
测试简单收益率奖励函数
"""

import pytest
import numpy as np
from src.environment.rewards.simple_return import SimpleReturnReward


class TestSimpleReturnReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_function = SimpleReturnReward(
            initial_balance=10000.0,
            step_weight=1.0,
            total_weight=0.1
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_function.initial_balance == 10000.0
        assert self.reward_function.step_weight == 1.0
        assert self.reward_function.total_weight == 0.1
        assert len(self.reward_function.portfolio_history) == 0
        assert len(self.reward_function.reward_history) == 0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_function.get_reward_info()
        
        assert info['name'] == "简单收益率奖励"
        assert info['description'] is not None
        assert info['parameters'] is not None
    
    def test_first_step_reward(self):
        """测试第一步的奖励计算"""
        portfolio_value = 10000.0
        
        reward = self.reward_function.calculate_reward(
            portfolio_value=portfolio_value,
            action=0.5,
            price=100.0,
            portfolio_info={},
            trade_info={},
            step=1
        )
        
        # 第一步应该没有步骤收益率，只有总收益率贡献
        expected_reward = 0.0 * 1.0 + 0.0 * 0.1  # 无收益变化
        assert reward == expected_reward
        assert len(self.reward_function.portfolio_history) == 1
        assert len(self.reward_function.reward_history) == 1
    
    def test_positive_return_reward(self):
        """测试正收益率奖励"""
        # 第一步
        self.reward_function.calculate_reward(10000.0, 0.5, 100.0, {}, {}, 1)
        
        # 第二步 - 投资组合增值
        portfolio_value = 11000.0  # 增长10%
        
        reward = self.reward_function.calculate_reward(
            portfolio_value=portfolio_value,
            action=0.3,
            price=110.0,
            portfolio_info={},
            trade_info={},
            step=2
        )
        
        # 步骤收益率 = (11000 - 10000) / 10000 * 100 = 10%
        # 总收益率 = (11000 - 10000) / 10000 * 100 = 10%
        step_return = 10.0
        total_return = 10.0
        expected_reward = step_return * 1.0 + total_return * 0.1
        
        assert abs(reward - expected_reward) < 0.01
    
    def test_negative_return_reward(self):
        """测试负收益率奖励"""
        # 第一步
        self.reward_function.calculate_reward(10000.0, 0.5, 100.0, {}, {}, 1)
        
        # 第二步 - 投资组合贬值
        portfolio_value = 9000.0  # 下降10%
        
        reward = self.reward_function.calculate_reward(
            portfolio_value=portfolio_value,
            action=-0.3,
            price=90.0,
            portfolio_info={},
            trade_info={},
            step=2
        )
        
        # 步骤收益率 = (9000 - 10000) / 10000 * 100 = -10%
        # 总收益率 = (9000 - 10000) / 10000 * 100 = -10%
        step_return = -10.0
        total_return = -10.0
        expected_reward = step_return * 1.0 + total_return * 0.1
        
        assert abs(reward - expected_reward) < 0.01
    
    def test_multiple_steps(self):
        """测试多步奖励计算"""
        portfolio_values = [10000.0, 10500.0, 11000.0, 10800.0]
        rewards = []
        
        for i, value in enumerate(portfolio_values):
            reward = self.reward_function.calculate_reward(
                portfolio_value=value,
                action=0.1 * i,
                price=100.0 + i * 5,
                portfolio_info={},
                trade_info={},
                step=i + 1
            )
            rewards.append(reward)
        
        # 验证奖励序列合理性
        assert len(rewards) == 4
        assert len(self.reward_function.portfolio_history) == 4
        assert len(self.reward_function.reward_history) == 4
        
        # 第二步应该是正奖励（增长）
        assert rewards[1] > 0
        
        # 第四步应该是负奖励（下降）
        assert rewards[3] < 0
    
    def test_zero_previous_value_handling(self):
        """测试处理零前值的情况"""
        # 手动设置前值为0的异常情况
        self.reward_function.portfolio_history = [0.0]
        
        reward = self.reward_function.calculate_reward(
            portfolio_value=10000.0,
            action=0.0,
            price=100.0,
            portfolio_info={},
            trade_info={},
            step=2
        )
        
        # 应该处理除零情况，步骤收益率为0
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_custom_weights(self):
        """测试自定义权重"""
        custom_reward = SimpleReturnReward(
            initial_balance=10000.0,
            step_weight=2.0,
            total_weight=0.5
        )
        
        # 第一步
        custom_reward.calculate_reward(10000.0, 0.5, 100.0, {}, {}, 1)
        
        # 第二步 - 增长10%
        reward = custom_reward.calculate_reward(11000.0, 0.3, 110.0, {}, {}, 2)
        
        # 验证权重影响
        step_return = 10.0
        total_return = 10.0
        expected_reward = step_return * 2.0 + total_return * 0.5
        
        assert abs(reward - expected_reward) < 0.01
    
    def test_reward_history_tracking(self):
        """测试奖励历史追踪"""
        portfolio_values = [10000.0, 10200.0, 10100.0]
        
        for i, value in enumerate(portfolio_values):
            self.reward_function.calculate_reward(value, 0.1, 100.0, {}, {}, i + 1)
        
        # 验证历史记录
        assert len(self.reward_function.reward_history) == 3
        assert len(self.reward_function.portfolio_history) == 3
        
        # 验证历史数据类型
        assert all(isinstance(r, (int, float)) for r in self.reward_function.reward_history)
        assert all(isinstance(v, (int, float)) for v in self.reward_function.portfolio_history)
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 测试旧的reward方法接口
        from unittest.mock import MagicMock
        
        # 模拟TensorTrade环境对象
        mock_env = MagicMock()
        mock_env.portfolio.total_value = 10500.0
        
        # 设置一些初始状态
        self.reward_function.portfolio_history = [10000.0]
        
        # 调用兼容方法
        reward = self.reward_function.reward(mock_env)
        
        # 应该返回数值型结果
        assert isinstance(reward, (int, float))
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试极小投资组合价值
        reward1 = self.reward_function.calculate_reward(0.01, 0.0, 100.0, {}, {}, 1)
        assert isinstance(reward1, (int, float))
        
        # 测试极大投资组合价值
        reward2 = self.reward_function.calculate_reward(1e10, 0.0, 100.0, {}, {}, 1)
        assert isinstance(reward2, (int, float))
        
        # 测试负价格（异常情况）
        reward3 = self.reward_function.calculate_reward(10000.0, 0.0, -100.0, {}, {}, 1)
        assert isinstance(reward3, (int, float))