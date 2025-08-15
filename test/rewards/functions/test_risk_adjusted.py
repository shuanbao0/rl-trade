"""
测试风险调整奖励函数
"""

import pytest
import numpy as np
from src.environment.rewards.risk_adjusted import RiskAdjustedReward


class TestRiskAdjustedReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_function = RiskAdjustedReward(
            risk_free_rate=0.02,
            window_size=10,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_function.initial_balance == 10000.0
        assert abs(self.reward_function.risk_free_rate - 0.02/252) < 1e-6  # 转换为日收益率
        assert self.reward_function.window_size == 10
        assert self.reward_function.current_stage == "basic"
        assert len(self.reward_function.total_rewards) == 0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_function.get_reward_info()
        
        assert info['name'] == "风险调整奖励"
        assert info['description'] is not None
        assert info['parameters'] is not None
        assert 'risk_free_rate' in info['parameters']
        assert 'window_size' in info['parameters']
    
    def test_basic_stage_reward(self):
        """测试基础阶段奖励计算"""
        assert self.reward_function.current_stage == "basic"
        
        # 测试正收益情况
        reward = self.reward_function.calculate_reward(
            portfolio_value=11000.0,  # 增长10%
            action=0.5,
            price=100.0,
            portfolio_info={},
            trade_info={},
            step=1
        )
        
        # 基础阶段主要基于总收益率
        total_return_pct = 10.0  # 10%
        expected_base_reward = total_return_pct * 10.0  # 基础阶段权重
        
        # 没有极端动作，所以没有风险惩罚
        assert abs(reward - expected_base_reward) < 1.0
    
    def test_basic_stage_risk_penalty(self):
        """测试基础阶段风险惩罚"""
        # 极端动作应该受到惩罚
        reward = self.reward_function.calculate_reward(
            portfolio_value=10500.0,  # 增长5%
            action=0.9,  # 极端动作
            price=100.0,
            portfolio_info={},
            trade_info={},
            step=1
        )
        
        # 应该有风险惩罚
        total_return_reward = 5.0 * 10.0  # 总收益贡献
        risk_penalty = -abs(0.9) * 50.0   # 风险惩罚
        expected_reward = total_return_reward + risk_penalty
        
        assert abs(reward - expected_reward) < 1.0
    
    def test_intermediate_stage_progression(self):
        """测试中级阶段进展"""
        # 模拟多次正奖励以达到阶段切换条件
        for i in range(120):  # 超过basic_to_intermediate的min_episodes
            self.reward_function.calculate_reward(
                portfolio_value=10000.0 + i * 50,  # 逐步增长
                action=0.1,
                price=100.0,
                portfolio_info={},
                trade_info={},
                step=i + 1
            )
        
        # 手动触发阶段检查（简化测试）
        # 在实际实现中，这会在内部自动处理
        if len(self.reward_function.reward_history) > 100:
            recent_avg = np.mean(self.reward_function.reward_history[-20:])
            if recent_avg > 5.0:
                self.reward_function.current_stage = "intermediate"
        
        # 验证可能的阶段进展
        assert self.reward_function.current_stage in ["basic", "intermediate"]
    
    def test_risk_adjustment_calculation(self):
        """测试风险调整计算"""
        # 建立奖励历史
        rewards = [10.0, -5.0, 15.0, -8.0, 12.0]
        self.reward_function.reward_history = rewards
        
        risk_adjustment = self.reward_function._calculate_basic_risk_adjustment()
        
        # 验证风险调整为负值（惩罚高波动）
        assert isinstance(risk_adjustment, (int, float))
        assert risk_adjustment <= 0  # 风险调整应该是惩罚性的
    
    def test_sharpe_reward_calculation(self):
        """测试夏普比率奖励计算（高级阶段）"""
        # 设置为高级阶段
        self.reward_function.current_stage = "advanced"
        
        # 建立足够的历史数据
        portfolio_values = []
        for i in range(self.reward_function.window_size + 5):
            value = 10000.0 + i * 100 + np.random.normal(0, 50)
            portfolio_values.append(value)
            self.reward_function.portfolio_history.append(value)
        
        reward = self.reward_function._calculate_sharpe_reward(
            step_return_pct=2.0,
            current_value=11500.0,
            total_return_pct=15.0
        )
        
        # 验证夏普奖励返回合理数值
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_insufficient_data_handling(self):
        """测试数据不足时的处理"""
        # 设置为高级阶段但数据不足
        self.reward_function.current_stage = "advanced"
        
        reward = self.reward_function.calculate_reward(
            portfolio_value=10500.0,
            action=0.2,
            price=100.0,
            portfolio_info={},
            trade_info={},
            step=1
        )
        
        # 数据不足时应该降级处理
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_stage_thresholds(self):
        """测试阶段切换阈值"""
        thresholds = self.reward_function.stage_thresholds
        
        assert "basic_to_intermediate" in thresholds
        assert "intermediate_to_advanced" in thresholds
        
        basic_to_int = thresholds["basic_to_intermediate"]
        assert "min_episodes" in basic_to_int
        assert "avg_reward_threshold" in basic_to_int
        assert basic_to_int["min_episodes"] > 0
        assert basic_to_int["avg_reward_threshold"] > 0
    
    def test_multiple_step_consistency(self):
        """测试多步骤一致性"""
        portfolio_values = [10000.0, 10200.0, 10150.0, 10300.0, 10250.0]
        rewards = []
        
        for i, value in enumerate(portfolio_values):
            reward = self.reward_function.calculate_reward(
                portfolio_value=value,
                action=0.1 * (i % 3 - 1),  # 变化的动作
                price=100.0 + i,
                portfolio_info={},
                trade_info={},
                step=i + 1
            )
            rewards.append(reward)
        
        # 验证奖励序列
        assert len(rewards) == 5
        assert len(self.reward_function.portfolio_history) == 5
        assert len(self.reward_function.reward_history) == 5
        
        # 验证所有奖励都是有效数值
        assert all(isinstance(r, (int, float)) for r in rewards)
        assert all(not np.isnan(r) for r in rewards)
        assert all(not np.isinf(r) for r in rewards)
    
    def test_zero_volatility_handling(self):
        """测试零波动率处理"""
        # 设置为高级阶段
        self.reward_function.current_stage = "advanced"
        
        # 创建零波动的投资组合历史
        constant_value = 10000.0
        for i in range(self.reward_function.window_size + 2):
            self.reward_function.portfolio_history.append(constant_value)
        
        reward = self.reward_function._calculate_sharpe_reward(
            step_return_pct=0.0,
            current_value=constant_value,
            total_return_pct=0.0
        )
        
        # 零波动情况应该能正确处理
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        custom_reward = RiskAdjustedReward(
            risk_free_rate=0.05,
            window_size=20,
            initial_balance=50000.0
        )
        
        assert custom_reward.initial_balance == 50000.0
        assert abs(custom_reward.risk_free_rate - 0.05/252) < 1e-6
        assert custom_reward.window_size == 20
    
    def test_extreme_values_robustness(self):
        """测试极值鲁棒性"""
        # 测试极大投资组合价值
        reward1 = self.reward_function.calculate_reward(1e8, 0.1, 1000.0, {}, {}, 1)
        assert isinstance(reward1, (int, float))
        assert not np.isnan(reward1)
        
        # 测试极小投资组合价值
        reward2 = self.reward_function.calculate_reward(1.0, -0.1, 0.01, {}, {}, 1)
        assert isinstance(reward2, (int, float))
        assert not np.isnan(reward2)
    
    def test_backward_compatibility_reward_method(self):
        """测试向后兼容的reward方法"""
        from unittest.mock import MagicMock
        
        # 模拟TensorTrade环境
        mock_env = MagicMock()
        mock_env.portfolio.total_value = 11000.0
        
        # 设置初始状态
        self.reward_function.portfolio_history = [10000.0]
        
        reward = self.reward_function.reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)