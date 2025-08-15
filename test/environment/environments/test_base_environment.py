"""
基础环境测试 - Base Environment Tests

测试基础环境类和配置的正确性。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd

from src.rewards.environments.base_environment import (
    EnvironmentConfiguration, PortfolioState, MarketState,
    BaseTradingEnvironment, BasePortfolioManager, BaseMarketDataProvider,
    EnvironmentFactory, create_trading_environment
)
from src.rewards.enums.market_types import MarketType
from src.rewards.enums.time_granularities import TimeGranularity
from src.rewards.enums.risk_profiles import RiskProfile
from src.rewards.enums.environment_features import EnvironmentFeature
from src.rewards.core.reward_context import RewardContext
from src.rewards.core.base_reward import BaseReward


class TestEnvironmentConfiguration(unittest.TestCase):
    """测试环境配置"""
    
    def test_environment_configuration_creation(self):
        """测试环境配置创建"""
        config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            initial_balance=20000.0,
            transaction_costs=0.002,
            window_size=30
        )
        
        self.assertEqual(config.market_type, MarketType.STOCK)
        self.assertEqual(config.initial_balance, 20000.0)
        self.assertEqual(config.transaction_costs, 0.002)
        self.assertEqual(config.window_size, 30)
        self.assertTrue(config.normalize_observations)
    
    def test_portfolio_state(self):
        """测试投资组合状态"""
        state = PortfolioState(
            cash=10000.0,
            positions={"AAPL": 100.0},
            total_value=15000.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
            transaction_costs_paid=50.0,
            net_worth_history=[10000.0, 12000.0, 15000.0]
        )
        
        self.assertEqual(state.cash, 10000.0)
        self.assertEqual(state.positions["AAPL"], 100.0)
        self.assertEqual(len(state.net_worth_history), 3)
    
    def test_market_state(self):
        """测试市场状态"""
        state = MarketState(
            current_price={"AAPL": 150.0},
            price_history=pd.DataFrame({"price": [140, 145, 150]}),
            volume={"AAPL": 1000000},
            spread={"AAPL": 0.01},
            volatility={"AAPL": 0.02},
            technical_indicators={"rsi": 60.0, "macd": 0.5}
        )
        
        self.assertEqual(state.current_price["AAPL"], 150.0)
        self.assertEqual(state.technical_indicators["rsi"], 60.0)


class MockPortfolioManager(BasePortfolioManager):
    """测试用投资组合管理器"""
    
    def execute_action(self, action, market_state):
        return {
            "success": True,
            "trade_value": abs(float(action)) * 1000,
            "action_type": "buy" if float(action) > 0 else "sell"
        }
    
    def update_portfolio_value(self, market_state):
        # 简单更新逻辑
        self.portfolio_state.total_value = self.portfolio_state.cash + 1000
        self.portfolio_state.net_worth_history.append(self.portfolio_state.total_value)
    
    def get_portfolio_metrics(self):
        return {
            'total_return': 0.1,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.05,
            'volatility': 0.15
        }


class MockMarketDataProvider(BaseMarketDataProvider):
    """测试用市场数据提供者"""
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.feature_columns = ["price", "volume"]
    
    def get_market_state(self, step):
        return MarketState(
            current_price={"TEST": 100.0 + step},
            price_history=pd.DataFrame(),
            volume={"TEST": 1000},
            spread={"TEST": 0.01},
            volatility={"TEST": 0.02},
            technical_indicators={}
        )
    
    def get_observation(self, step):
        # 返回固定形状的观察
        return np.random.rand(self.config.window_size, 2).astype(np.float32)
    
    def get_feature_columns(self):
        return self.feature_columns
    
    def validate_data(self):
        return True, []


class MockTradingEnvironment(BaseTradingEnvironment):
    """测试用交易环境"""
    
    def _create_portfolio_manager(self):
        return MockPortfolioManager(self.config)
    
    def _create_market_data_provider(self):
        return MockMarketDataProvider(self.config, self.data)
    
    def _create_action_space(self):
        from gymnasium import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def _create_observation_space(self):
        from gymnasium import spaces
        n_features = 2  # price, volume
        if self.config.include_positions:
            n_features += 2
        if self.config.include_portfolio_metrics:
            n_features += 4
        
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.window_size, n_features),
            dtype=np.float32
        )


class MockReward(BaseReward):
    """测试用奖励函数"""
    
    def calculate(self, context: RewardContext) -> float:
        return context.portfolio_value / 10000.0 - 1.0
    
    def reset(self):
        pass
    
    def get_info(self):
        return {"type": "mock_reward"}


class TestBaseTradingEnvironment(unittest.TestCase):
    """测试基础交易环境"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.data = pd.DataFrame({
            'price': np.random.rand(100) * 100 + 50,
            'volume': np.random.rand(100) * 1000
        })
        
        # 创建测试配置
        self.config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            window_size=10,
            max_steps=50
        )
        
        # 创建奖励函数
        self.reward_function = MockReward()
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        env = MockTradingEnvironment(self.config, self.data, self.reward_function)
        
        self.assertEqual(env.config.market_type, MarketType.STOCK)
        self.assertIsNotNone(env.portfolio_manager)
        self.assertIsNotNone(env.market_data_provider)
        self.assertIsNotNone(env.action_space)
        self.assertIsNotNone(env.observation_space)
    
    def test_environment_reset(self):
        """测试环境重置"""
        env = MockTradingEnvironment(self.config, self.data, self.reward_function)
        
        observation, info = env.reset()
        
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape[0], self.config.window_size)
        self.assertIsInstance(info, dict)
        self.assertIn("portfolio", info)
        self.assertIn("market", info)
        self.assertEqual(env.current_step, self.config.window_size)
        self.assertFalse(env.done)
    
    def test_environment_step(self):
        """测试环境步进"""
        env = MockTradingEnvironment(self.config, self.data, self.reward_function)
        env.reset()
        
        action = np.array([0.5], dtype=np.float32)
        observation, reward, done, truncated, info = env.step(action)
        
        self.assertIsInstance(observation, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
        self.assertEqual(env.current_step, self.config.window_size + 1)
        self.assertIn("trade_result", info)
    
    def test_portfolio_features(self):
        """测试投资组合特征"""
        env = MockTradingEnvironment(self.config, self.data, self.reward_function)
        
        position_features = env._get_position_features()
        self.assertEqual(len(position_features), 2)  # 现金比例 + 仓位比例
        
        metric_features = env._get_portfolio_metric_features()
        self.assertEqual(len(metric_features), 4)  # 4个投资组合指标
    
    def test_drawdown_calculation(self):
        """测试回撤计算"""
        env = MockTradingEnvironment(self.config, self.data, self.reward_function)
        
        # 设置测试历史数据
        env.portfolio_manager.portfolio_state.net_worth_history = [10000, 12000, 8000, 9000]
        
        drawdown = env._calculate_drawdown()
        expected_drawdown = (12000 - 9000) / 12000  # 从峰值的回撤
        
        self.assertAlmostEqual(drawdown, expected_drawdown, places=4)
    
    def test_done_conditions(self):
        """测试结束条件"""
        env = MockTradingEnvironment(self.config, self.data, self.reward_function)
        env.reset()
        
        # 测试最大步数
        env.current_step = self.config.max_steps
        self.assertTrue(env._check_done())
        
        # 测试投资组合破产
        env.current_step = 20
        env.portfolio_manager.portfolio_state.total_value = 0
        self.assertTrue(env._check_done())
    
    def test_portfolio_summary(self):
        """测试投资组合摘要"""
        env = MockTradingEnvironment(self.config, self.data, self.reward_function)
        env.reset()
        
        summary = env.get_portfolio_summary()
        
        self.assertIn("initial_balance", summary)
        self.assertIn("final_value", summary)
        self.assertIn("total_return", summary)
        self.assertIn("metrics", summary)
        self.assertIn("config", summary)
        
        self.assertEqual(summary["initial_balance"], self.config.initial_balance)


class TestEnvironmentFactory(unittest.TestCase):
    """测试环境工厂"""
    
    def setUp(self):
        """设置测试环境"""
        self.data = pd.DataFrame({
            'close': np.random.rand(100) * 100 + 50,
            'volume': np.random.rand(100) * 1000
        })
    
    def test_create_trading_environment_function(self):
        """测试便利函数创建环境"""
        env = create_trading_environment(
            market_type="stock",
            time_granularity="1d",
            data=self.data,
            risk_profile="balanced"
        )
        
        self.assertIsNotNone(env)
        self.assertEqual(env.config.market_type, MarketType.STOCK)
        self.assertEqual(env.config.time_granularity, TimeGranularity.DAY_1)
    
    def test_environment_features_auto_detection(self):
        """测试环境特征自动检测"""
        # 测试高频交易特征
        env_hf = create_trading_environment(
            market_type="forex",
            time_granularity="1min",
            data=self.data
        )
        
        self.assertTrue(env_hf.config.environment_features & EnvironmentFeature.HIGH_FREQUENCY)
        self.assertTrue(env_hf.config.environment_features & EnvironmentFeature.REAL_TIME_DATA)
        
        # 测试股票特征
        env_stock = create_trading_environment(
            market_type="stock",
            time_granularity="1d",
            data=self.data
        )
        
        self.assertTrue(env_stock.config.environment_features & EnvironmentFeature.NEWS_DRIVEN)
        self.assertTrue(env_stock.config.environment_features & EnvironmentFeature.EARNINGS_SENSITIVE)


class TestEnvironmentValidation(unittest.TestCase):
    """测试环境验证"""
    
    def test_invalid_market_time_combination(self):
        """测试无效的市场-时间粒度组合"""
        data = pd.DataFrame({'price': [100, 101, 102]})
        
        with self.assertRaises(ValueError):
            # 尝试创建不兼容的组合
            create_trading_environment(
                market_type="bond",  # 债券市场
                time_granularity="1s",  # 秒级数据（不兼容）
                data=data
            )
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        # 创建太少的数据
        small_data = pd.DataFrame({'price': [100, 101]})
        
        with self.assertRaises(ValueError):
            create_trading_environment(
                market_type="stock",
                time_granularity="1d",
                data=small_data,
                window_size=50  # 窗口大小大于数据长度
            )


if __name__ == '__main__':
    unittest.main()