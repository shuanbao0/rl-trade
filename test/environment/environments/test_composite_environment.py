"""
多市场组合环境测试 - Composite Environment Tests

测试多市场统一交易环境和组合管理功能。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.rewards.environments.composite_environment import (
    AllocationStrategy, RebalanceFrequency, MarketWeight, CompositeEnvironmentConfig,
    MarketPosition, CompositePortfolio, BaseCompositeEnvironment,
    MultiMarketCompositeEnvironment, create_composite_environment,
    create_multi_asset_config, analyze_portfolio_correlation
)
from src.rewards.environments.base_environment import EnvironmentConfiguration
from src.rewards.enums.market_types import MarketType
from src.rewards.enums.time_granularities import TimeGranularity
from src.rewards.enums.risk_profiles import RiskProfile
from src.rewards.enums.environment_features import EnvironmentFeature


class MockTradingEnvironment:
    """测试用交易环境"""
    
    def __init__(self, config: EnvironmentConfiguration):
        self.config = config
        self.portfolio_value = config.initial_balance
        self.daily_return = 0.0
        self.current_step = 0
        self.is_done = False
    
    def reset(self):
        self.portfolio_value = self.config.initial_balance
        self.daily_return = 0.0
        self.current_step = 0
        self.is_done = False
        return np.array([1.0, 0.0, 0.5, 0.2])  # 模拟观察值
    
    def step(self, action):
        self.current_step += 1
        
        # 模拟价格变化和奖励
        price_change = np.random.normal(0.001, 0.02)
        self.daily_return = price_change
        self.portfolio_value *= (1 + price_change)
        
        reward = price_change * action  # 简化的奖励计算
        done = self.current_step >= 100 or self.portfolio_value <= self.config.initial_balance * 0.1
        self.is_done = done
        
        obs = np.array([
            self.portfolio_value / self.config.initial_balance,
            action,
            price_change,
            self.current_step / 100.0
        ])
        
        info = {
            'portfolio_value': self.portfolio_value,
            'daily_return': self.daily_return,
            'action': action
        }
        
        return obs, reward, done, info


class TestAllocationStrategy(unittest.TestCase):
    """测试分配策略枚举"""
    
    def test_allocation_strategies(self):
        """测试分配策略枚举值"""
        strategies = list(AllocationStrategy)
        expected_strategies = [
            AllocationStrategy.EQUAL_WEIGHT,
            AllocationStrategy.MARKET_CAP_WEIGHT,
            AllocationStrategy.VOLATILITY_WEIGHT,
            AllocationStrategy.RISK_PARITY,
            AllocationStrategy.MOMENTUM_WEIGHT,
            AllocationStrategy.CUSTOM_WEIGHT,
            AllocationStrategy.DYNAMIC_OPTIMIZATION
        ]
        
        self.assertEqual(len(strategies), len(expected_strategies))
        for strategy in expected_strategies:
            self.assertIn(strategy, strategies)


class TestRebalanceFrequency(unittest.TestCase):
    """测试再平衡频率枚举"""
    
    def test_rebalance_frequencies(self):
        """测试再平衡频率枚举值"""
        frequencies = list(RebalanceFrequency)
        expected_frequencies = [
            RebalanceFrequency.NEVER,
            RebalanceFrequency.DAILY,
            RebalanceFrequency.WEEKLY,
            RebalanceFrequency.MONTHLY,
            RebalanceFrequency.QUARTERLY,
            RebalanceFrequency.PERFORMANCE_BASED,
            RebalanceFrequency.THRESHOLD_BASED
        ]
        
        self.assertEqual(len(frequencies), len(expected_frequencies))
        for frequency in expected_frequencies:
            self.assertIn(frequency, frequencies)


class TestMarketWeight(unittest.TestCase):
    """测试市场权重"""
    
    def test_market_weight_creation(self):
        """测试市场权重创建"""
        weight = MarketWeight(
            market_type=MarketType.STOCK,
            symbol="AAPL",
            weight=0.3,
            min_weight=0.1,
            max_weight=0.5,
            constraints={"sector": "technology"}
        )
        
        self.assertEqual(weight.market_type, MarketType.STOCK)
        self.assertEqual(weight.symbol, "AAPL")
        self.assertEqual(weight.weight, 0.3)
        self.assertEqual(weight.min_weight, 0.1)
        self.assertEqual(weight.max_weight, 0.5)
        self.assertEqual(weight.constraints["sector"], "technology")
    
    def test_market_weight_defaults(self):
        """测试市场权重默认值"""
        weight = MarketWeight(
            market_type=MarketType.FOREX,
            symbol="EURUSD",
            weight=0.25
        )
        
        self.assertEqual(weight.min_weight, 0.0)
        self.assertEqual(weight.max_weight, 1.0)
        self.assertEqual(len(weight.constraints), 0)


class TestCompositeEnvironmentConfig(unittest.TestCase):
    """测试组合环境配置"""
    
    def setUp(self):
        """设置测试"""
        self.base_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
            initial_balance=100000.0,
            window_size=20,
            max_steps=1000
        )
        
        self.market_configs = {
            "STOCK_AAPL": self.base_config,
            "FOREX_EURUSD": EnvironmentConfiguration(
                market_type=MarketType.FOREX,
                time_granularity=TimeGranularity.DAY_1,
                risk_profile=RiskProfile.BALANCED,
                environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
                initial_balance=100000.0,
                window_size=20,
                max_steps=1000
            )
        }
    
    def test_config_creation(self):
        """测试配置创建"""
        config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            correlation_threshold=0.7,
            risk_budget=0.03
        )
        
        self.assertEqual(len(config.market_configurations), 2)
        self.assertEqual(config.allocation_strategy, AllocationStrategy.EQUAL_WEIGHT)
        self.assertEqual(config.rebalance_frequency, RebalanceFrequency.WEEKLY)
        self.assertEqual(config.correlation_threshold, 0.7)
        self.assertEqual(config.risk_budget, 0.03)
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.RISK_PARITY,
            rebalance_frequency=RebalanceFrequency.MONTHLY
        )
        
        self.assertEqual(config.correlation_threshold, 0.8)
        self.assertEqual(config.risk_budget, 0.02)
        self.assertTrue(len(config.transaction_costs) == 0)
        self.assertFalse(config.currency_hedging)
        self.assertEqual(config.leverage_limit, 1.0)


class TestCompositePortfolio(unittest.TestCase):
    """测试组合投资组合"""
    
    def test_portfolio_creation(self):
        """测试投资组合创建"""
        position = MarketPosition(
            market_type=MarketType.STOCK,
            symbol="AAPL",
            quantity=100,
            market_value=15000.0,
            weight=0.3,
            last_price=150.0,
            unrealized_pnl=1000.0,
            realized_pnl=500.0
        )
        
        portfolio = CompositePortfolio(
            total_value=50000.0,
            cash=5000.0,
            positions={"AAPL": position},
            market_weights={"STOCK_AAPL": 0.3, "FOREX_EURUSD": 0.7},
            daily_returns=[0.01, -0.005, 0.02],
            performance_metrics={"sharpe_ratio": 1.2},
            risk_metrics={"volatility": 0.15}
        )
        
        self.assertEqual(portfolio.total_value, 50000.0)
        self.assertEqual(portfolio.cash, 5000.0)
        self.assertEqual(len(portfolio.positions), 1)
        self.assertEqual(len(portfolio.market_weights), 2)
        self.assertEqual(len(portfolio.daily_returns), 3)


@patch('src.rewards.environments.factory.TradingEnvironmentFactory.create_environment')
class TestBaseCompositeEnvironment(unittest.TestCase):
    """测试基础组合环境"""
    
    def setUp(self):
        """设置测试"""
        self.base_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
            initial_balance=100000.0,
            window_size=20,
            max_steps=1000
        )
        
        self.market_configs = {
            "STOCK_AAPL": self.base_config,
            "FOREX_EURUSD": EnvironmentConfiguration(
                market_type=MarketType.FOREX,
                time_granularity=TimeGranularity.DAY_1,
                risk_profile=RiskProfile.BALANCED,
                environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
                initial_balance=100000.0,
                window_size=20,
                max_steps=1000
            )
        }
        
        self.composite_config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.WEEKLY
        )
    
    def test_environment_initialization(self, mock_create):
        """测试环境初始化"""
        # 设置mock
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(self.composite_config)
        
        self.assertEqual(len(env.sub_environments), 2)
        self.assertIn("STOCK_AAPL", env.sub_environments)
        self.assertIn("FOREX_EURUSD", env.sub_environments)
        self.assertEqual(env.portfolio.total_value, 100000.0)
        self.assertEqual(len(env.target_weights), 2)
        self.assertEqual(env.target_weights["STOCK_AAPL"], 0.5)
        self.assertEqual(env.target_weights["FOREX_EURUSD"], 0.5)
    
    def test_environment_reset(self, mock_create):
        """测试环境重置"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(self.composite_config)
        
        # 修改一些状态
        env.current_step = 50
        env.portfolio.total_value = 120000.0
        
        # 重置
        obs = env.reset()
        
        self.assertEqual(env.current_step, 0)
        self.assertEqual(env.portfolio.total_value, 100000.0)
        self.assertEqual(env.last_rebalance_step, 0)
        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(len(obs) > 0)
    
    def test_step_execution(self, mock_create):
        """测试步骤执行"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 执行动作
        actions = {
            "STOCK_AAPL": 0.5,
            "FOREX_EURUSD": -0.3
        }
        
        obs, reward, done, info = env.step(actions)
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(env.current_step, 1)
        
        # 检查信息内容
        self.assertIn('portfolio_value', info)
        self.assertIn('market_weights', info)
        self.assertIn('sub_environment_info', info)
    
    def test_rebalance_frequency_never(self, mock_create):
        """测试永不再平衡"""
        config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.NEVER
        )
        
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(config)
        env.reset()
        
        # 执行多步
        for i in range(10):
            should_rebalance = env._should_rebalance()
            self.assertFalse(should_rebalance)
            env.current_step += 1
    
    def test_rebalance_frequency_daily(self, mock_create):
        """测试日度再平衡"""
        config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.DAILY
        )
        
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(config)
        env.reset()
        
        # 每天都应该再平衡
        for i in range(5):
            should_rebalance = env._should_rebalance()
            self.assertTrue(should_rebalance)
            env.current_step += 1
    
    def test_rebalance_frequency_weekly(self, mock_create):
        """测试周度再平衡"""
        config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.WEEKLY
        )
        
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(config)
        env.reset()
        
        # 测试周度再平衡逻辑
        env.current_step = 3
        self.assertFalse(env._should_rebalance())
        
        env.current_step = 5
        self.assertTrue(env._should_rebalance())
        
        env.current_step = 10
        self.assertTrue(env._should_rebalance())
    
    def test_volatility_weight_calculation(self, mock_create):
        """测试波动率权重计算"""
        config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.VOLATILITY_WEIGHT,
            rebalance_frequency=RebalanceFrequency.DAILY
        )
        
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(config)
        env.reset()
        
        # 创建收益率历史数据
        np.random.seed(42)
        returns_data = {
            "STOCK_AAPL": np.random.normal(0.001, 0.02, 25),  # 较高波动率
            "FOREX_EURUSD": np.random.normal(0.001, 0.01, 25)  # 较低波动率
        }
        env.returns_history = pd.DataFrame(returns_data)
        
        # 计算波动率权重
        env._calculate_volatility_weights()
        
        # 低波动率资产应该获得更高权重
        self.assertGreater(env.target_weights["FOREX_EURUSD"], env.target_weights["STOCK_AAPL"])
        
        # 权重总和应该为1
        total_weight = sum(env.target_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_momentum_weight_calculation(self, mock_create):
        """测试动量权重计算"""
        config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.MOMENTUM_WEIGHT,
            rebalance_frequency=RebalanceFrequency.DAILY
        )
        
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(config)
        env.reset()
        
        # 创建不同动量的收益率数据
        stock_returns = [0.002] * 10 + [0.005] * 15  # 正动量
        forex_returns = [-0.001] * 10 + [0.001] * 15  # 较弱动量
        
        returns_data = {
            "STOCK_AAPL": stock_returns,
            "FOREX_EURUSD": forex_returns
        }
        env.returns_history = pd.DataFrame(returns_data)
        
        # 计算动量权重
        env._calculate_momentum_weights()
        
        # 强动量资产应该获得更高权重
        self.assertGreater(env.target_weights["STOCK_AAPL"], env.target_weights["FOREX_EURUSD"])
    
    def test_portfolio_features_extraction(self, mock_create):
        """测试组合特征提取"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 设置一些投资组合状态
        env.portfolio.total_value = 110000.0
        env.portfolio.cash = 10000.0
        env.portfolio.market_weights = {"STOCK_AAPL": 0.6, "FOREX_EURUSD": 0.4}
        env.portfolio.daily_returns = [0.01, -0.005, 0.02, 0.008, -0.003]
        env.current_step = 10
        env.last_rebalance_step = 5
        
        features = env._get_portfolio_features()
        
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)
        
        # 检查特征范围合理性
        self.assertGreater(features[0], 0)  # 标准化价值
        self.assertTrue(0 <= features[1] <= 1)  # 现金比例
    
    def test_done_condition_max_steps(self, mock_create):
        """测试最大步数终止条件"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 设置接近最大步数
        env.current_step = 999
        
        sub_results = {
            "STOCK_AAPL": {"done": False},
            "FOREX_EURUSD": {"done": False}
        }
        
        done = env._check_done_condition(sub_results)
        self.assertFalse(done)
        
        # 达到最大步数
        env.current_step = 1000
        done = env._check_done_condition(sub_results)
        self.assertTrue(done)
    
    def test_done_condition_portfolio_loss(self, mock_create):
        """测试投资组合损失终止条件"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 设置巨大损失
        env.portfolio.total_value = 5000.0  # 95%损失
        
        sub_results = {
            "STOCK_AAPL": {"done": False},
            "FOREX_EURUSD": {"done": False}
        }
        
        done = env._check_done_condition(sub_results)
        self.assertTrue(done)


@patch('src.rewards.environments.factory.TradingEnvironmentFactory.create_environment')
class TestMultiMarketCompositeEnvironment(unittest.TestCase):
    """测试多市场组合环境"""
    
    def setUp(self):
        """设置测试"""
        self.base_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
            initial_balance=100000.0,
            window_size=20,
            max_steps=1000
        )
        
        self.market_configs = {
            "STOCK_AAPL": self.base_config,
            "FOREX_EURUSD": EnvironmentConfiguration(
                market_type=MarketType.FOREX,
                time_granularity=TimeGranularity.DAY_1,
                risk_profile=RiskProfile.BALANCED,
                environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
                initial_balance=100000.0,
                window_size=20,
                max_steps=1000
            )
        }
        
        self.composite_config = CompositeEnvironmentConfig(
            market_configurations=self.market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            correlation_threshold=0.6,
            risk_budget=0.03,
            diversification_requirement={MarketType.STOCK: 0.3, MarketType.FOREX: 0.2}
        )
    
    def test_risk_monitoring(self, mock_create):
        """测试风险监控"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = MultiMarketCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 添加高风险的收益率历史
        high_risk_returns = [-0.05, -0.08, 0.06, -0.04, 0.03] * 4
        env.portfolio.daily_returns = high_risk_returns
        
        # 监控风险应该发出警告
        with self.assertLogs(env.logger, level='WARNING') as log:
            env._monitor_risk()
        
        # 检查是否有风险预算超出的警告
        warning_found = any("Risk budget exceeded" in message for message in log.output)
        self.assertTrue(warning_found)
    
    def test_diversification_check(self, mock_create):
        """测试多样化检查"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = MultiMarketCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 设置不满足多样化要求的权重
        env.portfolio.market_weights = {
            "STOCK_AAPL": 0.1,  # 低于要求的0.3
            "FOREX_EURUSD": 0.9
        }
        
        # 检查多样化应该发出警告
        with self.assertLogs(env.logger, level='WARNING') as log:
            env._check_diversification()
        
        # 检查是否有多样化不足的警告
        warning_found = any("Diversification requirement not met" in message for message in log.output)
        self.assertTrue(warning_found)
    
    def test_correlation_monitoring(self, mock_create):
        """测试相关性监控"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = MultiMarketCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 创建高相关性的收益率数据
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, 35)
        correlated_returns = base_returns + np.random.normal(0, 0.005, 35)  # 高相关
        
        returns_data = {
            "STOCK_AAPL": base_returns,
            "FOREX_EURUSD": correlated_returns
        }
        env.returns_history = pd.DataFrame(returns_data)
        
        # 监控相关性
        env._monitor_correlation()
        
        # 应该检测到高相关性警告
        self.assertTrue(len(env.correlation_warnings) > 0)
        warning_text = " ".join(env.correlation_warnings)
        self.assertIn("High correlation", warning_text)
    
    def test_risk_metrics_calculation(self, mock_create):
        """测试风险指标计算"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = MultiMarketCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 添加收益率历史
        returns = [0.01, -0.02, 0.015, -0.008, 0.005, 0.02, -0.01] * 5
        env.portfolio.daily_returns = returns
        
        # 设置市场权重
        env.portfolio.market_weights = {"STOCK_AAPL": 0.7, "FOREX_EURUSD": 0.3}
        
        risk_metrics = env._calculate_risk_metrics()
        
        # 检查风险指标
        self.assertIn('volatility', risk_metrics)
        self.assertIn('downside_volatility', risk_metrics)
        self.assertIn('var_95', risk_metrics)
        self.assertIn('max_drawdown', risk_metrics)
        self.assertIn('sharpe_ratio', risk_metrics)
        self.assertIn('concentration_risk', risk_metrics)
        
        # 检查值的合理性
        self.assertGreater(risk_metrics['volatility'], 0)
        self.assertGreater(risk_metrics['concentration_risk'], 0)
        self.assertLessEqual(risk_metrics['concentration_risk'], 1.0)
    
    def test_step_with_risk_monitoring(self, mock_create):
        """测试包含风险监控的步骤执行"""
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(self.market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = MultiMarketCompositeEnvironment(self.composite_config)
        env.reset()
        
        # 执行几步来建立历史
        actions = {"STOCK_AAPL": 0.5, "FOREX_EURUSD": -0.3}
        
        for _ in range(5):
            obs, reward, done, info = env.step(actions)
        
        # 检查信息中包含风险指标
        self.assertIn('risk_metrics', info)
        self.assertIn('correlation_warnings', info)
        
        risk_metrics = info['risk_metrics']
        if len(env.portfolio.daily_returns) > 0:
            self.assertIn('volatility', risk_metrics)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便利函数"""
    
    def setUp(self):
        """设置测试"""
        self.base_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
            initial_balance=100000.0,
            window_size=20,
            max_steps=1000
        )
    
    @patch('src.rewards.environments.factory.TradingEnvironmentFactory.create_environment')
    def test_create_composite_environment_function(self, mock_create):
        """测试创建组合环境便利函数"""
        market_configs = {
            "STOCK_AAPL": self.base_config,
            "FOREX_EURUSD": EnvironmentConfiguration(
                market_type=MarketType.FOREX,
                time_granularity=TimeGranularity.DAY_1,
                risk_profile=RiskProfile.BALANCED,
                environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
                initial_balance=100000.0,
                window_size=20,
                max_steps=1000
            )
        }
        
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = create_composite_environment(
            market_configs=market_configs,
            allocation_strategy="volatility_weight",
            rebalance_frequency="monthly",
            correlation_threshold=0.75,
            risk_budget=0.025
        )
        
        self.assertIsInstance(env, MultiMarketCompositeEnvironment)
        self.assertEqual(env.composite_config.allocation_strategy, AllocationStrategy.VOLATILITY_WEIGHT)
        self.assertEqual(env.composite_config.rebalance_frequency, RebalanceFrequency.MONTHLY)
        self.assertEqual(env.composite_config.correlation_threshold, 0.75)
        self.assertEqual(env.composite_config.risk_budget, 0.025)
    
    def test_create_multi_asset_config_function(self):
        """测试创建多资产配置便利函数"""
        symbols = ["AAPL", "EURUSD", "BTC"]
        market_types = [MarketType.STOCK, MarketType.FOREX, MarketType.CRYPTO]
        
        configs = create_multi_asset_config(symbols, market_types, self.base_config)
        
        self.assertEqual(len(configs), 3)
        self.assertIn("stock_AAPL", configs)
        self.assertIn("forex_EURUSD", configs)
        self.assertIn("crypto_BTC", configs)
        
        # 检查配置正确性
        for config in configs.values():
            self.assertIsInstance(config, EnvironmentConfiguration)
            self.assertEqual(config.initial_balance, self.base_config.initial_balance)
            self.assertEqual(config.window_size, self.base_config.window_size)
    
    def test_create_multi_asset_config_mismatch_length(self):
        """测试符号和市场类型长度不匹配"""
        symbols = ["AAPL", "EURUSD"]
        market_types = [MarketType.STOCK]  # 长度不匹配
        
        with self.assertRaises(ValueError):
            create_multi_asset_config(symbols, market_types, self.base_config)
    
    def test_analyze_portfolio_correlation_function(self):
        """测试分析投资组合相关性便利函数"""
        # 创建测试收益率数据
        np.random.seed(42)
        returns_data = {
            "Asset1": np.random.normal(0.001, 0.02, 50),
            "Asset2": np.random.normal(0.001, 0.015, 50),
            "Asset3": np.random.normal(-0.001, 0.025, 50)
        }
        
        # 创建一些相关性
        returns_data["Asset2"] = returns_data["Asset1"] * 0.7 + returns_data["Asset2"] * 0.3
        
        returns_history = pd.DataFrame(returns_data)
        
        analysis = analyze_portfolio_correlation(returns_history, threshold=0.5)
        
        self.assertIn('correlation_matrix', analysis)
        self.assertIn('average_correlation', analysis)
        self.assertIn('diversification_score', analysis)
        self.assertIn('high_correlations', analysis)
        self.assertIn('correlation_warnings', analysis)
        
        # 检查相关性分析结果
        self.assertIsInstance(analysis['average_correlation'], float)
        self.assertTrue(0 <= analysis['diversification_score'] <= 1)
        self.assertIsInstance(analysis['high_correlations'], list)
    
    def test_analyze_portfolio_correlation_insufficient_data(self):
        """测试数据不足的相关性分析"""
        # 空数据
        empty_returns = pd.DataFrame()
        analysis = analyze_portfolio_correlation(empty_returns)
        self.assertIn('error', analysis)
        
        # 数据太少
        small_returns = pd.DataFrame({"Asset1": [0.01, 0.02], "Asset2": [0.015, 0.005]})
        analysis = analyze_portfolio_correlation(small_returns)
        self.assertIn('error', analysis)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        """设置测试"""
        self.base_config = EnvironmentConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
            initial_balance=100000.0,
            window_size=20,
            max_steps=1000
        )
    
    @patch('src.rewards.environments.factory.TradingEnvironmentFactory.create_environment')
    def test_single_market_environment(self, mock_create):
        """测试单市场环境"""
        market_configs = {"STOCK_AAPL": self.base_config}
        
        composite_config = CompositeEnvironmentConfig(
            market_configurations=market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.DAILY
        )
        
        mock_env = MockTradingEnvironment(self.base_config)
        mock_create.return_value = mock_env
        
        env = BaseCompositeEnvironment(composite_config)
        
        self.assertEqual(len(env.sub_environments), 1)
        self.assertEqual(env.target_weights["STOCK_AAPL"], 1.0)
    
    @patch('src.rewards.environments.factory.TradingEnvironmentFactory.create_environment')
    def test_empty_action_handling(self, mock_create):
        """测试空动作处理"""
        market_configs = {
            "STOCK_AAPL": self.base_config,
            "FOREX_EURUSD": EnvironmentConfiguration(
                market_type=MarketType.FOREX,
                time_granularity=TimeGranularity.DAY_1,
                risk_profile=RiskProfile.BALANCED,
                environment_features=EnvironmentFeature.HIGH_LIQUIDITY,
                initial_balance=100000.0,
                window_size=20,
                max_steps=1000
            )
        }
        
        composite_config = CompositeEnvironmentConfig(
            market_configurations=market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.WEEKLY
        )
        
        mock_env1 = MockTradingEnvironment(self.base_config)
        mock_env2 = MockTradingEnvironment(market_configs["FOREX_EURUSD"])
        mock_create.side_effect = [mock_env1, mock_env2]
        
        env = BaseCompositeEnvironment(composite_config)
        env.reset()
        
        # 空动作字典
        obs, reward, done, info = env.step({})
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    @patch('src.rewards.environments.factory.TradingEnvironmentFactory.create_environment')
    def test_zero_portfolio_value_handling(self, mock_create):
        """测试零投资组合价值处理"""
        market_configs = {"STOCK_AAPL": self.base_config}
        
        composite_config = CompositeEnvironmentConfig(
            market_configurations=market_configs,
            allocation_strategy=AllocationStrategy.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.DAILY
        )
        
        mock_env = MockTradingEnvironment(self.base_config)
        mock_create.return_value = mock_env
        
        env = BaseCompositeEnvironment(composite_config)
        env.reset()
        
        # 设置零投资组合价值
        env.portfolio.total_value = 0.0
        
        # 更新投资组合状态不应该崩溃
        sub_results = {"STOCK_AAPL": {"done": False, "info": {}}}
        env._update_portfolio_state(sub_results)
        
        # 权重应该为空或零
        self.assertEqual(env.portfolio.market_weights.get("STOCK_AAPL", 0.0), 0.0)


if __name__ == '__main__':
    unittest.main()