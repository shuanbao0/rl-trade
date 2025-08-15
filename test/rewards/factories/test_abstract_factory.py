"""
抽象工厂测试 - Abstract Factory Tests

测试抽象工厂基类和接口定义的正确性。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.rewards.factories.abstract_factory import (
    FactoryConfiguration, CreationResult, AbstractRewardFactory,
    AbstractEnvironmentFactory, AbstractComponentFactory, 
    MasterAbstractFactory, FactoryRegistry, 
    create_factory_configuration, get_factory
)
from src.rewards.enums.market_types import MarketType
from src.rewards.enums.time_granularities import TimeGranularity
from src.rewards.enums.risk_profiles import RiskProfile
from src.rewards.enums.environment_features import EnvironmentFeature
from src.rewards.enums.reward_categories import RewardCategory


class TestFactoryConfiguration(unittest.TestCase):
    """测试工厂配置"""
    
    def test_factory_configuration_creation(self):
        """测试工厂配置创建"""
        config = FactoryConfiguration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            preferred_categories=[RewardCategory.BASIC],
            custom_parameters={"test": "value"}
        )
        
        self.assertEqual(config.market_type, MarketType.STOCK)
        self.assertEqual(config.time_granularity, TimeGranularity.DAY_1)
        self.assertEqual(config.risk_profile, RiskProfile.BALANCED)
        self.assertEqual(config.performance_mode, "balanced")
        self.assertTrue(config.logging_enabled)
        self.assertTrue(config.cache_enabled)
    
    def test_creation_result(self):
        """测试创建结果"""
        result = CreationResult(
            instance="test_instance",
            metadata={"type": "test"},
            warnings=["warning1"],
            configuration_used={"param": "value"},
            performance_metrics={"score": 8.5}
        )
        
        self.assertEqual(result.instance, "test_instance")
        self.assertEqual(result.metadata["type"], "test")
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(result.performance_metrics["score"], 8.5)


class MockRewardFactory(AbstractRewardFactory):
    """测试用的模拟奖励工厂"""
    
    def create_reward(self, reward_type: str, **kwargs) -> CreationResult:
        return CreationResult(
            instance=f"Mock_{reward_type}",
            metadata={"type": reward_type},
            warnings=[],
            configuration_used=kwargs,
            performance_metrics={"score": 7.0}
        )
    
    def get_available_rewards(self) -> list:
        return ["mock_reward_1", "mock_reward_2"]
    
    def get_recommended_rewards(self) -> list:
        return [("mock_reward_1", 8.0), ("mock_reward_2", 6.0)]
    
    def validate_configuration(self) -> tuple:
        return True, []


class MockEnvironmentFactory(AbstractEnvironmentFactory):
    """测试用的模拟环境工厂"""
    
    def create_environment(self, env_type: str = "trading", **kwargs) -> CreationResult:
        return CreationResult(
            instance=f"Mock_Environment_{env_type}",
            metadata={"env_type": env_type},
            warnings=[],
            configuration_used=kwargs,
            performance_metrics={"init_time": 0.1}
        )
    
    def create_observation_space(self, **kwargs):
        return "MockObservationSpace"
    
    def create_action_space(self, **kwargs):
        return "MockActionSpace"
    
    def get_environment_configuration(self) -> dict:
        return {"mock": True}


class MockComponentFactory(AbstractComponentFactory):
    """测试用的模拟组件工厂"""
    
    def create_feature_engineer(self, **kwargs) -> CreationResult:
        return CreationResult(
            instance="MockFeatureEngineer",
            metadata={"component": "feature_engineer"},
            warnings=[],
            configuration_used=kwargs,
            performance_metrics={"speed": 1000}
        )
    
    def create_risk_manager(self, **kwargs) -> CreationResult:
        return CreationResult(
            instance="MockRiskManager",
            metadata={"component": "risk_manager"},
            warnings=[],
            configuration_used=kwargs,
            performance_metrics={"latency": 0.001}
        )
    
    def create_data_processor(self, **kwargs) -> CreationResult:
        return CreationResult(
            instance="MockDataProcessor",
            metadata={"component": "data_processor"},
            warnings=[],
            configuration_used=kwargs,
            performance_metrics={"throughput": 5000}
        )


class MockMasterFactory(MasterAbstractFactory):
    """测试用的模拟主工厂"""
    
    def initialize(self) -> None:
        if self._initialized:
            return
        
        self._reward_factory = MockRewardFactory(self.config)
        self._environment_factory = MockEnvironmentFactory(self.config)
        self._component_factory = MockComponentFactory(self.config)
        self._initialized = True
    
    def create_complete_system(self, **kwargs) -> dict:
        self.initialize()
        
        return {
            "reward": self._reward_factory.create_reward("mock_reward_1"),
            "environment": self._environment_factory.create_environment(),
            "feature_engineer": self._component_factory.create_feature_engineer(),
            "risk_manager": self._component_factory.create_risk_manager(),
            "data_processor": self._component_factory.create_data_processor()
        }
    
    def get_system_compatibility_score(self) -> float:
        return 8.5


class TestAbstractFactories(unittest.TestCase):
    """测试抽象工厂基类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = create_factory_configuration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED
        )
    
    def test_reward_factory(self):
        """测试奖励工厂"""
        factory = MockRewardFactory(self.config)
        
        # 测试创建奖励
        result = factory.create_reward("test_reward")
        self.assertEqual(result.instance, "Mock_test_reward")
        self.assertEqual(result.metadata["type"], "test_reward")
        
        # 测试获取可用奖励
        rewards = factory.get_available_rewards()
        self.assertIn("mock_reward_1", rewards)
        
        # 测试获取推荐奖励
        recommendations = factory.get_recommended_rewards()
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0][0], "mock_reward_1")
        
        # 测试验证配置
        valid, errors = factory.validate_configuration()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
    
    def test_environment_factory(self):
        """测试环境工厂"""
        factory = MockEnvironmentFactory(self.config)
        
        # 测试创建环境
        result = factory.create_environment("test_env")
        self.assertEqual(result.instance, "Mock_Environment_test_env")
        
        # 测试创建空间
        obs_space = factory.create_observation_space()
        self.assertEqual(obs_space, "MockObservationSpace")
        
        action_space = factory.create_action_space()
        self.assertEqual(action_space, "MockActionSpace")
        
        # 测试获取配置
        config = factory.get_environment_configuration()
        self.assertTrue(config["mock"])
    
    def test_component_factory(self):
        """测试组件工厂"""
        factory = MockComponentFactory(self.config)
        
        # 测试创建特征工程器
        result = factory.create_feature_engineer()
        self.assertEqual(result.instance, "MockFeatureEngineer")
        
        # 测试创建风险管理器
        result = factory.create_risk_manager()
        self.assertEqual(result.instance, "MockRiskManager")
        
        # 测试创建数据处理器
        result = factory.create_data_processor()
        self.assertEqual(result.instance, "MockDataProcessor")
    
    def test_master_factory(self):
        """测试主工厂"""
        factory = MockMasterFactory(self.config)
        
        # 测试初始化
        self.assertFalse(factory._initialized)
        factory.initialize()
        self.assertTrue(factory._initialized)
        
        # 测试获取子工厂
        reward_factory = factory.reward_factory
        self.assertIsInstance(reward_factory, MockRewardFactory)
        
        # 测试创建完整系统
        system = factory.create_complete_system()
        self.assertEqual(len(system), 5)
        self.assertIn("reward", system)
        self.assertIn("environment", system)
        
        # 测试兼容性评分
        score = factory.get_system_compatibility_score()
        self.assertEqual(score, 8.5)
        
        # 测试配置验证
        valid, errors = factory.validate_system_configuration()
        self.assertTrue(valid)


class TestFactoryRegistry(unittest.TestCase):
    """测试工厂注册表"""
    
    def setUp(self):
        """设置测试环境"""
        self.registry = FactoryRegistry()
        self.config = create_factory_configuration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1
        )
    
    def test_register_and_create_factory(self):
        """测试工厂注册和创建"""
        # 注册工厂
        self.registry.register_factory("test_factory", MockMasterFactory, self.config)
        
        # 验证注册
        factories = self.registry.list_available_factories()
        self.assertIn("test_factory", factories)
        
        # 创建工厂实例
        factory = self.registry.create_factory("test_factory")
        self.assertIsInstance(factory, MockMasterFactory)
    
    def test_get_factory_for_market(self):
        """测试根据市场获取工厂"""
        # 注册多个工厂
        self.registry.register_factory("stock_1d", MockMasterFactory)
        self.registry.register_factory("stock_*", MockMasterFactory)
        self.registry.register_factory("default", MockMasterFactory)
        
        # 测试精确匹配
        key = self.registry.get_factory_for_market(MarketType.STOCK, TimeGranularity.DAY_1)
        self.assertEqual(key, "stock_1d")
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        self.registry.register_factory("test_factory", MockMasterFactory, self.config)
        
        info = self.registry.get_factory_info("test_factory")
        self.assertEqual(info["key"], "test_factory")
        self.assertEqual(info["class"], "MockMasterFactory")
        self.assertTrue(info["has_default_config"])


class TestFactoryConfigurationUtils(unittest.TestCase):
    """测试工厂配置工具函数"""
    
    def test_create_factory_configuration_from_strings(self):
        """测试从字符串创建工厂配置"""
        config = create_factory_configuration(
            market_type="stock",
            time_granularity="1d",
            risk_profile="balanced"
        )
        
        self.assertEqual(config.market_type, MarketType.STOCK)
        self.assertEqual(config.time_granularity, TimeGranularity.DAY_1)
        self.assertEqual(config.risk_profile, RiskProfile.BALANCED)
    
    def test_create_factory_configuration_with_features(self):
        """测试创建带环境特征的工厂配置"""
        # 测试高频外汇配置
        config_hf = create_factory_configuration(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_1,  # 1分钟是高频
            risk_profile=RiskProfile.AGGRESSIVE
        )
        
        # 验证高频特征设置
        self.assertTrue(config_hf.environment_features & EnvironmentFeature.HIGH_FREQUENCY)
        self.assertTrue(config_hf.environment_features & EnvironmentFeature.REAL_TIME_DATA)
        self.assertTrue(config_hf.environment_features & EnvironmentFeature.HIGH_LIQUIDITY)
        
        # 测试非高频外汇配置
        config_lf = create_factory_configuration(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_5,  # 5分钟不是高频
            risk_profile=RiskProfile.AGGRESSIVE
        )
        
        # 验证外汇特征但无高频特征
        self.assertFalse(config_lf.environment_features & EnvironmentFeature.HIGH_FREQUENCY)
        self.assertTrue(config_lf.environment_features & EnvironmentFeature.HIGH_LIQUIDITY)
    
    def test_create_factory_configuration_preferred_categories(self):
        """测试创建带偏好类别的工厂配置"""
        config = create_factory_configuration(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.CONSERVATIVE
        )
        
        # 验证保守型风险配置的默认偏好
        self.assertIn(RewardCategory.RISK_ADJUSTED, config.preferred_categories)
        self.assertIn(RewardCategory.BASIC, config.preferred_categories)


if __name__ == '__main__':
    unittest.main()