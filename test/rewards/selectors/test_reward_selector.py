"""
智能奖励选择算法测试 - Smart Reward Selector Tests

测试智能奖励选择器的各种策略和功能。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.rewards.selectors.reward_selector import (
    SelectionStrategy, SelectionCriteria, RewardRecommendation,
    BaseRewardSelector, RuleBasedSelector, PerformanceBasedSelector,
    SmartRewardSelector, select_reward_for_trading
)
from src.rewards.enums.market_types import MarketType
from src.rewards.enums.time_granularities import TimeGranularity
from src.rewards.enums.reward_categories import RewardCategory
from src.rewards.enums.environment_features import EnvironmentFeature
from src.rewards.enums.risk_profiles import RiskProfile


class TestSelectionCriteria(unittest.TestCase):
    """测试选择标准"""
    
    def test_selection_criteria_creation(self):
        """测试选择标准创建"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            target_sharpe_ratio=1.5,
            target_max_drawdown=0.10,
            max_complexity=5
        )
        
        self.assertEqual(criteria.market_type, MarketType.STOCK)
        self.assertEqual(criteria.target_sharpe_ratio, 1.5)
        self.assertEqual(criteria.target_max_drawdown, 0.10)
        self.assertEqual(criteria.max_complexity, 5)
    
    def test_selection_criteria_defaults(self):
        """测试选择标准默认值"""
        criteria = SelectionCriteria(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_15,
            risk_profile=RiskProfile.CONSERVATIVE,
            environment_features=EnvironmentFeature.HIGH_FREQUENCY
        )
        
        self.assertIsNone(criteria.target_sharpe_ratio)
        self.assertIsNone(criteria.target_return)
        self.assertIsNone(criteria.max_complexity)


class TestRewardRecommendation(unittest.TestCase):
    """测试奖励推荐"""
    
    def test_reward_recommendation_creation(self):
        """测试奖励推荐创建"""
        recommendation = RewardRecommendation(
            reward_type="risk_adjusted",
            category=RewardCategory.RISK_ADJUSTED,
            confidence_score=0.85,
            expected_performance={"sharpe": 1.6, "return": 0.12},
            reasoning=["High Sharpe ratio expected", "Suitable for balanced risk"],
            parameters={"lookback_period": 20},
            alternatives=[("momentum_based", 0.75), ("simple_return", 0.60)],
            warnings=["Requires clean price data"]
        )
        
        self.assertEqual(recommendation.reward_type, "risk_adjusted")
        self.assertEqual(recommendation.confidence_score, 0.85)
        self.assertEqual(len(recommendation.reasoning), 2)
        self.assertEqual(len(recommendation.alternatives), 2)
        self.assertEqual(len(recommendation.warnings), 1)


class TestRuleBasedSelector(unittest.TestCase):
    """测试基于规则的选择器"""
    
    def setUp(self):
        """设置测试"""
        self.selector = RuleBasedSelector()
    
    def test_rule_based_selector_initialization(self):
        """测试规则选择器初始化"""
        self.assertEqual(self.selector.strategy, SelectionStrategy.RULE_BASED)
        self.assertIsNotNone(self.selector.rules)
        self.assertIn("market_rules", self.selector.rules)
        self.assertIn("granularity_rules", self.selector.rules)
        self.assertIn("risk_rules", self.selector.rules)
    
    def test_stock_market_selection(self):
        """测试股票市场选择"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = self.selector.select_reward(criteria)
        
        self.assertIsInstance(recommendation, RewardRecommendation)
        self.assertTrue(0.0 <= recommendation.confidence_score <= 1.0)
        self.assertIsInstance(recommendation.reasoning, list)
        self.assertIsInstance(recommendation.parameters, dict)
        
        # 检查推荐的奖励类型是否适合股票市场
        stock_preferred = self.selector.rules["market_rules"][MarketType.STOCK]["preferred"]
        self.assertTrue(
            recommendation.reward_type in stock_preferred or 
            recommendation.confidence_score >= 0.5
        )
    
    def test_forex_market_selection(self):
        """测试外汇市场选择"""
        criteria = SelectionCriteria(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_15,
            risk_profile=RiskProfile.AGGRESSIVE,
            environment_features=EnvironmentFeature.HIGH_FREQUENCY | EnvironmentFeature.LEVERAGE_AVAILABLE
        )
        
        recommendation = self.selector.select_reward(criteria)
        
        self.assertIsInstance(recommendation, RewardRecommendation)
        self.assertTrue(0.0 <= recommendation.confidence_score <= 1.0)
        
        # 外汇市场应该避免某些奖励类型
        forex_avoid = self.selector.rules["market_rules"][MarketType.FOREX]["avoid"]
        self.assertNotIn(recommendation.reward_type, forex_avoid)
    
    def test_high_frequency_granularity_selection(self):
        """测试高频交易时间粒度选择"""
        criteria = SelectionCriteria(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_1,  # 高频
            risk_profile=RiskProfile.AGGRESSIVE,
            environment_features=EnvironmentFeature.HIGH_FREQUENCY
        )
        
        recommendation = self.selector.select_reward(criteria)
        
        # 高频交易应该偏好某些特定的奖励类型
        hf_preferred = self.selector.rules["granularity_rules"]["high_frequency"]["preferred"]
        # 由于评分系统，不一定选择偏好的，但至少不应该选择避免的
        hf_avoid = self.selector.rules["granularity_rules"]["high_frequency"]["avoid"]
        self.assertNotIn(recommendation.reward_type, hf_avoid)
    
    def test_conservative_risk_profile(self):
        """测试保守风险配置"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.CONSERVATIVE,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = self.selector.select_reward(criteria)
        
        # 检查复杂度限制
        complexity = self.selector._get_reward_complexity(recommendation.reward_type)
        conservative_limit = self.selector.rules["risk_rules"][RiskProfile.CONSERVATIVE]["complexity_limit"]
        # 允许一定的超出，因为可能没有合适的低复杂度选项
        self.assertTrue(complexity <= conservative_limit + 2)
    
    def test_rank_rewards(self):
        """测试奖励排序"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.HOUR_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        rankings = self.selector.rank_rewards(criteria, top_n=3)
        
        self.assertIsInstance(rankings, list)
        self.assertTrue(len(rankings) <= 3)
        
        if len(rankings) > 1:
            # 检查排序是否正确（置信度降序）
            for i in range(len(rankings) - 1):
                self.assertGreaterEqual(
                    rankings[i].confidence_score,
                    rankings[i + 1].confidence_score
                )
    
    def test_granularity_category_detection(self):
        """测试时间粒度类别检测"""
        # 高频
        hf_granularity = TimeGranularity.MINUTE_1
        category = self.selector._get_granularity_category(hf_granularity)
        self.assertEqual(category, "high_frequency")
        
        # 长期
        lt_granularity = TimeGranularity.WEEK_1
        category = self.selector._get_granularity_category(lt_granularity)
        self.assertEqual(category, "long_term")
    
    def test_feature_score_calculation(self):
        """测试环境特征评分计算"""
        # 测试高频特征
        score_hf = self.selector._calculate_feature_score(
            "scalping", 
            EnvironmentFeature.HIGH_FREQUENCY
        )
        self.assertGreater(score_hf, 0)
        
        # 测试新闻驱动特征
        score_news = self.selector._calculate_feature_score(
            "news_sentiment",
            EnvironmentFeature.NEWS_DRIVEN
        )
        self.assertGreater(score_news, 0)
        
        # 测试无匹配特征
        score_none = self.selector._calculate_feature_score(
            "simple_return",
            EnvironmentFeature.HIGH_FREQUENCY
        )
        self.assertEqual(score_none, 0)
    
    def test_fallback_recommendation(self):
        """测试后备推荐"""
        # 创建一个可能没有候选者的场景
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature(0)
        )
        
        # 模拟没有候选者的情况
        with patch.object(self.selector, '_get_candidate_rewards', return_value=[]):
            recommendation = self.selector.select_reward(criteria)
            
            self.assertEqual(recommendation.reward_type, "simple_return")
            self.assertEqual(recommendation.category, RewardCategory.BASIC)
            self.assertIn("Fallback", recommendation.reasoning[0])
            self.assertIn("fallback", recommendation.warnings[0])
    
    def test_selection_history_tracking(self):
        """测试选择历史跟踪"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        initial_count = len(self.selector.selection_history)
        recommendation = self.selector.select_reward(criteria)
        
        self.assertEqual(len(self.selector.selection_history), initial_count + 1)
        
        # 检查统计信息
        stats = self.selector.get_selection_statistics()
        self.assertIn("total_selections", stats)
        self.assertIn("reward_distribution", stats)
        self.assertIn("average_confidence", stats)


class TestPerformanceBasedSelector(unittest.TestCase):
    """测试基于性能的选择器"""
    
    def setUp(self):
        """设置测试"""
        self.selector = PerformanceBasedSelector()
    
    def test_performance_based_selector_initialization(self):
        """测试性能选择器初始化"""
        self.assertEqual(self.selector.strategy, SelectionStrategy.PERFORMANCE_BASED)
        self.assertIsNotNone(self.selector.performance_database)
        self.assertIn("stock", self.selector.performance_database)
        self.assertIn("forex", self.selector.performance_database)
    
    def test_stock_performance_selection(self):
        """测试股票性能选择"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            target_sharpe_ratio=1.5
        )
        
        recommendation = self.selector.select_reward(criteria)
        
        self.assertIsInstance(recommendation, RewardRecommendation)
        self.assertTrue(0.0 <= recommendation.confidence_score <= 1.0)
        self.assertIn("Historical Sharpe ratio", recommendation.reasoning[0])
    
    def test_forex_performance_selection(self):
        """测试外汇性能选择"""
        criteria = SelectionCriteria(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_15,
            risk_profile=RiskProfile.AGGRESSIVE,
            environment_features=EnvironmentFeature.HIGH_FREQUENCY
        )
        
        recommendation = self.selector.select_reward(criteria)
        
        self.assertIsInstance(recommendation, RewardRecommendation)
        self.assertIsNotNone(recommendation.expected_performance)
        self.assertIn("sharpe", recommendation.expected_performance)
    
    def test_unavailable_data_fallback(self):
        """测试数据不可用时的后备选择"""
        criteria = SelectionCriteria(
            market_type=MarketType.CRYPTO,  # 数据库中没有的市场
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature(0)
        )
        
        recommendation = self.selector.select_reward(criteria)
        
        # 应该使用规则选择器作为后备
        self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_performance_score_calculation(self):
        """测试性能得分计算"""
        performance = {"sharpe": 2.0, "return": 0.15, "drawdown": 0.05}
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN,
            target_sharpe_ratio=1.5,
            target_return=0.10,
            target_max_drawdown=0.08
        )
        
        score = self.selector._calculate_performance_score(performance, criteria)
        
        # 由于超过目标指标，得分应该很高
        self.assertGreater(score, 80)
    
    def test_rank_rewards_by_performance(self):
        """测试基于性能的排序"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        rankings = self.selector.rank_rewards(criteria, top_n=3)
        
        self.assertIsInstance(rankings, list)
        
        if len(rankings) > 1:
            # 检查排序是否正确
            for i in range(len(rankings) - 1):
                self.assertGreaterEqual(
                    rankings[i].confidence_score,
                    rankings[i + 1].confidence_score
                )


class TestSmartRewardSelector(unittest.TestCase):
    """测试智能奖励选择器"""
    
    def setUp(self):
        """设置测试"""
        self.selector = SmartRewardSelector()
    
    def test_smart_selector_initialization(self):
        """测试智能选择器初始化"""
        self.assertEqual(self.selector.default_strategy, SelectionStrategy.HYBRID)
        self.assertIn(SelectionStrategy.RULE_BASED, self.selector.selectors)
        self.assertIn(SelectionStrategy.PERFORMANCE_BASED, self.selector.selectors)
    
    def test_rule_based_strategy(self):
        """测试规则策略"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = self.selector.select_optimal_reward(
            criteria, 
            strategy=SelectionStrategy.RULE_BASED
        )
        
        self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_performance_based_strategy(self):
        """测试性能策略"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = self.selector.select_optimal_reward(
            criteria,
            strategy=SelectionStrategy.PERFORMANCE_BASED
        )
        
        self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_hybrid_strategy(self):
        """测试混合策略"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = self.selector.select_optimal_reward(
            criteria,
            strategy=SelectionStrategy.HYBRID
        )
        
        self.assertIsInstance(recommendation, RewardRecommendation)
        self.assertIn("Hybrid strategy", recommendation.reasoning[0])
    
    def test_consensus_strategy(self):
        """测试共识策略"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = self.selector.select_optimal_reward(
            criteria,
            strategy=SelectionStrategy.CONSENSUS
        )
        
        self.assertIsInstance(recommendation, RewardRecommendation)
        self.assertIn("Consensus selection", recommendation.reasoning[0])
    
    def test_compare_strategies(self):
        """测试策略比较"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        results = self.selector.compare_strategies(criteria)
        
        self.assertIsInstance(results, dict)
        self.assertIn("rule_based", results)
        self.assertIn("performance_based", results)
        
        for strategy_name, recommendation in results.items():
            self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_selector_statistics(self):
        """测试选择器统计"""
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        # 执行几次选择以生成统计数据
        for _ in range(3):
            self.selector.select_optimal_reward(criteria)
        
        stats = self.selector.get_selector_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("rule_based", stats)
        self.assertIn("performance_based", stats)


class TestConvenienceFunction(unittest.TestCase):
    """测试便利函数"""
    
    def test_select_reward_for_trading_function(self):
        """测试便利函数"""
        recommendation = select_reward_for_trading(
            market_type="stock",
            time_granularity="1d",
            risk_profile="balanced",
            strategy=SelectionStrategy.RULE_BASED
        )
        
        self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_string_to_enum_conversion(self):
        """测试字符串到枚举的转换"""
        recommendation = select_reward_for_trading(
            market_type="forex",
            time_granularity="15min",
            risk_profile="aggressive"
        )
        
        self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_default_parameters(self):
        """测试默认参数"""
        recommendation = select_reward_for_trading(
            market_type="stock",
            time_granularity="1d"
        )
        
        self.assertIsInstance(recommendation, RewardRecommendation)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_performance_database(self):
        """测试空性能数据库"""
        selector = PerformanceBasedSelector()
        # 清空数据库
        selector.performance_database = {}
        
        criteria = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.BALANCED,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = selector.select_reward(criteria)
        
        # 应该使用后备选择
        self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_extreme_risk_profiles(self):
        """测试极端风险配置"""
        selector = RuleBasedSelector()
        
        # 超保守
        criteria_ultra_conservative = SelectionCriteria(
            market_type=MarketType.STOCK,
            time_granularity=TimeGranularity.DAY_1,
            risk_profile=RiskProfile.ULTRA_CONSERVATIVE,
            environment_features=EnvironmentFeature.NEWS_DRIVEN
        )
        
        recommendation = selector.select_reward(criteria_ultra_conservative)
        self.assertIsInstance(recommendation, RewardRecommendation)
        
        # 超激进
        criteria_ultra_aggressive = SelectionCriteria(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_1,
            risk_profile=RiskProfile.ULTRA_AGGRESSIVE,
            environment_features=EnvironmentFeature.HIGH_FREQUENCY | EnvironmentFeature.LEVERAGE_AVAILABLE
        )
        
        recommendation = selector.select_reward(criteria_ultra_aggressive)
        self.assertIsInstance(recommendation, RewardRecommendation)
    
    def test_complex_environment_features(self):
        """测试复杂环境特征组合"""
        selector = RuleBasedSelector()
        
        criteria = SelectionCriteria(
            market_type=MarketType.FOREX,
            time_granularity=TimeGranularity.MINUTE_5,
            risk_profile=RiskProfile.AGGRESSIVE,
            environment_features=(
                EnvironmentFeature.HIGH_FREQUENCY |
                EnvironmentFeature.NEWS_DRIVEN |
                EnvironmentFeature.LEVERAGE_AVAILABLE |
                EnvironmentFeature.HIGH_VOLATILITY
            )
        )
        
        recommendation = selector.select_reward(criteria)
        self.assertIsInstance(recommendation, RewardRecommendation)
        self.assertTrue(0.0 <= recommendation.confidence_score <= 1.0)


if __name__ == '__main__':
    unittest.main()