"""
智能奖励选择算法 - Smart Reward Selection Algorithm

基于市场特征、交易策略和性能目标自动选择和推荐最优奖励函数。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from enum import Enum
import logging

from ..enums.market_types import MarketType
from ..enums.time_granularities import TimeGranularity
from ..enums.reward_categories import RewardCategory
# 从环境模块导入 EnvironmentFeature
from ...environment.enums.environment_features import EnvironmentFeature
from ..enums.risk_profiles import RiskProfile
from ..core.base_reward import BaseReward


class SelectionStrategy(Enum):
    """选择策略"""
    PERFORMANCE_BASED = "performance_based"  # 基于历史性能
    RULE_BASED = "rule_based"  # 基于规则
    ML_BASED = "ml_based"  # 基于机器学习
    HYBRID = "hybrid"  # 混合策略
    CONSENSUS = "consensus"  # 共识算法


@dataclass
class SelectionCriteria:
    """选择标准"""
    market_type: MarketType
    time_granularity: TimeGranularity
    risk_profile: RiskProfile
    environment_features: EnvironmentFeature
    
    # 性能目标
    target_sharpe_ratio: Optional[float] = None
    target_max_drawdown: Optional[float] = None
    target_volatility: Optional[float] = None
    target_return: Optional[float] = None
    
    # 约束条件
    max_complexity: Optional[int] = None
    computational_budget: Optional[float] = None
    interpretability_requirement: Optional[str] = None
    
    # 历史数据
    historical_data: Optional[pd.DataFrame] = None
    backtest_period: Optional[int] = None


@dataclass
class RewardRecommendation:
    """奖励推荐结果"""
    reward_type: str
    category: RewardCategory
    confidence_score: float
    expected_performance: Dict[str, float]
    reasoning: List[str]
    parameters: Dict[str, Any]
    alternatives: List[Tuple[str, float]]  # (reward_type, score)
    warnings: List[str]


class BaseRewardSelector(ABC):
    """奖励选择器基类"""
    
    def __init__(self, strategy: SelectionStrategy):
        self.strategy = strategy
        # 使用统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # 历史选择记录
        self.selection_history: List[Dict[str, Any]] = []
        
        # 性能缓存
        self.performance_cache: Dict[str, Dict[str, float]] = {}
    
    @abstractmethod
    def select_reward(self, criteria: SelectionCriteria) -> RewardRecommendation:
        """选择最优奖励函数"""
        pass
    
    @abstractmethod
    def rank_rewards(self, criteria: SelectionCriteria, top_n: int = 5) -> List[RewardRecommendation]:
        """排序奖励函数"""
        pass
    
    def record_selection(self, criteria: SelectionCriteria, recommendation: RewardRecommendation) -> None:
        """记录选择历史"""
        record = {
            "timestamp": pd.Timestamp.now(),
            "criteria": criteria,
            "recommendation": recommendation,
            "strategy": self.strategy.value
        }
        self.selection_history.append(record)
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        if not self.selection_history:
            return {"total_selections": 0}
        
        # 统计选择的奖励类型分布
        reward_counts = {}
        confidence_scores = []
        
        for record in self.selection_history:
            reward_type = record["recommendation"].reward_type
            reward_counts[reward_type] = reward_counts.get(reward_type, 0) + 1
            confidence_scores.append(record["recommendation"].confidence_score)
        
        return {
            "total_selections": len(self.selection_history),
            "reward_distribution": reward_counts,
            "average_confidence": np.mean(confidence_scores),
            "confidence_std": np.std(confidence_scores),
            "most_popular_reward": max(reward_counts, key=reward_counts.get)
        }


class RuleBasedSelector(BaseRewardSelector):
    """基于规则的奖励选择器"""
    
    def __init__(self):
        super().__init__(SelectionStrategy.RULE_BASED)
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, Dict[str, Any]]:
        """初始化选择规则"""
        return {
            # 市场类型规则
            "market_rules": {
                MarketType.STOCK: {
                    "preferred": ["risk_adjusted", "dividend_adjusted", "sector_relative"],
                    "avoid": ["pip_based", "carry_trade"],
                    "weight": 0.3
                },
                MarketType.FOREX: {
                    "preferred": ["forex_optimized", "pip_based", "carry_trade"],
                    "avoid": ["dividend_adjusted", "earnings_momentum"],
                    "weight": 0.3
                },
                MarketType.CRYPTO: {
                    "preferred": ["volatility_adjusted", "momentum_based", "fear_greed_index"],
                    "avoid": ["dividend_adjusted", "carry_trade"],
                    "weight": 0.3
                }
            },
            
            # 时间粒度规则
            "granularity_rules": {
                "high_frequency": {  # < 5分钟
                    "preferred": ["scalping", "spread_aware", "momentum_breakout"],
                    "avoid": ["carry_trade", "fundamental_analysis"],
                    "weight": 0.25
                },
                "short_term": {  # 5分钟 - 1小时
                    "preferred": ["technical_pattern", "mean_reversion", "trend_following"],
                    "avoid": ["long_term_value", "dividend_focused"],
                    "weight": 0.25
                },
                "medium_term": {  # 1小时 - 1天
                    "preferred": ["risk_adjusted", "balanced_return", "indicator_based"],
                    "avoid": ["scalping", "ultra_short_momentum"],
                    "weight": 0.25
                },
                "long_term": {  # > 1天
                    "preferred": ["fundamental_analysis", "value_investing", "dividend_growth"],
                    "avoid": ["scalping", "high_frequency"],
                    "weight": 0.25
                }
            },
            
            # 风险配置规则
            "risk_rules": {
                RiskProfile.ULTRA_CONSERVATIVE: {
                    "preferred": ["capital_preservation", "minimum_variance", "bond_like"],
                    "complexity_limit": 3,
                    "weight": 0.2
                },
                RiskProfile.CONSERVATIVE: {
                    "preferred": ["risk_adjusted", "downside_protection", "low_volatility"],
                    "complexity_limit": 4,
                    "weight": 0.2
                },
                RiskProfile.MODERATE: {
                    "preferred": ["balanced_return", "moderate_risk", "steady_growth"],
                    "complexity_limit": 6,
                    "weight": 0.2
                },
                RiskProfile.AGGRESSIVE: {
                    "preferred": ["high_return", "momentum_strong", "leverage_enhanced"],
                    "complexity_limit": 8,
                    "weight": 0.2
                },
                RiskProfile.ULTRA_AGGRESSIVE: {
                    "preferred": ["maximum_return", "ultra_momentum", "speculative"],
                    "complexity_limit": 10,
                    "weight": 0.2
                }
            }
        }
    
    def select_reward(self, criteria: SelectionCriteria) -> RewardRecommendation:
        """基于规则选择奖励函数"""
        
        # 获取候选奖励函数
        candidates = self._get_candidate_rewards(criteria)
        
        # 计算每个候选的得分
        scored_candidates = []
        for reward_type in candidates:
            score = self._calculate_rule_score(reward_type, criteria)
            scored_candidates.append((reward_type, score))
        
        # 排序并选择最高分
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_candidates:
            return self._create_fallback_recommendation(criteria)
        
        best_reward, best_score = scored_candidates[0]
        
        # 构建推荐结果
        recommendation = RewardRecommendation(
            reward_type=best_reward,
            category=self._get_reward_category(best_reward),
            confidence_score=min(best_score / 100.0, 1.0),  # 标准化到0-1
            expected_performance=self._estimate_performance(best_reward, criteria),
            reasoning=self._generate_reasoning(best_reward, criteria),
            parameters=self._suggest_parameters(best_reward, criteria),
            alternatives=scored_candidates[1:6],  # 前5个备选
            warnings=self._generate_warnings(best_reward, criteria)
        )
        
        # 记录选择
        self.record_selection(criteria, recommendation)
        
        return recommendation
    
    def rank_rewards(self, criteria: SelectionCriteria, top_n: int = 5) -> List[RewardRecommendation]:
        """排序奖励函数"""
        candidates = self._get_candidate_rewards(criteria)
        
        recommendations = []
        for reward_type in candidates:
            score = self._calculate_rule_score(reward_type, criteria)
            
            recommendation = RewardRecommendation(
                reward_type=reward_type,
                category=self._get_reward_category(reward_type),
                confidence_score=min(score / 100.0, 1.0),
                expected_performance=self._estimate_performance(reward_type, criteria),
                reasoning=self._generate_reasoning(reward_type, criteria),
                parameters=self._suggest_parameters(reward_type, criteria),
                alternatives=[],
                warnings=self._generate_warnings(reward_type, criteria)
            )
            recommendations.append(recommendation)
        
        # 按置信度排序
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations[:top_n]
    
    def _get_candidate_rewards(self, criteria: SelectionCriteria) -> List[str]:
        """获取候选奖励函数"""
        # 基础候选集合
        base_candidates = [
            "simple_return", "risk_adjusted", "profit_loss", "momentum_based",
            "trend_following", "mean_reversion", "volatility_adjusted"
        ]
        
        # 根据市场类型添加特定候选
        if criteria.market_type == MarketType.STOCK:
            base_candidates.extend([
                "dividend_adjusted", "earnings_momentum", "sector_relative",
                "technical_pattern", "volume_price"
            ])
        elif criteria.market_type == MarketType.FOREX:
            base_candidates.extend([
                "forex_optimized", "pip_based", "spread_aware", "carry_trade",
                "correlation_pair", "news_sentiment"
            ])
        elif criteria.market_type == MarketType.CRYPTO:
            base_candidates.extend([
                "volatility_adjusted", "momentum_based", "sentiment_driven"
            ])
        
        return list(set(base_candidates))  # 去重
    
    def _calculate_rule_score(self, reward_type: str, criteria: SelectionCriteria) -> float:
        """计算规则得分"""
        total_score = 0.0
        
        # 市场类型得分
        market_rules = self.rules["market_rules"].get(criteria.market_type, {})
        if reward_type in market_rules.get("preferred", []):
            total_score += 30 * market_rules.get("weight", 0.3)
        elif reward_type in market_rules.get("avoid", []):
            total_score -= 20 * market_rules.get("weight", 0.3)
        
        # 时间粒度得分
        granularity_category = self._get_granularity_category(criteria.time_granularity)
        granularity_rules = self.rules["granularity_rules"].get(granularity_category, {})
        if reward_type in granularity_rules.get("preferred", []):
            total_score += 25 * granularity_rules.get("weight", 0.25)
        elif reward_type in granularity_rules.get("avoid", []):
            total_score -= 15 * granularity_rules.get("weight", 0.25)
        
        # 风险配置得分
        risk_rules = self.rules["risk_rules"].get(criteria.risk_profile, {})
        if reward_type in risk_rules.get("preferred", []):
            total_score += 20 * risk_rules.get("weight", 0.2)
        
        # 复杂度检查
        reward_complexity = self._get_reward_complexity(reward_type)
        complexity_limit = risk_rules.get("complexity_limit", 10)
        if reward_complexity > complexity_limit:
            total_score -= (reward_complexity - complexity_limit) * 5
        
        # 环境特征得分
        feature_score = self._calculate_feature_score(reward_type, criteria.environment_features)
        total_score += feature_score
        
        return max(0.0, total_score)
    
    def _get_granularity_category(self, granularity: TimeGranularity) -> str:
        """获取时间粒度类别"""
        if granularity.is_high_frequency():
            return "high_frequency"
        elif granularity.category in ["short_term"]:
            return "short_term"
        elif granularity.category in ["medium_term"]:
            return "medium_term"
        else:
            return "long_term"
    
    def _calculate_feature_score(self, reward_type: str, features: EnvironmentFeature) -> float:
        """计算环境特征得分"""
        score = 0.0
        
        # 高频交易特征
        if features & EnvironmentFeature.HIGH_FREQUENCY:
            if reward_type in ["scalping", "spread_aware", "momentum_breakout"]:
                score += 10
        
        # 新闻驱动特征
        if features & EnvironmentFeature.NEWS_DRIVEN:
            if reward_type in ["news_sentiment", "earnings_momentum", "event_driven"]:
                score += 8
        
        # 杠杆特征
        if features & EnvironmentFeature.LEVERAGE_AVAILABLE:
            if reward_type in ["leverage_enhanced", "margin_optimized", "carry_trade"]:
                score += 6
        
        # 高波动性特征
        if features & EnvironmentFeature.HIGH_VOLATILITY:
            if reward_type in ["volatility_adjusted", "risk_parity", "adaptive_sizing"]:
                score += 8
        
        return score
    
    def _get_reward_complexity(self, reward_type: str) -> int:
        """获取奖励函数复杂度"""
        complexity_map = {
            "simple_return": 1,
            "profit_loss": 1,
            "risk_adjusted": 3,
            "momentum_based": 4,
            "trend_following": 4,
            "mean_reversion": 4,
            "volatility_adjusted": 6,
            "forex_optimized": 5,
            "carry_trade": 6,
            "news_sentiment": 8,
            "correlation_pair": 7,
            "earnings_momentum": 6
        }
        return complexity_map.get(reward_type, 5)  # 默认复杂度
    
    def _get_reward_category(self, reward_type: str) -> RewardCategory:
        """获取奖励函数类别"""
        category_map = {
            "simple_return": RewardCategory.BASIC,
            "risk_adjusted": RewardCategory.RISK_ADJUSTED,
            "momentum_based": RewardCategory.MOMENTUM,
            "trend_following": RewardCategory.TREND_FOLLOWING,
            "mean_reversion": RewardCategory.MEAN_REVERSION,
            "volatility_adjusted": RewardCategory.VOLATILITY_AWARE,
            "forex_optimized": RewardCategory.MARKET_SPECIFIC,
            "news_sentiment": RewardCategory.ML_ENHANCED
        }
        return category_map.get(reward_type, RewardCategory.BASIC)
    
    def _estimate_performance(self, reward_type: str, criteria: SelectionCriteria) -> Dict[str, float]:
        """估算性能指标"""
        # 基于历史经验的性能估算
        base_performance = {
            "expected_sharpe": 1.0,
            "expected_return": 0.08,
            "expected_volatility": 0.15,
            "expected_max_drawdown": 0.10
        }
        
        # 根据奖励类型调整
        if reward_type == "risk_adjusted":
            base_performance["expected_sharpe"] = 1.5
            base_performance["expected_max_drawdown"] = 0.08
        elif reward_type == "momentum_based":
            base_performance["expected_return"] = 0.12
            base_performance["expected_volatility"] = 0.20
        elif reward_type == "volatility_adjusted":
            base_performance["expected_volatility"] = 0.12
            base_performance["expected_max_drawdown"] = 0.06
        
        return base_performance
    
    def _generate_reasoning(self, reward_type: str, criteria: SelectionCriteria) -> List[str]:
        """生成选择理由"""
        reasoning = []
        
        # 市场匹配理由
        market_rules = self.rules["market_rules"].get(criteria.market_type, {})
        if reward_type in market_rules.get("preferred", []):
            reasoning.append(f"Optimized for {criteria.market_type.value} market characteristics")
        
        # 时间粒度匹配理由
        granularity_category = self._get_granularity_category(criteria.time_granularity)
        if granularity_category == "high_frequency":
            reasoning.append("Suitable for high-frequency trading strategies")
        elif granularity_category == "long_term":
            reasoning.append("Appropriate for long-term investment horizons")
        
        # 风险配置理由
        if criteria.risk_profile in [RiskProfile.CONSERVATIVE, RiskProfile.ULTRA_CONSERVATIVE]:
            reasoning.append("Aligned with conservative risk management approach")
        elif criteria.risk_profile in [RiskProfile.AGGRESSIVE, RiskProfile.ULTRA_AGGRESSIVE]:
            reasoning.append("Designed for aggressive return targeting")
        
        return reasoning
    
    def _suggest_parameters(self, reward_type: str, criteria: SelectionCriteria) -> Dict[str, Any]:
        """建议参数配置"""
        base_params = {"lookback_period": 20, "smoothing_factor": 0.1}
        
        # 根据时间粒度调整参数
        if criteria.time_granularity.is_high_frequency():
            base_params["lookback_period"] = 10
            base_params["smoothing_factor"] = 0.2
        elif criteria.time_granularity.category == "long_term":
            base_params["lookback_period"] = 50
            base_params["smoothing_factor"] = 0.05
        
        # 根据风险配置调整
        if criteria.risk_profile in [RiskProfile.CONSERVATIVE, RiskProfile.ULTRA_CONSERVATIVE]:
            base_params["risk_adjustment"] = 0.8
        elif criteria.risk_profile in [RiskProfile.AGGRESSIVE, RiskProfile.ULTRA_AGGRESSIVE]:
            base_params["risk_adjustment"] = 1.2
        
        return base_params
    
    def _generate_warnings(self, reward_type: str, criteria: SelectionCriteria) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        # 复杂度警告
        complexity = self._get_reward_complexity(reward_type)
        if complexity > 7:
            warnings.append("High complexity reward function may require more computational resources")
        
        # 市场兼容性警告
        market_rules = self.rules["market_rules"].get(criteria.market_type, {})
        if reward_type in market_rules.get("avoid", []):
            warnings.append(f"May not be optimal for {criteria.market_type.value} market")
        
        # 数据需求警告
        if reward_type in ["news_sentiment", "correlation_pair"] and criteria.historical_data is None:
            warnings.append("Requires additional data sources for optimal performance")
        
        return warnings
    
    def _create_fallback_recommendation(self, criteria: SelectionCriteria) -> RewardRecommendation:
        """创建后备推荐"""
        return RewardRecommendation(
            reward_type="simple_return",
            category=RewardCategory.BASIC,
            confidence_score=0.5,
            expected_performance={"expected_sharpe": 0.8},
            reasoning=["Fallback to basic reward function"],
            parameters={"lookback_period": 10},
            alternatives=[],
            warnings=["No suitable reward found, using fallback option"]
        )


class PerformanceBasedSelector(BaseRewardSelector):
    """基于性能的奖励选择器"""
    
    def __init__(self):
        super().__init__(SelectionStrategy.PERFORMANCE_BASED)
        self.performance_database = self._initialize_performance_db()
    
    def _initialize_performance_db(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """初始化性能数据库"""
        # 这里应该从实际的回测结果或历史数据中加载
        return {
            "stock": {
                "1d": {
                    "risk_adjusted": {"sharpe": 1.8, "return": 0.12, "drawdown": 0.08},
                    "dividend_adjusted": {"sharpe": 1.5, "return": 0.10, "drawdown": 0.06},
                    "momentum_based": {"sharpe": 1.3, "return": 0.15, "drawdown": 0.12}
                },
                "1h": {
                    "technical_pattern": {"sharpe": 1.6, "return": 0.14, "drawdown": 0.10},
                    "volume_price": {"sharpe": 1.4, "return": 0.11, "drawdown": 0.09}
                }
            },
            "forex": {
                "15min": {
                    "forex_optimized": {"sharpe": 2.1, "return": 0.18, "drawdown": 0.07},
                    "carry_trade": {"sharpe": 1.7, "return": 0.13, "drawdown": 0.09},
                    "momentum_breakout": {"sharpe": 1.4, "return": 0.16, "drawdown": 0.14}
                }
            }
        }
    
    def select_reward(self, criteria: SelectionCriteria) -> RewardRecommendation:
        """基于历史性能选择奖励函数"""
        
        # 获取相关的性能数据
        market_key = criteria.market_type.value
        granularity_key = criteria.time_granularity.value
        
        performance_data = self.performance_database.get(market_key, {}).get(granularity_key, {})
        
        if not performance_data:
            # 如果没有精确匹配，使用规则选择器作为后备
            fallback_selector = RuleBasedSelector()
            return fallback_selector.select_reward(criteria)
        
        # 根据目标指标选择最优奖励
        best_reward = self._select_by_performance_target(performance_data, criteria)
        
        if best_reward:
            reward_type, performance = best_reward
            
            recommendation = RewardRecommendation(
                reward_type=reward_type,
                category=self._get_reward_category(reward_type),
                confidence_score=self._calculate_confidence(performance, criteria),
                expected_performance=performance,
                reasoning=[f"Historical Sharpe ratio: {performance.get('sharpe', 0):.2f}"],
                parameters={},
                alternatives=self._get_performance_alternatives(performance_data, reward_type),
                warnings=[]
            )
            
            self.record_selection(criteria, recommendation)
            return recommendation
        
        # 后备选择
        return self._create_default_recommendation()
    
    def rank_rewards(self, criteria: SelectionCriteria, top_n: int = 5) -> List[RewardRecommendation]:
        """基于性能排序奖励函数"""
        market_key = criteria.market_type.value
        granularity_key = criteria.time_granularity.value
        
        performance_data = self.performance_database.get(market_key, {}).get(granularity_key, {})
        
        if not performance_data:
            return []
        
        # 按目标指标排序
        scored_rewards = []
        for reward_type, performance in performance_data.items():
            score = self._calculate_performance_score(performance, criteria)
            scored_rewards.append((reward_type, score, performance))
        
        scored_rewards.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for i, (reward_type, score, performance) in enumerate(scored_rewards[:top_n]):
            recommendation = RewardRecommendation(
                reward_type=reward_type,
                category=self._get_reward_category(reward_type),
                confidence_score=min(score / 100.0, 1.0),
                expected_performance=performance,
                reasoning=[f"Rank #{i+1} based on historical performance"],
                parameters={},
                alternatives=[],
                warnings=[]
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _select_by_performance_target(
        self, 
        performance_data: Dict[str, Dict[str, float]], 
        criteria: SelectionCriteria
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """根据性能目标选择"""
        
        best_reward = None
        best_score = -float('inf')
        
        for reward_type, performance in performance_data.items():
            score = self._calculate_performance_score(performance, criteria)
            
            if score > best_score:
                best_score = score
                best_reward = (reward_type, performance)
        
        return best_reward
    
    def _calculate_performance_score(
        self, 
        performance: Dict[str, float], 
        criteria: SelectionCriteria
    ) -> float:
        """计算性能得分"""
        score = 0.0
        
        # Sharpe比率权重
        sharpe = performance.get('sharpe', 0)
        if criteria.target_sharpe_ratio:
            score += 50 * min(sharpe / criteria.target_sharpe_ratio, 2.0)
        else:
            score += 30 * min(sharpe / 1.5, 2.0)  # 默认目标1.5
        
        # 回报率权重
        ret = performance.get('return', 0)
        if criteria.target_return:
            score += 30 * min(ret / criteria.target_return, 2.0)
        else:
            score += 20 * min(ret / 0.10, 2.0)  # 默认目标10%
        
        # 最大回撤权重（越小越好）
        drawdown = performance.get('drawdown', 1.0)
        target_drawdown = criteria.target_max_drawdown or 0.15
        if drawdown <= target_drawdown:
            score += 20 * (1 - drawdown / target_drawdown)
        else:
            score -= 20 * (drawdown / target_drawdown - 1)
        
        return score
    
    def _calculate_confidence(
        self, 
        performance: Dict[str, float], 
        criteria: SelectionCriteria
    ) -> float:
        """计算置信度"""
        
        # 基于性能稳定性和目标匹配度
        base_confidence = 0.7
        
        sharpe = performance.get('sharpe', 0)
        if sharpe > 1.5:
            base_confidence += 0.2
        elif sharpe < 0.8:
            base_confidence -= 0.3
        
        return max(0.1, min(1.0, base_confidence))
    
    def _get_performance_alternatives(
        self, 
        performance_data: Dict[str, Dict[str, float]], 
        selected_reward: str
    ) -> List[Tuple[str, float]]:
        """获取性能备选方案"""
        
        alternatives = []
        for reward_type, performance in performance_data.items():
            if reward_type != selected_reward:
                score = performance.get('sharpe', 0) * 50 + performance.get('return', 0) * 100
                alternatives.append((reward_type, score))
        
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[:3]
    
    def _get_reward_category(self, reward_type: str) -> RewardCategory:
        """获取奖励类别"""
        # 简化实现
        category_map = {
            "risk_adjusted": RewardCategory.RISK_ADJUSTED,
            "momentum_based": RewardCategory.MOMENTUM,
            "forex_optimized": RewardCategory.MARKET_SPECIFIC
        }
        return category_map.get(reward_type, RewardCategory.BASIC)
    
    def _create_default_recommendation(self) -> RewardRecommendation:
        """创建默认推荐"""
        return RewardRecommendation(
            reward_type="risk_adjusted",
            category=RewardCategory.RISK_ADJUSTED,
            confidence_score=0.6,
            expected_performance={"sharpe": 1.2, "return": 0.08, "drawdown": 0.10},
            reasoning=["Default selection based on general performance"],
            parameters={},
            alternatives=[],
            warnings=["Limited historical data available"]
        )


class SmartRewardSelector:
    """智能奖励选择器 - 统一接口"""
    
    def __init__(self, default_strategy: SelectionStrategy = SelectionStrategy.HYBRID):
        self.default_strategy = default_strategy
        self.selectors = {
            SelectionStrategy.RULE_BASED: RuleBasedSelector(),
            SelectionStrategy.PERFORMANCE_BASED: PerformanceBasedSelector()
        }
        
        # 使用统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger("SmartRewardSelector")
    
    def select_optimal_reward(
        self, 
        criteria: SelectionCriteria,
        strategy: Optional[SelectionStrategy] = None
    ) -> RewardRecommendation:
        """选择最优奖励函数"""
        
        strategy = strategy or self.default_strategy
        
        if strategy == SelectionStrategy.HYBRID:
            return self._hybrid_selection(criteria)
        elif strategy == SelectionStrategy.CONSENSUS:
            return self._consensus_selection(criteria)
        else:
            selector = self.selectors.get(strategy)
            if selector:
                return selector.select_reward(criteria)
            else:
                # 后备到规则选择
                return self.selectors[SelectionStrategy.RULE_BASED].select_reward(criteria)
    
    def compare_strategies(self, criteria: SelectionCriteria) -> Dict[str, RewardRecommendation]:
        """比较不同策略的选择结果"""
        
        results = {}
        
        for strategy, selector in self.selectors.items():
            try:
                recommendation = selector.select_reward(criteria)
                results[strategy.value] = recommendation
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.value} failed: {e}")
                continue
        
        return results
    
    def _hybrid_selection(self, criteria: SelectionCriteria) -> RewardRecommendation:
        """混合策略选择"""
        
        # 获取多个策略的结果
        rule_recommendation = self.selectors[SelectionStrategy.RULE_BASED].select_reward(criteria)
        
        try:
            perf_recommendation = self.selectors[SelectionStrategy.PERFORMANCE_BASED].select_reward(criteria)
        except:
            # 如果性能选择失败，使用规则选择结果
            return rule_recommendation
        
        # 比较两个结果的置信度
        if perf_recommendation.confidence_score > rule_recommendation.confidence_score:
            # 增强推理信息
            perf_recommendation.reasoning.insert(0, "Hybrid strategy: Performance-based selection preferred")
            return perf_recommendation
        else:
            rule_recommendation.reasoning.insert(0, "Hybrid strategy: Rule-based selection preferred")
            return rule_recommendation
    
    def _consensus_selection(self, criteria: SelectionCriteria) -> RewardRecommendation:
        """共识算法选择"""
        
        # 获取所有策略的排序结果
        all_rankings = {}
        
        for strategy, selector in self.selectors.items():
            try:
                rankings = selector.rank_rewards(criteria, top_n=5)
                all_rankings[strategy] = rankings
            except:
                continue
        
        if not all_rankings:
            # 如果所有策略都失败，使用默认
            return self.selectors[SelectionStrategy.RULE_BASED].select_reward(criteria)
        
        # 计算共识得分
        consensus_scores = {}
        
        for strategy, rankings in all_rankings.items():
            for i, recommendation in enumerate(rankings):
                reward_type = recommendation.reward_type
                score = (len(rankings) - i) * recommendation.confidence_score
                
                if reward_type not in consensus_scores:
                    consensus_scores[reward_type] = []
                consensus_scores[reward_type].append(score)
        
        # 计算总分
        final_scores = {}
        for reward_type, scores in consensus_scores.items():
            final_scores[reward_type] = sum(scores) / len(scores)
        
        # 选择得分最高的
        best_reward = max(final_scores, key=final_scores.get)
        
        # 构建共识推荐
        return RewardRecommendation(
            reward_type=best_reward,
            category=RewardCategory.BASIC,  # 简化
            confidence_score=final_scores[best_reward] / 100.0,
            expected_performance={},
            reasoning=[f"Consensus selection from {len(all_rankings)} strategies"],
            parameters={},
            alternatives=list(final_scores.items())[:5],
            warnings=[]
        )
    
    def get_selector_statistics(self) -> Dict[str, Any]:
        """获取选择器统计信息"""
        
        stats = {}
        
        for strategy, selector in self.selectors.items():
            stats[strategy.value] = selector.get_selection_statistics()
        
        return stats


# 便利函数
def select_reward_for_trading(
    market_type: Union[str, MarketType],
    time_granularity: Union[str, TimeGranularity],
    risk_profile: Union[str, RiskProfile] = RiskProfile.BALANCED,
    strategy: SelectionStrategy = SelectionStrategy.HYBRID,
    **kwargs
) -> RewardRecommendation:
    """便利函数：为交易场景选择奖励函数"""
    
    # 类型转换
    if isinstance(market_type, str):
        market_type = MarketType.from_string(market_type)
    if isinstance(time_granularity, str):
        time_granularity = TimeGranularity.from_string(time_granularity)
    if isinstance(risk_profile, str):
        risk_profile = RiskProfile.from_string(risk_profile)
    
    # 设置默认环境特征
    environment_features = kwargs.pop('environment_features', EnvironmentFeature(0))
    
    # 创建选择标准
    criteria = SelectionCriteria(
        market_type=market_type,
        time_granularity=time_granularity,
        risk_profile=risk_profile,
        environment_features=environment_features,
        **kwargs
    )
    
    # 创建选择器并选择
    selector = SmartRewardSelector()
    return selector.select_optimal_reward(criteria, strategy)