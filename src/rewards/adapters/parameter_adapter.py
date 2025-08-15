"""
动态参数调整器 - Dynamic Parameter Adapter

实现奖励函数参数的动态调整和自适应功能，根据市场条件和性能反馈自动优化参数。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
from datetime import datetime, timedelta
import warnings

from ..core.reward_context import RewardContext
from ..core.base_reward import BaseReward
from ..enums.market_types import MarketType
from ..enums.time_granularities import TimeGranularity
# 从环境模块导入 EnvironmentFeature
from ...environment.enums.environment_features import EnvironmentFeature


class AdaptationStrategy(Enum):
    """自适应策略"""
    PERFORMANCE_BASED = "performance_based"  # 基于性能的调整
    MARKET_REGIME = "market_regime"  # 市场状态调整
    VOLATILITY_SCALING = "volatility_scaling"  # 波动率缩放
    TREND_FOLLOWING = "trend_following"  # 趋势跟踪
    MOMENTUM_ADAPTIVE = "momentum_adaptive"  # 动量自适应
    RISK_PARITY = "risk_parity"  # 风险平价
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # 强化学习
    ENSEMBLE = "ensemble"  # 集成策略


class MarketRegime(Enum):
    """市场状态"""
    BULL_MARKET = "bull_market"  # 牛市
    BEAR_MARKET = "bear_market"  # 熊市
    SIDEWAYS = "sideways"  # 震荡市
    HIGH_VOLATILITY = "high_volatility"  # 高波动
    LOW_VOLATILITY = "low_volatility"  # 低波动
    CRISIS = "crisis"  # 危机
    RECOVERY = "recovery"  # 恢复


@dataclass
class ParameterBounds:
    """参数边界"""
    min_value: Union[float, int]
    max_value: Union[float, int]
    step_size: Optional[float] = None
    constraint_type: str = "continuous"  # continuous, discrete, categorical
    allowed_values: Optional[List[Any]] = None


@dataclass
class AdaptationConfig:
    """自适应配置"""
    adaptation_strategy: AdaptationStrategy
    adaptation_frequency: int = 10  # 调整频率（步数）
    lookback_window: int = 50  # 回顾窗口
    performance_threshold: float = 0.05  # 性能阈值
    adaptation_rate: float = 0.1  # 调整速率
    stability_requirement: float = 0.8  # 稳定性要求
    parameter_bounds: Dict[str, ParameterBounds] = field(default_factory=dict)
    market_sensitivity: float = 1.0  # 市场敏感度
    risk_tolerance: float = 0.1  # 风险容忍度


@dataclass
class AdaptationResult:
    """自适应结果"""
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    performance_improvement: float
    confidence_score: float
    adaptation_reason: str
    market_regime: Optional[MarketRegime]
    timestamp: datetime
    validation_results: Dict[str, Any]


@dataclass
class MarketContext:
    """市场上下文"""
    volatility: float
    trend_strength: float
    momentum: float
    volume_profile: float
    correlation_breakdown: float
    regime: MarketRegime
    features: Dict[str, float] = field(default_factory=dict)


class BaseParameterAdapter(ABC):
    """参数适配器基类"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        # 使用统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # 适配历史
        self.adaptation_history: List[AdaptationResult] = []
        
        # 性能追踪
        self.performance_tracker = PerformanceTracker()
        
        # 市场状态检测器
        self.regime_detector = MarketRegimeDetector()
        
        # 参数历史
        self.parameter_history: Dict[str, List[Any]] = {}
        
        # 适配计数器
        self.adaptation_counter = 0
    
    @abstractmethod
    def adapt_parameters(
        self,
        reward_function: BaseReward,
        market_context: MarketContext,
        performance_history: List[float],
        current_step: int
    ) -> AdaptationResult:
        """适配参数"""
        pass
    
    @abstractmethod
    def should_adapt(
        self,
        market_context: MarketContext,
        performance_history: List[float],
        current_step: int
    ) -> bool:
        """判断是否需要适配"""
        pass
    
    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        reward_function: BaseReward
    ) -> Tuple[bool, List[str]]:
        """验证参数有效性"""
        errors = []
        
        for param_name, param_value in parameters.items():
            if param_name in self.config.parameter_bounds:
                bounds = self.config.parameter_bounds[param_name]
                
                # 检查数值边界
                if bounds.constraint_type == "continuous":
                    if not (bounds.min_value <= param_value <= bounds.max_value):
                        errors.append(f"Parameter {param_name} = {param_value} out of bounds [{bounds.min_value}, {bounds.max_value}]")
                
                elif bounds.constraint_type == "discrete":
                    if param_value not in range(int(bounds.min_value), int(bounds.max_value) + 1):
                        errors.append(f"Parameter {param_name} = {param_value} not in discrete range [{bounds.min_value}, {bounds.max_value}]")
                
                elif bounds.constraint_type == "categorical":
                    if bounds.allowed_values and param_value not in bounds.allowed_values:
                        errors.append(f"Parameter {param_name} = {param_value} not in allowed values {bounds.allowed_values}")
        
        return len(errors) == 0, errors
    
    def record_adaptation(self, result: AdaptationResult):
        """记录适配结果"""
        self.adaptation_history.append(result)
        
        # 更新参数历史
        for param_name, param_value in result.new_parameters.items():
            if param_name not in self.parameter_history:
                self.parameter_history[param_name] = []
            self.parameter_history[param_name].append(param_value)
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """获取适配统计"""
        if not self.adaptation_history:
            return {"total_adaptations": 0}
        
        improvements = [r.performance_improvement for r in self.adaptation_history]
        confidence_scores = [r.confidence_score for r in self.adaptation_history]
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "average_improvement": np.mean(improvements),
            "success_rate": np.mean([imp > 0 for imp in improvements]),
            "average_confidence": np.mean(confidence_scores),
            "adaptation_frequency": len(self.adaptation_history) / max(1, self.adaptation_counter)
        }


class PerformanceBasedAdapter(BaseParameterAdapter):
    """基于性能的参数适配器"""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__(config)
        self.performance_buffer = []
        self.baseline_performance = None
    
    def should_adapt(
        self,
        market_context: MarketContext,
        performance_history: List[float],
        current_step: int
    ) -> bool:
        """判断是否需要适配"""
        
        # 检查适配频率
        if current_step % self.config.adaptation_frequency != 0:
            return False
        
        # 需要足够的历史数据
        if len(performance_history) < self.config.lookback_window:
            return False
        
        # 计算近期性能
        recent_performance = np.mean(performance_history[-self.config.lookback_window//2:])
        historical_performance = np.mean(performance_history[-self.config.lookback_window:-self.config.lookback_window//2])
        
        # 性能下降超过阈值
        performance_decline = historical_performance - recent_performance
        
        return performance_decline > self.config.performance_threshold
    
    def adapt_parameters(
        self,
        reward_function: BaseReward,
        market_context: MarketContext,
        performance_history: List[float],
        current_step: int
    ) -> AdaptationResult:
        """基于性能适配参数"""
        
        old_parameters = self._extract_current_parameters(reward_function)
        
        # 分析性能模式
        performance_analysis = self._analyze_performance_pattern(performance_history)
        
        # 根据性能问题调整参数
        new_parameters = self._adjust_parameters_for_performance(
            old_parameters, 
            performance_analysis,
            market_context
        )
        
        # 验证参数
        is_valid, errors = self.validate_parameters(new_parameters, reward_function)
        if not is_valid:
            self.logger.warning(f"Invalid parameters generated: {errors}")
            new_parameters = old_parameters
        
        # 应用参数
        self._apply_parameters(reward_function, new_parameters)
        
        # 估计性能改善
        performance_improvement = self._estimate_performance_improvement(
            performance_analysis, new_parameters, old_parameters
        )
        
        # 计算置信度
        confidence_score = self._calculate_confidence(performance_analysis, market_context)
        
        result = AdaptationResult(
            old_parameters=old_parameters,
            new_parameters=new_parameters,
            performance_improvement=performance_improvement,
            confidence_score=confidence_score,
            adaptation_reason="Performance-based optimization",
            market_regime=market_context.regime,
            timestamp=datetime.now(),
            validation_results={}
        )
        
        self.record_adaptation(result)
        return result
    
    def _extract_current_parameters(self, reward_function: BaseReward) -> Dict[str, Any]:
        """提取当前参数"""
        parameters = {}
        
        # 通用参数
        common_params = ['lookback_period', 'smoothing_factor', 'risk_adjustment', 
                        'volatility_window', 'threshold', 'alpha', 'beta']
        
        for param in common_params:
            if hasattr(reward_function, param):
                parameters[param] = getattr(reward_function, param)
        
        return parameters
    
    def _analyze_performance_pattern(self, performance_history: List[float]) -> Dict[str, Any]:
        """分析性能模式"""
        if len(performance_history) < 10:
            return {"trend": "insufficient_data", "volatility": 0.0, "stability": 0.0}
        
        performance_array = np.array(performance_history[-self.config.lookback_window:])
        
        # 趋势分析
        x = np.arange(len(performance_array))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, performance_array)
        
        # 波动率分析
        volatility = np.std(performance_array)
        
        # 稳定性分析
        rolling_std = pd.Series(performance_array).rolling(window=10).std().fillna(0)
        stability = 1.0 / (1.0 + np.mean(rolling_std))
        
        # 异常值检测
        z_scores = np.abs(stats.zscore(performance_array))
        outlier_ratio = np.mean(z_scores > 2.0)
        
        return {
            "trend": "declining" if slope < -0.001 else "improving" if slope > 0.001 else "stable",
            "slope": slope,
            "r_squared": r_value ** 2,
            "volatility": volatility,
            "stability": stability,
            "outlier_ratio": outlier_ratio,
            "recent_performance": np.mean(performance_array[-10:]),
            "historical_performance": np.mean(performance_array[:-10])
        }
    
    def _adjust_parameters_for_performance(
        self,
        current_parameters: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        market_context: MarketContext
    ) -> Dict[str, Any]:
        """根据性能调整参数"""
        
        new_parameters = current_parameters.copy()
        
        # 根据趋势调整
        if performance_analysis["trend"] == "declining":
            # 性能下降，尝试增加敏感度
            if "lookback_period" in new_parameters:
                new_parameters["lookback_period"] = max(5, int(new_parameters["lookback_period"] * 0.8))
            
            if "smoothing_factor" in new_parameters:
                new_parameters["smoothing_factor"] = min(0.5, new_parameters["smoothing_factor"] * 1.2)
        
        elif performance_analysis["trend"] == "improving":
            # 性能改善，保持稳定
            if "smoothing_factor" in new_parameters:
                new_parameters["smoothing_factor"] = max(0.01, new_parameters["smoothing_factor"] * 0.9)
        
        # 根据波动率调整
        if performance_analysis["volatility"] > 0.1:  # 高波动
            if "risk_adjustment" in new_parameters:
                new_parameters["risk_adjustment"] = min(2.0, new_parameters["risk_adjustment"] * 1.1)
        
        # 根据市场状态调整
        if market_context.regime == MarketRegime.HIGH_VOLATILITY:
            if "volatility_window" in new_parameters:
                new_parameters["volatility_window"] = min(50, int(new_parameters["volatility_window"] * 1.2))
        
        elif market_context.regime == MarketRegime.LOW_VOLATILITY:
            if "volatility_window" in new_parameters:
                new_parameters["volatility_window"] = max(10, int(new_parameters["volatility_window"] * 0.8))
        
        return new_parameters
    
    def _apply_parameters(self, reward_function: BaseReward, parameters: Dict[str, Any]):
        """应用参数到奖励函数"""
        for param_name, param_value in parameters.items():
            if hasattr(reward_function, param_name):
                setattr(reward_function, param_name, param_value)
    
    def _estimate_performance_improvement(
        self,
        performance_analysis: Dict[str, Any],
        new_parameters: Dict[str, Any],
        old_parameters: Dict[str, Any]
    ) -> float:
        """估计性能改善"""
        
        # 简化的改善估计
        base_improvement = 0.0
        
        # 根据参数变化程度估计
        for param_name in new_parameters:
            if param_name in old_parameters:
                old_val = old_parameters[param_name]
                new_val = new_parameters[param_name]
                
                if isinstance(old_val, (int, float)) and old_val != 0:
                    change_ratio = abs(new_val - old_val) / abs(old_val)
                    base_improvement += change_ratio * 0.01  # 1%改善每10%参数变化
        
        # 根据性能趋势调整
        if performance_analysis["trend"] == "declining":
            base_improvement *= 2.0  # 下降趋势时改善潜力更大
        
        return min(0.1, base_improvement)  # 限制最大估计改善为10%
    
    def _calculate_confidence(
        self,
        performance_analysis: Dict[str, Any],
        market_context: MarketContext
    ) -> float:
        """计算置信度"""
        
        confidence = 0.5  # 基础置信度
        
        # 基于数据质量
        if performance_analysis["r_squared"] > 0.7:
            confidence += 0.2
        elif performance_analysis["r_squared"] > 0.4:
            confidence += 0.1
        
        # 基于稳定性
        confidence += performance_analysis["stability"] * 0.2
        
        # 基于异常值比例
        confidence -= performance_analysis["outlier_ratio"] * 0.3
        
        # 基于市场状态
        if market_context.regime in [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET]:
            confidence += 0.1  # 明确趋势时置信度更高
        elif market_context.regime == MarketRegime.CRISIS:
            confidence -= 0.2  # 危机时置信度降低
        
        return max(0.1, min(0.9, confidence))


class MarketRegimeAdapter(BaseParameterAdapter):
    """市场状态适配器"""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__(config)
        self.regime_parameters = self._initialize_regime_parameters()
    
    def _initialize_regime_parameters(self) -> Dict[MarketRegime, Dict[str, Any]]:
        """初始化不同市场状态的参数配置"""
        return {
            MarketRegime.BULL_MARKET: {
                "lookback_period": 15,
                "smoothing_factor": 0.1,
                "risk_adjustment": 0.8,
                "threshold": 0.02
            },
            MarketRegime.BEAR_MARKET: {
                "lookback_period": 25,
                "smoothing_factor": 0.05,
                "risk_adjustment": 1.2,
                "threshold": -0.01
            },
            MarketRegime.SIDEWAYS: {
                "lookback_period": 30,
                "smoothing_factor": 0.15,
                "risk_adjustment": 1.0,
                "threshold": 0.0
            },
            MarketRegime.HIGH_VOLATILITY: {
                "lookback_period": 10,
                "smoothing_factor": 0.2,
                "risk_adjustment": 1.5,
                "volatility_window": 20
            },
            MarketRegime.LOW_VOLATILITY: {
                "lookback_period": 40,
                "smoothing_factor": 0.05,
                "risk_adjustment": 0.6,
                "volatility_window": 50
            }
        }
    
    def should_adapt(
        self,
        market_context: MarketContext,
        performance_history: List[float],
        current_step: int
    ) -> bool:
        """判断是否需要适配"""
        
        # 检查适配频率
        if current_step % self.config.adaptation_frequency != 0:
            return False
        
        # 市场状态变化时需要适配
        if len(self.adaptation_history) == 0:
            return True
        
        last_regime = self.adaptation_history[-1].market_regime
        return last_regime != market_context.regime
    
    def adapt_parameters(
        self,
        reward_function: BaseReward,
        market_context: MarketContext,
        performance_history: List[float],
        current_step: int
    ) -> AdaptationResult:
        """根据市场状态适配参数"""
        
        old_parameters = self._extract_current_parameters(reward_function)
        
        # 获取当前市场状态的最优参数
        if market_context.regime in self.regime_parameters:
            regime_params = self.regime_parameters[market_context.regime]
            new_parameters = {**old_parameters, **regime_params}
        else:
            new_parameters = old_parameters
        
        # 根据市场强度微调
        new_parameters = self._fine_tune_for_market_strength(
            new_parameters, market_context
        )
        
        # 验证参数
        is_valid, errors = self.validate_parameters(new_parameters, reward_function)
        if not is_valid:
            self.logger.warning(f"Invalid parameters for regime {market_context.regime}: {errors}")
            new_parameters = old_parameters
        
        # 应用参数
        self._apply_parameters(reward_function, new_parameters)
        
        # 估计改善（根据历史市场状态表现）
        performance_improvement = self._estimate_regime_improvement(
            market_context.regime, performance_history
        )
        
        result = AdaptationResult(
            old_parameters=old_parameters,
            new_parameters=new_parameters,
            performance_improvement=performance_improvement,
            confidence_score=0.8,  # 市场状态适配置信度较高
            adaptation_reason=f"Market regime adaptation: {market_context.regime.value}",
            market_regime=market_context.regime,
            timestamp=datetime.now(),
            validation_results={}
        )
        
        self.record_adaptation(result)
        return result
    
    def _fine_tune_for_market_strength(
        self,
        parameters: Dict[str, Any],
        market_context: MarketContext
    ) -> Dict[str, Any]:
        """根据市场强度微调参数"""
        
        tuned_params = parameters.copy()
        
        # 根据趋势强度调整
        if market_context.trend_strength > 0.7:  # 强趋势
            if "lookback_period" in tuned_params:
                tuned_params["lookback_period"] = int(tuned_params["lookback_period"] * 0.8)
        elif market_context.trend_strength < 0.3:  # 弱趋势
            if "lookback_period" in tuned_params:
                tuned_params["lookback_period"] = int(tuned_params["lookback_period"] * 1.3)
        
        # 根据波动率调整
        vol_factor = market_context.volatility / 0.02  # 假设基准波动率2%
        if "risk_adjustment" in tuned_params:
            tuned_params["risk_adjustment"] *= vol_factor
        
        return tuned_params
    
    def _estimate_regime_improvement(
        self,
        regime: MarketRegime,
        performance_history: List[float]
    ) -> float:
        """估计市场状态适配的改善"""
        
        # 基于历史表现的简化估计
        regime_improvements = {
            MarketRegime.BULL_MARKET: 0.03,
            MarketRegime.BEAR_MARKET: 0.02,
            MarketRegime.SIDEWAYS: 0.01,
            MarketRegime.HIGH_VOLATILITY: 0.04,
            MarketRegime.LOW_VOLATILITY: 0.015,
            MarketRegime.CRISIS: 0.05,
            MarketRegime.RECOVERY: 0.035
        }
        
        base_improvement = regime_improvements.get(regime, 0.01)
        
        # 根据最近性能调整
        if len(performance_history) >= 20:
            recent_perf = np.mean(performance_history[-10:])
            if recent_perf < 0:  # 最近表现不佳时，适配改善更明显
                base_improvement *= 1.5
        
        return base_improvement
    
    def _extract_current_parameters(self, reward_function: BaseReward) -> Dict[str, Any]:
        """提取当前参数（复用PerformanceBasedAdapter的实现）"""
        adapter = PerformanceBasedAdapter(self.config)
        return adapter._extract_current_parameters(reward_function)
    
    def _apply_parameters(self, reward_function: BaseReward, parameters: Dict[str, Any]):
        """应用参数（复用PerformanceBasedAdapter的实现）"""
        adapter = PerformanceBasedAdapter(self.config)
        adapter._apply_parameters(reward_function, parameters)


class PerformanceTracker:
    """性能追踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = []
        self.timestamps = []
    
    def update(self, performance: float, timestamp: datetime = None):
        """更新性能"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.performance_history.append(performance)
        self.timestamps.append(timestamp)
        
        # 保持窗口大小
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            self.timestamps.pop(0)
    
    def get_recent_performance(self, n: int = 10) -> List[float]:
        """获取最近的性能"""
        return self.performance_history[-n:]
    
    def get_performance_statistics(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.performance_history:
            return {}
        
        perf_array = np.array(self.performance_history)
        
        return {
            "mean": np.mean(perf_array),
            "std": np.std(perf_array),
            "min": np.min(perf_array),
            "max": np.max(perf_array),
            "sharpe": np.mean(perf_array) / (np.std(perf_array) + 1e-8),
            "trend": np.corrcoef(range(len(perf_array)), perf_array)[0, 1] if len(perf_array) > 1 else 0.0
        }


class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self):
        self.regime_history = []
        self.features_history = []
    
    def detect_regime(
        self,
        price_history: List[float],
        volume_history: Optional[List[float]] = None,
        additional_features: Optional[Dict[str, List[float]]] = None
    ) -> MarketRegime:
        """检测市场状态"""
        
        if len(price_history) < 20:
            return MarketRegime.SIDEWAYS
        
        prices = np.array(price_history[-50:])  # 使用最近50个数据点
        
        # 计算技术指标
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 趋势检测
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        trend_strength = abs(r_value)
        
        # 移动平均
        if len(prices) >= 20:
            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            ma_ratio = ma_short / ma_long
        else:
            ma_ratio = 1.0
        
        # 状态判断逻辑
        if volatility > 0.3:  # 高波动率阈值
            if trend_strength > 0.7 and slope < -0.1:
                return MarketRegime.CRISIS
            else:
                return MarketRegime.HIGH_VOLATILITY
        
        elif volatility < 0.1:  # 低波动率阈值
            return MarketRegime.LOW_VOLATILITY
        
        elif trend_strength > 0.6:  # 强趋势
            if slope > 0 and ma_ratio > 1.02:
                return MarketRegime.BULL_MARKET
            elif slope < 0 and ma_ratio < 0.98:
                return MarketRegime.BEAR_MARKET
            elif slope > 0 and ma_ratio > 0.98:  # 从熊市恢复
                return MarketRegime.RECOVERY
        
        # 默认震荡市
        return MarketRegime.SIDEWAYS
    
    def get_market_context(
        self,
        price_history: List[float],
        volume_history: Optional[List[float]] = None
    ) -> MarketContext:
        """获取市场上下文"""
        
        regime = self.detect_regime(price_history, volume_history)
        
        if len(price_history) < 10:
            return MarketContext(
                volatility=0.0,
                trend_strength=0.0,
                momentum=0.0,
                volume_profile=1.0,
                correlation_breakdown=0.0,
                regime=regime
            )
        
        prices = np.array(price_history[-30:])
        returns = np.diff(prices) / prices[:-1]
        
        # 计算各项指标
        volatility = np.std(returns) * np.sqrt(252)
        
        # 趋势强度
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        trend_strength = abs(r_value)
        
        # 动量（价格动量）
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / prices[-10]
        else:
            momentum = 0.0
        
        # 成交量特征
        if volume_history and len(volume_history) >= 10:
            volumes = np.array(volume_history[-10:])
            volume_profile = np.mean(volumes) / (np.std(volumes) + 1e-8)
        else:
            volume_profile = 1.0
        
        # 相关性破坏（简化）
        correlation_breakdown = min(1.0, volatility * 2)
        
        return MarketContext(
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            volume_profile=volume_profile,
            correlation_breakdown=correlation_breakdown,
            regime=regime
        )


class ParameterAdapterFactory:
    """参数适配器工厂"""
    
    @staticmethod
    def create_adapter(
        strategy: AdaptationStrategy,
        config: AdaptationConfig
    ) -> BaseParameterAdapter:
        """创建参数适配器"""
        
        if strategy == AdaptationStrategy.PERFORMANCE_BASED:
            return PerformanceBasedAdapter(config)
        
        elif strategy == AdaptationStrategy.MARKET_REGIME:
            return MarketRegimeAdapter(config)
        
        else:
            raise ValueError(f"Unsupported adaptation strategy: {strategy}")


# 便利函数
def create_parameter_adapter(
    strategy: str = "performance_based",
    adaptation_frequency: int = 10,
    lookback_window: int = 50,
    performance_threshold: float = 0.05,
    adaptation_rate: float = 0.1,
    parameter_bounds: Optional[Dict[str, Dict[str, Any]]] = None
) -> BaseParameterAdapter:
    """便利函数：创建参数适配器"""
    
    # 转换枚举
    strategy_enum = AdaptationStrategy(strategy)
    
    # 处理参数边界
    bounds_dict = {}
    if parameter_bounds:
        for param_name, bounds_spec in parameter_bounds.items():
            bounds_dict[param_name] = ParameterBounds(**bounds_spec)
    
    # 创建配置
    config = AdaptationConfig(
        adaptation_strategy=strategy_enum,
        adaptation_frequency=adaptation_frequency,
        lookback_window=lookback_window,
        performance_threshold=performance_threshold,
        adaptation_rate=adaptation_rate,
        parameter_bounds=bounds_dict
    )
    
    return ParameterAdapterFactory.create_adapter(strategy_enum, config)


def detect_market_regime(
    price_history: List[float],
    volume_history: Optional[List[float]] = None
) -> MarketRegime:
    """便利函数：检测市场状态"""
    detector = MarketRegimeDetector()
    return detector.detect_regime(price_history, volume_history)


def get_market_context(
    price_history: List[float],
    volume_history: Optional[List[float]] = None
) -> MarketContext:
    """便利函数：获取市场上下文"""
    detector = MarketRegimeDetector()
    return detector.get_market_context(price_history, volume_history)