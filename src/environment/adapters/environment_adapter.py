"""
环境适配系统 - Environment Adaptation System

实现环境的自动适配、配置和优化功能，根据市场条件和数据特征自动选择最优环境配置。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import warnings

# 从奖励模块导入必要的组件
from ...rewards.core.reward_context import RewardContext
from ...rewards.core.base_reward import BaseReward
from ...rewards.enums.market_types import MarketType
from ...rewards.enums.time_granularities import TimeGranularity
from ...rewards.enums.risk_profiles import RiskProfile
from ..enums.environment_features import EnvironmentFeature
from ...rewards.selectors.reward_selector import SmartRewardSelector, SelectionCriteria
from ...rewards.optimizers.reward_optimizer import RewardOptimizerFactory, OptimizationMethod, OptimizationObjective
from ...rewards.adapters.parameter_adapter import MarketRegimeDetector, MarketContext, MarketRegime

# 临时环境配置类定义（待后续迁移完成时替换）
@dataclass
class EnvironmentConfiguration:
    """环境配置（临时定义）"""
    market_type: MarketType
    time_granularity: TimeGranularity
    risk_profile: RiskProfile
    environment_features: EnvironmentFeature
    initial_balance: float = 10000.0
    transaction_costs: float = 0.001
    window_size: int = 20
    max_steps: int = 1000
    normalize_observations: bool = True
    include_positions: bool = True
    include_portfolio_metrics: bool = True

class BaseTradingEnvironment:
    """基础交易环境（临时定义）"""
    def __init__(self, config: EnvironmentConfiguration):
        self.config = config

def create_trading_environment(config: EnvironmentConfiguration) -> BaseTradingEnvironment:
    """创建交易环境（临时实现）"""
    return BaseTradingEnvironment(config)


class AdaptationLevel(Enum):
    """适配级别"""
    BASIC = "basic"  # 基础适配
    INTERMEDIATE = "intermediate"  # 中级适配
    ADVANCED = "advanced"  # 高级适配
    EXPERT = "expert"  # 专家级适配


class EnvironmentOptimizationGoal(Enum):
    """环境优化目标"""
    PERFORMANCE = "performance"  # 性能优化
    STABILITY = "stability"  # 稳定性优化
    ROBUSTNESS = "robustness"  # 鲁棒性优化
    EFFICIENCY = "efficiency"  # 效率优化
    RISK_ADJUSTED = "risk_adjusted"  # 风险调整优化
    MULTI_OBJECTIVE = "multi_objective"  # 多目标优化


@dataclass
class DataCharacteristics:
    """数据特征"""
    sample_size: int
    feature_count: int
    data_quality_score: float
    missing_ratio: float
    noise_level: float
    stationarity_score: float
    seasonality_detected: bool
    outlier_ratio: float
    correlation_structure: Dict[str, float]
    temporal_consistency: float


@dataclass 
class EnvironmentAdaptationConfig:
    """环境适配配置"""
    adaptation_level: AdaptationLevel
    optimization_goal: EnvironmentOptimizationGoal
    auto_feature_detection: bool = True
    auto_reward_selection: bool = True
    auto_parameter_tuning: bool = True
    performance_tracking: bool = True
    regime_adaptation: bool = True
    constraint_enforcement: bool = True
    validation_enabled: bool = True
    adaptation_frequency: int = 100  # 适配频率（步数）
    minimum_adaptation_interval: int = 50  # 最小适配间隔
    performance_threshold: float = 0.05  # 性能阈值
    confidence_threshold: float = 0.7  # 置信度阈值


@dataclass
class EnvironmentAdaptationResult:
    """环境适配结果"""
    old_configuration: EnvironmentConfiguration
    new_configuration: EnvironmentConfiguration
    adaptation_reason: str
    performance_improvement: float
    confidence_score: float
    validation_results: Dict[str, Any]
    adaptation_metrics: Dict[str, float]
    recommendations: List[str]
    warnings: List[str]
    timestamp: datetime


class BaseEnvironmentAdapter(ABC):
    """环境适配器基类"""
    
    def __init__(self, config: EnvironmentAdaptationConfig):
        self.config = config
        # 使用统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # 适配历史
        self.adaptation_history: List[EnvironmentAdaptationResult] = []
        
        # 组件
        self.regime_detector = MarketRegimeDetector()
        self.reward_selector = SmartRewardSelector()
        
        # 性能监控
        self.performance_baseline = None
        self.adaptation_counter = 0
    
    @abstractmethod
    def adapt_environment(
        self,
        current_env: BaseTradingEnvironment,
        data: pd.DataFrame,
        performance_history: List[float],
        current_step: int
    ) -> EnvironmentAdaptationResult:
        """适配环境"""
        pass
    
    @abstractmethod
    def should_adapt(
        self,
        current_env: BaseTradingEnvironment,
        data: pd.DataFrame,
        performance_history: List[float],
        current_step: int
    ) -> bool:
        """判断是否需要适配"""
        pass
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> DataCharacteristics:
        """分析数据特征"""
        
        if data.empty:
            return DataCharacteristics(
                sample_size=0, feature_count=0, data_quality_score=0.0,
                missing_ratio=1.0, noise_level=1.0, stationarity_score=0.0,
                seasonality_detected=False, outlier_ratio=1.0,
                correlation_structure={}, temporal_consistency=0.0
            )
        
        sample_size = len(data)
        feature_count = len(data.columns)
        
        # 数据质量评分
        missing_ratio = data.isnull().sum().sum() / (sample_size * feature_count)
        data_quality_score = 1.0 - missing_ratio
        
        # 噪声水平（基于数值列的变异系数）
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            cv_values = []
            for col in numeric_cols:
                if data[col].std() > 0 and data[col].mean() != 0:
                    cv = data[col].std() / abs(data[col].mean())
                    cv_values.append(cv)
            noise_level = np.mean(cv_values) if cv_values else 0.0
        else:
            noise_level = 0.0
        
        # 平稳性评分（简化实现）
        stationarity_score = self._estimate_stationarity(data)
        
        # 季节性检测（简化）
        seasonality_detected = self._detect_seasonality(data)
        
        # 异常值比例
        outlier_ratio = self._calculate_outlier_ratio(data)
        
        # 相关性结构
        correlation_structure = self._analyze_correlation_structure(data)
        
        # 时间一致性
        temporal_consistency = self._calculate_temporal_consistency(data)
        
        return DataCharacteristics(
            sample_size=sample_size,
            feature_count=feature_count,
            data_quality_score=data_quality_score,
            missing_ratio=missing_ratio,
            noise_level=noise_level,
            stationarity_score=stationarity_score,
            seasonality_detected=seasonality_detected,
            outlier_ratio=outlier_ratio,
            correlation_structure=correlation_structure,
            temporal_consistency=temporal_consistency
        )
    
    def _estimate_stationarity(self, data: pd.DataFrame) -> float:
        """估算平稳性（简化实现）"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
        
        stationarity_scores = []
        for col in numeric_cols[:3]:  # 只检查前3列避免计算量过大
            try:
                series = data[col].dropna()
                if len(series) < 10:
                    continue
                
                # 使用简单的滑窗方差分析
                window_size = min(20, len(series) // 5)
                if window_size < 5:
                    continue
                
                rolling_var = series.rolling(window=window_size).var().dropna()
                if len(rolling_var) > 1:
                    var_stability = 1.0 / (1.0 + np.std(rolling_var) / (np.mean(rolling_var) + 1e-8))
                    stationarity_scores.append(var_stability)
            except:
                continue
        
        return np.mean(stationarity_scores) if stationarity_scores else 0.5
    
    def _detect_seasonality(self, data: pd.DataFrame) -> bool:
        """检测季节性（简化实现）"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0 or len(data) < 50:
            return False
        
        # 检查是否有明显的周期性模式
        try:
            main_col = numeric_cols[0]
            series = data[main_col].dropna()
            
            if len(series) < 50:
                return False
            
            # 简单的自相关分析
            max_lag = min(20, len(series) // 4)
            autocorr_values = []
            
            for lag in range(1, max_lag + 1):
                if len(series) > lag:
                    shifted = series.shift(lag)
                    corr = series.corr(shifted)
                    if not np.isnan(corr):
                        autocorr_values.append(abs(corr))
            
            # 如果存在高自相关，可能有季节性
            if autocorr_values:
                max_autocorr = max(autocorr_values)
                return bool(max_autocorr > 0.3)
            
        except:
            pass
        
        return False
    
    def _calculate_outlier_ratio(self, data: pd.DataFrame) -> float:
        """计算异常值比例"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
        
        total_outliers = 0
        total_values = 0
        
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < 4:
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                
                total_outliers += outliers
                total_values += len(series)
        
        return total_outliers / total_values if total_values > 0 else 0.0
    
    def _analyze_correlation_structure(self, data: pd.DataFrame) -> Dict[str, float]:
        """分析相关性结构"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"max_correlation": 0.0, "mean_correlation": 0.0, "correlation_diversity": 1.0}
        
        try:
            corr_matrix = data[numeric_cols].corr()
            
            # 提取上三角（不包括对角线）
            upper_triangle = np.triu(corr_matrix.values, k=1)
            correlations = upper_triangle[upper_triangle != 0]
            
            if len(correlations) == 0:
                return {"max_correlation": 0.0, "mean_correlation": 0.0, "correlation_diversity": 1.0}
            
            # 去除NaN值
            correlations = correlations[~np.isnan(correlations)]
            
            if len(correlations) == 0:
                return {"max_correlation": 0.0, "mean_correlation": 0.0, "correlation_diversity": 1.0}
            
            max_corr = np.max(np.abs(correlations))
            mean_corr = np.mean(np.abs(correlations))
            correlation_diversity = 1.0 - mean_corr  # 相关性越高，多样性越低
            
            return {
                "max_correlation": max_corr,
                "mean_correlation": mean_corr,
                "correlation_diversity": correlation_diversity
            }
        
        except:
            return {"max_correlation": 0.0, "mean_correlation": 0.0, "correlation_diversity": 1.0}
    
    def _calculate_temporal_consistency(self, data: pd.DataFrame) -> float:
        """计算时间一致性"""
        if len(data) < 10:
            return 0.0
        
        # 检查数据的时间顺序一致性
        try:
            # 如果有时间索引
            if isinstance(data.index, pd.DatetimeIndex):
                time_diffs = data.index.to_series().diff().dropna()
                if len(time_diffs) > 1:
                    # 计算时间间隔的一致性
                    cv = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > pd.Timedelta(0) else float('inf')
                    consistency = 1.0 / (1.0 + cv.total_seconds() if not np.isinf(cv) else 1.0)
                    return min(1.0, consistency)
            
            # 如果没有时间索引，检查数值的时间连续性
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                main_col = numeric_cols[0]
                series = data[main_col].dropna()
                
                if len(series) > 1:
                    # 计算一阶差分的稳定性
                    diffs = series.diff().dropna()
                    if len(diffs) > 1 and diffs.std() > 0:
                        consistency = 1.0 / (1.0 + abs(diffs.std() / diffs.mean()) if diffs.mean() != 0 else 1.0)
                        return min(1.0, consistency)
            
        except:
            pass
        
        return 0.5  # 默认中等一致性
    
    def validate_environment_configuration(
        self,
        config: EnvironmentConfiguration,
        data: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """验证环境配置"""
        errors = []
        
        # 检查数据兼容性
        if len(data) < config.window_size:
            errors.append(f"Data length {len(data)} < window size {config.window_size}")
        
        # 检查市场类型兼容性
        if not config.time_granularity.is_compatible_with_market(config.market_type.value):
            errors.append(f"Time granularity {config.time_granularity.value} not compatible with market {config.market_type.value}")
        
        # 检查特征数量
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("No numeric columns found in data")
        
        # 检查配置的合理性
        if config.max_steps > len(data):
            errors.append(f"Max steps {config.max_steps} > data length {len(data)}")
        
        return len(errors) == 0, errors
    
    def record_adaptation(self, result: EnvironmentAdaptationResult):
        """记录适配结果"""
        self.adaptation_history.append(result)
        self.adaptation_counter += 1
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """获取适配统计信息"""
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


class AutoEnvironmentAdapter(BaseEnvironmentAdapter):
    """自动环境适配器"""
    
    def __init__(self, config: EnvironmentAdaptationConfig):
        super().__init__(config)
        self.data_characteristics_cache = {}
        self.last_adaptation_step = 0
    
    def should_adapt(
        self,
        current_env: BaseTradingEnvironment,
        data: pd.DataFrame,
        performance_history: List[float],
        current_step: int
    ) -> bool:
        """判断是否需要适配"""
        
        # 检查适配频率
        steps_since_last = current_step - self.last_adaptation_step
        if steps_since_last < self.config.minimum_adaptation_interval:
            return False
        
        if current_step % self.config.adaptation_frequency != 0:
            return False
        
        # 需要足够的性能历史
        if len(performance_history) < 20:
            return False
        
        # 检查性能下降
        if self.config.performance_tracking:
            recent_performance = np.mean(performance_history[-10:])
            baseline_performance = np.mean(performance_history[-30:-10]) if len(performance_history) >= 30 else np.mean(performance_history[:-10])
            
            performance_decline = baseline_performance - recent_performance
            if performance_decline > self.config.performance_threshold:
                return True
        
        # 检查市场状态变化
        if self.config.regime_adaptation:
            try:
                price_col = self._identify_price_column(data)
                if price_col and len(data) >= 50:
                    recent_prices = data[price_col].iloc[-50:].tolist()
                    current_regime = self.regime_detector.detect_regime(recent_prices)
                    
                    # 如果市场状态发生显著变化
                    if len(self.adaptation_history) > 0:
                        last_result = self.adaptation_history[-1]
                        if hasattr(last_result, 'market_regime') and last_result.market_regime != current_regime:
                            return True
            except:
                pass
        
        return False
    
    def adapt_environment(
        self,
        current_env: BaseTradingEnvironment,
        data: pd.DataFrame,
        performance_history: List[float],
        current_step: int
    ) -> EnvironmentAdaptationResult:
        """自动适配环境"""
        
        old_config = current_env.config
        
        # 分析数据特征
        data_characteristics = self.analyze_data_characteristics(data)
        
        # 检测市场状态
        market_context = self._get_market_context(data)
        
        # 分析性能问题
        performance_analysis = self._analyze_performance_issues(performance_history, data_characteristics)
        
        # 生成新配置
        new_config = self._generate_optimized_configuration(
            old_config, data_characteristics, market_context, performance_analysis
        )
        
        # 验证配置
        is_valid, errors = self.validate_environment_configuration(new_config, data)
        if not is_valid:
            self.logger.warning(f"Generated invalid configuration: {errors}")
            new_config = old_config
        
        # 估计性能改善
        performance_improvement = self._estimate_adaptation_improvement(
            old_config, new_config, performance_analysis, market_context
        )
        
        # 计算置信度
        confidence_score = self._calculate_adaptation_confidence(
            data_characteristics, market_context, performance_analysis
        )
        
        # 生成建议和警告
        recommendations = self._generate_recommendations(new_config, data_characteristics, market_context)
        warnings = self._generate_adaptation_warnings(new_config, data_characteristics)
        
        # 创建结果
        result = EnvironmentAdaptationResult(
            old_configuration=old_config,
            new_configuration=new_config,
            adaptation_reason=self._generate_adaptation_reason(performance_analysis, market_context),
            performance_improvement=performance_improvement,
            confidence_score=confidence_score,
            validation_results={"is_valid": is_valid, "errors": errors},
            adaptation_metrics=self._calculate_adaptation_metrics(old_config, new_config),
            recommendations=recommendations,
            warnings=warnings,
            timestamp=datetime.now()
        )
        
        # 记录适配
        self.record_adaptation(result)
        self.last_adaptation_step = current_step
        
        return result
    
    def _identify_price_column(self, data: pd.DataFrame) -> Optional[str]:
        """识别价格列"""
        price_candidates = ['close', 'Close', 'CLOSE', 'price', 'Price', 'PRICE']
        
        for candidate in price_candidates:
            if candidate in data.columns:
                return candidate
        
        # 如果没找到明确的价格列，使用第一个数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else None
    
    def _get_market_context(self, data: pd.DataFrame) -> MarketContext:
        """获取市场上下文"""
        try:
            price_col = self._identify_price_column(data)
            if price_col and len(data) >= 20:
                prices = data[price_col].iloc[-50:].tolist()
                return self.regime_detector.get_market_context(prices)
        except:
            pass
        
        # 默认上下文
        return MarketContext(
            volatility=0.02,
            trend_strength=0.5,
            momentum=0.0,
            volume_profile=1.0,
            correlation_breakdown=0.1,
            regime=MarketRegime.SIDEWAYS
        )
    
    def _analyze_performance_issues(
        self,
        performance_history: List[float],
        data_characteristics: DataCharacteristics
    ) -> Dict[str, Any]:
        """分析性能问题"""
        
        if len(performance_history) < 10:
            return {"trend": "insufficient_data", "issues": []}
        
        performance_array = np.array(performance_history[-30:])
        issues = []
        
        # 趋势分析
        recent_trend = np.mean(performance_array[-10:]) - np.mean(performance_array[-20:-10]) if len(performance_array) >= 20 else 0
        
        if recent_trend < -0.02:
            issues.append("performance_declining")
        
        # 波动性分析
        volatility = np.std(performance_array)
        if volatility > 0.1:
            issues.append("high_volatility")
        elif volatility < 0.01:
            issues.append("low_volatility")
        
        # 数据质量问题
        if data_characteristics.data_quality_score < 0.8:
            issues.append("poor_data_quality")
        
        if data_characteristics.noise_level > 0.5:
            issues.append("high_noise")
        
        if data_characteristics.outlier_ratio > 0.1:
            issues.append("many_outliers")
        
        return {
            "trend": "declining" if recent_trend < -0.01 else "improving" if recent_trend > 0.01 else "stable",
            "volatility": volatility,
            "recent_performance": np.mean(performance_array[-5:]),
            "baseline_performance": np.mean(performance_array[:-5]) if len(performance_array) > 5 else np.mean(performance_array),
            "issues": issues
        }
    
    def _generate_optimized_configuration(
        self,
        old_config: EnvironmentConfiguration,
        data_characteristics: DataCharacteristics,
        market_context: MarketContext,
        performance_analysis: Dict[str, Any]
    ) -> EnvironmentConfiguration:
        """生成优化的配置"""
        
        # 创建新配置（复制旧配置）
        new_config = EnvironmentConfiguration(
            market_type=old_config.market_type,
            time_granularity=old_config.time_granularity,
            risk_profile=old_config.risk_profile,
            environment_features=old_config.environment_features,
            initial_balance=old_config.initial_balance,
            transaction_costs=old_config.transaction_costs,
            window_size=old_config.window_size,
            max_steps=old_config.max_steps,
            normalize_observations=old_config.normalize_observations,
            include_positions=old_config.include_positions,
            include_portfolio_metrics=old_config.include_portfolio_metrics
        )
        
        # 基于数据特征调整窗口大小
        if data_characteristics.noise_level > 0.3:
            # 高噪声时增加窗口大小以平滑数据
            new_config.window_size = min(50, int(old_config.window_size * 1.5))
        elif data_characteristics.noise_level < 0.1:
            # 低噪声时可以减少窗口大小以提高响应性
            new_config.window_size = max(5, int(old_config.window_size * 0.8))
        
        # 基于市场状态调整
        if market_context.regime == MarketRegime.HIGH_VOLATILITY:
            new_config.window_size = max(10, int(new_config.window_size * 1.2))
            new_config.normalize_observations = True
        elif market_context.regime == MarketRegime.LOW_VOLATILITY:
            new_config.window_size = max(5, int(new_config.window_size * 0.9))
        
        # 基于性能问题调整
        if "high_volatility" in performance_analysis["issues"]:
            new_config.normalize_observations = True
            new_config.include_portfolio_metrics = True
        
        if "poor_data_quality" in performance_analysis["issues"]:
            new_config.window_size = min(30, new_config.window_size)  # 减少窗口避免累积错误
        
        # 调整环境特征
        new_features = new_config.environment_features
        
        if market_context.volatility > 0.3:
            new_features |= EnvironmentFeature.HIGH_VOLATILITY
        
        if market_context.regime in [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET]:
            new_features |= EnvironmentFeature.TREND_FOLLOWING
        
        if data_characteristics.sample_size > 1000:
            new_features |= EnvironmentFeature.LARGE_DATASET
        
        new_config.environment_features = new_features
        
        return new_config
    
    def _estimate_adaptation_improvement(
        self,
        old_config: EnvironmentConfiguration,
        new_config: EnvironmentConfiguration,
        performance_analysis: Dict[str, Any],
        market_context: MarketContext
    ) -> float:
        """估计适配改善"""
        
        improvement = 0.0
        
        # 基于配置变化估计改善
        if new_config.window_size != old_config.window_size:
            improvement += 0.01  # 1%基础改善
        
        if new_config.normalize_observations != old_config.normalize_observations:
            improvement += 0.02  # 2%改善
        
        if new_config.environment_features != old_config.environment_features:
            improvement += 0.015  # 1.5%改善
        
        # 基于性能问题调整
        if "performance_declining" in performance_analysis["issues"]:
            improvement *= 2.0  # 下降时改善潜力更大
        
        # 基于市场状态调整
        if market_context.regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
            improvement *= 1.5  # 危机时适配更重要
        
        return min(0.1, improvement)  # 限制最大估计改善为10%
    
    def _calculate_adaptation_confidence(
        self,
        data_characteristics: DataCharacteristics,
        market_context: MarketContext,
        performance_analysis: Dict[str, Any]
    ) -> float:
        """计算适配置信度"""
        
        confidence = 0.6  # 基础置信度
        
        # 基于数据质量
        confidence += data_characteristics.data_quality_score * 0.2
        
        # 基于数据量
        if data_characteristics.sample_size > 500:
            confidence += 0.1
        elif data_characteristics.sample_size < 100:
            confidence -= 0.2
        
        # 基于稳定性
        confidence += data_characteristics.stationarity_score * 0.1
        
        # 基于市场状态确定性
        if market_context.trend_strength > 0.7:
            confidence += 0.15
        elif market_context.trend_strength < 0.3:
            confidence -= 0.1
        
        # 基于性能问题的明确性
        if len(performance_analysis["issues"]) > 0:
            confidence += 0.1  # 问题明确时适配置信度更高
        
        return max(0.1, min(0.95, confidence))
    
    def _generate_recommendations(
        self,
        config: EnvironmentConfiguration,
        data_characteristics: DataCharacteristics,
        market_context: MarketContext
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if data_characteristics.data_quality_score < 0.8:
            recommendations.append("Consider data cleaning and preprocessing")
        
        if data_characteristics.noise_level > 0.3:
            recommendations.append("Apply noise reduction techniques")
        
        if market_context.volatility > 0.3:
            recommendations.append("Use risk management strategies for high volatility")
        
        if data_characteristics.outlier_ratio > 0.1:
            recommendations.append("Implement outlier detection and handling")
        
        if not data_characteristics.seasonality_detected and data_characteristics.sample_size > 200:
            recommendations.append("Consider longer-term patterns and cycles")
        
        return recommendations
    
    def _generate_adaptation_warnings(
        self,
        config: EnvironmentConfiguration,
        data_characteristics: DataCharacteristics
    ) -> List[str]:
        """生成适配警告"""
        warnings = []
        
        if data_characteristics.sample_size < 100:
            warnings.append("Limited data may affect adaptation effectiveness")
        
        if data_characteristics.missing_ratio > 0.2:
            warnings.append("High missing data ratio may impact performance")
        
        if config.window_size > data_characteristics.sample_size // 4:
            warnings.append("Window size may be too large relative to data size")
        
        if data_characteristics.temporal_consistency < 0.5:
            warnings.append("Low temporal consistency may affect time-series modeling")
        
        return warnings
    
    def _generate_adaptation_reason(
        self,
        performance_analysis: Dict[str, Any],
        market_context: MarketContext
    ) -> str:
        """生成适配原因"""
        reasons = []
        
        if performance_analysis["trend"] == "declining":
            reasons.append("performance decline detected")
        
        if "high_volatility" in performance_analysis.get("issues", []):
            reasons.append("high performance volatility")
        
        if market_context.regime == MarketRegime.HIGH_VOLATILITY:
            reasons.append("high market volatility")
        
        if market_context.regime in [MarketRegime.CRISIS, MarketRegime.RECOVERY]:
            reasons.append(f"market regime change to {market_context.regime.value}")
        
        if not reasons:
            reasons.append("routine optimization")
        
        return "; ".join(reasons)
    
    def _calculate_adaptation_metrics(
        self,
        old_config: EnvironmentConfiguration,
        new_config: EnvironmentConfiguration
    ) -> Dict[str, float]:
        """计算适配指标"""
        
        metrics = {}
        
        # 配置变化程度
        changes = 0
        total_params = 0
        
        # 检查各项配置是否变化
        config_params = [
            "window_size", "normalize_observations", "include_positions",
            "include_portfolio_metrics", "environment_features"
        ]
        
        for param in config_params:
            total_params += 1
            if getattr(old_config, param) != getattr(new_config, param):
                changes += 1
        
        metrics["configuration_change_ratio"] = changes / total_params
        
        # 窗口大小变化
        if old_config.window_size != 0:
            metrics["window_size_change_ratio"] = abs(new_config.window_size - old_config.window_size) / old_config.window_size
        else:
            metrics["window_size_change_ratio"] = 0.0
        
        # 特征数量变化
        old_feature_count = bin(old_config.environment_features.value).count('1')
        new_feature_count = bin(new_config.environment_features.value).count('1')
        metrics["feature_count_change"] = new_feature_count - old_feature_count
        
        return metrics


class EnvironmentAdapterFactory:
    """环境适配器工厂"""
    
    @staticmethod
    def create_adapter(
        adaptation_level: AdaptationLevel,
        optimization_goal: EnvironmentOptimizationGoal,
        **kwargs
    ) -> BaseEnvironmentAdapter:
        """创建环境适配器"""
        
        config = EnvironmentAdaptationConfig(
            adaptation_level=adaptation_level,
            optimization_goal=optimization_goal,
            **kwargs
        )
        
        # 目前只实现了自动适配器
        return AutoEnvironmentAdapter(config)


# 便利函数
def create_environment_adapter(
    adaptation_level: str = "intermediate",
    optimization_goal: str = "performance",
    auto_feature_detection: bool = True,
    auto_reward_selection: bool = True,
    auto_parameter_tuning: bool = True,
    adaptation_frequency: int = 100,
    performance_threshold: float = 0.05,
    **kwargs
) -> BaseEnvironmentAdapter:
    """便利函数：创建环境适配器"""
    
    # 转换枚举
    level_enum = AdaptationLevel(adaptation_level)
    goal_enum = EnvironmentOptimizationGoal(optimization_goal)
    
    return EnvironmentAdapterFactory.create_adapter(
        adaptation_level=level_enum,
        optimization_goal=goal_enum,
        auto_feature_detection=auto_feature_detection,
        auto_reward_selection=auto_reward_selection,
        auto_parameter_tuning=auto_parameter_tuning,
        adaptation_frequency=adaptation_frequency,
        performance_threshold=performance_threshold,
        **kwargs
    )


def analyze_environment_requirements(
    data: pd.DataFrame,
    market_type: Union[str, MarketType] = None,
    time_granularity: Union[str, TimeGranularity] = None
) -> Dict[str, Any]:
    """便利函数：分析环境需求"""
    
    adapter = AutoEnvironmentAdapter(
        EnvironmentAdaptationConfig(
            adaptation_level=AdaptationLevel.BASIC,
            optimization_goal=EnvironmentOptimizationGoal.PERFORMANCE
        )
    )
    
    data_characteristics = adapter.analyze_data_characteristics(data)
    market_context = adapter._get_market_context(data)
    
    # 推荐配置
    recommended_config = {
        "window_size": min(30, max(10, len(data) // 20)),
        "normalize_observations": data_characteristics.noise_level > 0.2,
        "include_positions": True,
        "include_portfolio_metrics": data_characteristics.sample_size > 100,
        "environment_features": []
    }
    
    # 基于分析结果推荐特征
    if market_context.volatility > 0.2:
        recommended_config["environment_features"].append("HIGH_VOLATILITY")
    
    if data_characteristics.sample_size > 1000:
        recommended_config["environment_features"].append("LARGE_DATASET")
    
    if market_context.trend_strength > 0.6:
        recommended_config["environment_features"].append("TREND_FOLLOWING")
    
    return {
        "data_characteristics": data_characteristics,
        "market_context": market_context,
        "recommended_configuration": recommended_config,
        "analysis_summary": {
            "data_quality": "good" if data_characteristics.data_quality_score > 0.8 else "fair" if data_characteristics.data_quality_score > 0.6 else "poor",
            "market_regime": market_context.regime.value,
            "complexity_level": "high" if data_characteristics.feature_count > 10 else "medium" if data_characteristics.feature_count > 5 else "low",
            "recommended_adaptation_level": "advanced" if data_characteristics.sample_size > 1000 else "intermediate" if data_characteristics.sample_size > 200 else "basic"
        }
    }