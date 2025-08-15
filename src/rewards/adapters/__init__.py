"""
自适应器模块 - Adapters Module

提供奖励函数参数的动态调整和自适应功能。

主要组件：
- ParameterAdapter: 参数动态调整器基类
- PerformanceBasedAdapter: 基于性能的参数适配器
- MarketRegimeAdapter: 市场状态参数适配器
- MarketRegimeDetector: 市场状态检测器
- PerformanceTracker: 性能追踪器

使用示例：
    ```python
    from src.rewards.adapters import create_parameter_adapter, detect_market_regime
    
    # 创建参数适配器
    adapter = create_parameter_adapter(
        strategy="performance_based",
        adaptation_frequency=10,
        lookback_window=50
    )
    
    # 检测市场状态
    regime = detect_market_regime(price_history)
    
    # 获取市场上下文
    context = get_market_context(price_history, volume_history)
    
    # 适配参数
    if adapter.should_adapt(context, performance_history, current_step):
        result = adapter.adapt_parameters(reward_function, context, performance_history, current_step)
    ```
"""

from typing import Dict, List, Optional, Any

from .parameter_adapter import (
    # 核心类
    BaseParameterAdapter,
    PerformanceBasedAdapter,
    MarketRegimeAdapter,
    PerformanceTracker,
    MarketRegimeDetector,
    ParameterAdapterFactory,
    
    # 枚举
    AdaptationStrategy,
    MarketRegime,
    
    # 数据类
    ParameterBounds,
    AdaptationConfig,
    AdaptationResult,
    MarketContext,
    
    # 便利函数
    create_parameter_adapter,
    detect_market_regime,
    get_market_context
)

# 注意：环境适配器已迁移到 src.environment.adapters 模块
# from .environment_adapter import ...


# 导出主要接口
__all__ = [
    # 参数适配类
    "BaseParameterAdapter",
    "PerformanceBasedAdapter",
    "MarketRegimeAdapter", 
    "PerformanceTracker",
    "MarketRegimeDetector",
    "ParameterAdapterFactory",
    
    # 环境适配类（已迁移到 src.environment.adapters）
    # "BaseEnvironmentAdapter",
    # "AutoEnvironmentAdapter", 
    # "EnvironmentAdapterFactory",
    
    # 枚举
    "AdaptationStrategy",
    "MarketRegime",
    # "AdaptationLevel",  # 已迁移到环境模块
    # "EnvironmentOptimizationGoal",  # 已迁移到环境模块
    
    # 数据类
    "ParameterBounds",
    "AdaptationConfig",
    "AdaptationResult",
    "MarketContext",
    # "DataCharacteristics",  # 已迁移到环境模块
    # "EnvironmentAdaptationConfig",  # 已迁移到环境模块
    # "EnvironmentAdaptationResult",  # 已迁移到环境模块
    
    # 便利函数
    "create_parameter_adapter",
    "detect_market_regime",
    "get_market_context",
    # "create_environment_adapter",  # 已迁移到环境模块
    # "analyze_environment_requirements"  # 已迁移到环境模块
]


def get_available_strategies():
    """获取所有可用的自适应策略"""
    return [strategy.value for strategy in AdaptationStrategy]


def get_available_regimes():
    """获取所有可用的市场状态"""
    return [regime.value for regime in MarketRegime]


def get_strategy_info(strategy: str) -> dict:
    """获取自适应策略的详细信息"""
    
    strategy_info = {
        "performance_based": {
            "name": "Performance-Based Adaptation",
            "description": "Adapts parameters based on recent performance trends",
            "pros": [
                "Directly optimizes for performance improvement",
                "Responsive to strategy effectiveness",
                "Self-correcting mechanism"
            ],
            "cons": [
                "May overfit to recent data",
                "Can be unstable in noisy environments",
                "Requires sufficient performance history"
            ],
            "best_for": "Strategies with clear performance metrics and stable market conditions",
            "parameters": [
                "adaptation_frequency", "lookback_window", 
                "performance_threshold", "adaptation_rate"
            ]
        },
        
        "market_regime": {
            "name": "Market Regime Adaptation",
            "description": "Adapts parameters based on detected market conditions",
            "pros": [
                "Responds to changing market dynamics",
                "Pre-configured optimal parameters for different regimes",
                "More stable than performance-based adaptation"
            ],
            "cons": [
                "Requires accurate regime detection",
                "May lag behind rapid market changes",
                "Limited to predefined regime types"
            ],
            "best_for": "Long-term strategies across different market cycles",
            "parameters": [
                "adaptation_frequency", "market_sensitivity",
                "regime_detection_parameters"
            ]
        },
        
        "volatility_scaling": {
            "name": "Volatility Scaling Adaptation",
            "description": "Scales parameters based on market volatility",
            "pros": [
                "Simple and robust",
                "Good for risk management",
                "Fast adaptation to volatility changes"
            ],
            "cons": [
                "Only considers volatility dimension",
                "May miss other important market features",
                "Not implemented yet"
            ],
            "best_for": "Risk-sensitive strategies and volatile markets",
            "parameters": ["volatility_window", "scaling_factor"]
        },
        
        "reinforcement_learning": {
            "name": "Reinforcement Learning Adaptation", 
            "description": "Uses RL to learn optimal parameter adjustments",
            "pros": [
                "Can learn complex adaptation patterns",
                "Continuous improvement capability",
                "Handles multi-dimensional parameter spaces"
            ],
            "cons": [
                "Complex implementation",
                "Requires extensive training",
                "Not implemented yet"
            ],
            "best_for": "Complex strategies with many parameters",
            "parameters": ["learning_rate", "exploration_rate", "reward_function"]
        }
    }
    
    return strategy_info.get(strategy, {"error": "Unknown adaptation strategy"})


def get_regime_info(regime: str) -> dict:
    """获取市场状态的详细信息"""
    
    regime_info = {
        "bull_market": {
            "name": "Bull Market",
            "description": "Sustained upward price movement",
            "characteristics": [
                "Rising prices over extended period",
                "High investor confidence",
                "Increasing trading volume",
                "Positive economic indicators"
            ],
            "optimal_strategies": ["momentum", "growth", "trend_following"],
            "parameter_adjustments": {
                "lookback_period": "shorter",
                "risk_adjustment": "lower", 
                "smoothing_factor": "lower"
            }
        },
        
        "bear_market": {
            "name": "Bear Market",
            "description": "Sustained downward price movement",
            "characteristics": [
                "Falling prices over extended period",
                "Low investor confidence",
                "Decreasing trading volume",
                "Negative economic indicators"
            ],
            "optimal_strategies": ["defensive", "value", "short_selling"],
            "parameter_adjustments": {
                "lookback_period": "longer",
                "risk_adjustment": "higher",
                "smoothing_factor": "lower"
            }
        },
        
        "sideways": {
            "name": "Sideways Market",
            "description": "Price movement within a range",
            "characteristics": [
                "Limited price movement",
                "Unclear directional trend",
                "Range-bound trading",
                "Market indecision"
            ],
            "optimal_strategies": ["mean_reversion", "range_trading", "contrarian"],
            "parameter_adjustments": {
                "lookback_period": "medium",
                "risk_adjustment": "moderate",
                "smoothing_factor": "higher"
            }
        },
        
        "high_volatility": {
            "name": "High Volatility",
            "description": "Large price swings and uncertainty",
            "characteristics": [
                "Large price movements",
                "High uncertainty",
                "Increased risk",
                "Potential opportunities"
            ],
            "optimal_strategies": ["volatility_trading", "options_strategies"],
            "parameter_adjustments": {
                "lookback_period": "shorter",
                "risk_adjustment": "much_higher",
                "volatility_window": "shorter"
            }
        },
        
        "low_volatility": {
            "name": "Low Volatility",
            "description": "Small price movements and stability",
            "characteristics": [
                "Small price movements",
                "Market stability",
                "Lower risk",
                "Consistent patterns"
            ],
            "optimal_strategies": ["carry_trade", "dividend_strategies"],
            "parameter_adjustments": {
                "lookback_period": "longer",
                "risk_adjustment": "lower",
                "volatility_window": "longer"
            }
        },
        
        "crisis": {
            "name": "Crisis",
            "description": "Extreme market stress and uncertainty",
            "characteristics": [
                "Extreme price volatility",
                "Market panic",
                "Liquidity issues",
                "Correlation breakdown"
            ],
            "optimal_strategies": ["cash", "defensive", "hedging"],
            "parameter_adjustments": {
                "lookback_period": "very_short",
                "risk_adjustment": "maximum",
                "adaptation_frequency": "higher"
            }
        },
        
        "recovery": {
            "name": "Recovery",
            "description": "Market recovering from crisis or downturn",
            "characteristics": [
                "Improving sentiment",
                "Gradual price recovery",
                "Increasing volume",
                "Selective opportunities"
            ],
            "optimal_strategies": ["value", "growth", "momentum"],
            "parameter_adjustments": {
                "lookback_period": "short",
                "risk_adjustment": "moderate",
                "smoothing_factor": "moderate"
            }
        }
    }
    
    return regime_info.get(regime, {"error": "Unknown market regime"})


def create_default_parameter_bounds() -> Dict[str, ParameterBounds]:
    """创建默认参数边界"""
    return {
        "lookback_period": ParameterBounds(
            min_value=5, max_value=100, 
            constraint_type="discrete"
        ),
        "smoothing_factor": ParameterBounds(
            min_value=0.01, max_value=0.5,
            constraint_type="continuous"
        ),
        "risk_adjustment": ParameterBounds(
            min_value=0.1, max_value=3.0,
            constraint_type="continuous"
        ),
        "volatility_window": ParameterBounds(
            min_value=10, max_value=100,
            constraint_type="discrete"
        ),
        "threshold": ParameterBounds(
            min_value=-0.1, max_value=0.1,
            constraint_type="continuous"
        ),
        "alpha": ParameterBounds(
            min_value=0.01, max_value=0.99,
            constraint_type="continuous"
        ),
        "beta": ParameterBounds(
            min_value=0.01, max_value=2.0,
            constraint_type="continuous"
        )
    }


# 模块级配置
VERSION = "1.0.0"
AUTHOR = "Parameter Adapter Team"
DESCRIPTION = "Dynamic parameter adaptation for reward functions"