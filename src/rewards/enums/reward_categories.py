"""
奖励函数分类枚举 - Reward Category Enumeration

定义不同类别的奖励函数，用于组织和选择适合的奖励策略。
"""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class CategoryCharacteristics:
    """奖励类别特征"""
    complexity_level: str  # 复杂度级别: 'low', 'medium', 'high', 'very_high'
    computation_cost: str  # 计算成本: 'low', 'medium', 'high'
    data_requirements: str  # 数据需求: 'minimal', 'moderate', 'extensive'
    suitable_experience: str  # 适合的经验水平: 'beginner', 'intermediate', 'advanced', 'expert'
    risk_awareness: str  # 风险意识: 'basic', 'moderate', 'advanced'
    market_adaptability: str  # 市场适应性: 'fixed', 'adaptive', 'dynamic'
    interpretability: str  # 可解释性: 'high', 'medium', 'low'
    stability: str  # 稳定性: 'stable', 'moderate', 'volatile'


class RewardCategory(Enum):
    """
    奖励函数分类枚举
    
    根据复杂度、使用场景和功能特性对奖励函数进行分类，
    有助于用户根据需求选择合适的奖励策略。
    """
    
    # 基础类别
    BASIC = "basic"
    
    # 风险调整类别
    RISK_ADJUSTED = "risk_adjusted"
    
    # 趋势跟随类别
    TREND_FOLLOWING = "trend_following"
    
    # 均值回归类别
    MEAN_REVERSION = "mean_reversion"
    
    # 动量策略类别
    MOMENTUM = "momentum"
    
    # 波动率感知类别
    VOLATILITY_AWARE = "volatility_aware"
    
    # 多因子类别
    MULTI_FACTOR = "multi_factor"
    
    # 机器学习增强类别
    ML_ENHANCED = "ml_enhanced"
    
    # 深度学习类别
    DEEP_LEARNING = "deep_learning"
    
    # 强化学习专用类别
    RL_SPECIALIZED = "rl_specialized"
    
    # 市场特定类别
    MARKET_SPECIFIC = "market_specific"
    
    # 实验性类别
    EXPERIMENTAL = "experimental"
    
    @property
    def characteristics(self) -> CategoryCharacteristics:
        """获取分类特征"""
        return _CATEGORY_CHARACTERISTICS[self]
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        return _DISPLAY_NAMES[self]
    
    @property
    def description(self) -> str:
        """获取详细描述"""
        return _DESCRIPTIONS[self]
    
    @property
    def typical_functions(self) -> List[str]:
        """获取典型奖励函数"""
        return _TYPICAL_FUNCTIONS[self]
    
    @property
    def compatible_markets(self) -> Set[str]:
        """获取兼容的市场类型"""
        return _COMPATIBLE_MARKETS[self]
    
    @property
    def suitable_granularities(self) -> Set[str]:
        """获取适合的时间粒度"""
        return _SUITABLE_GRANULARITIES[self]
    
    @classmethod
    def from_string(cls, category_str: str) -> 'RewardCategory':
        """从字符串创建RewardCategory"""
        category_str = category_str.lower().strip()
        
        # 直接匹配
        for category in cls:
            if category.value == category_str:
                return category
        
        # 别名匹配
        aliases = {
            'simple': cls.BASIC,
            'basic_return': cls.BASIC,
            'elementary': cls.BASIC,
            
            'risk': cls.RISK_ADJUSTED,
            'sharpe': cls.RISK_ADJUSTED,
            'risk_aware': cls.RISK_ADJUSTED,
            
            'trend': cls.TREND_FOLLOWING,
            'trending': cls.TREND_FOLLOWING,
            'directional': cls.TREND_FOLLOWING,
            
            'mean_rev': cls.MEAN_REVERSION,
            'reversion': cls.MEAN_REVERSION,
            'contrarian': cls.MEAN_REVERSION,
            
            'mom': cls.MOMENTUM,
            'momentum_based': cls.MOMENTUM,
            'breakout': cls.MOMENTUM,
            
            'vol': cls.VOLATILITY_AWARE,
            'volatility': cls.VOLATILITY_AWARE,
            'vol_aware': cls.VOLATILITY_AWARE,
            
            'multi': cls.MULTI_FACTOR,
            'factor': cls.MULTI_FACTOR,
            'composite': cls.MULTI_FACTOR,
            
            'ml': cls.ML_ENHANCED,
            'machine_learning': cls.ML_ENHANCED,
            'ai': cls.ML_ENHANCED,
            
            'dl': cls.DEEP_LEARNING,
            'deep': cls.DEEP_LEARNING,
            'neural': cls.DEEP_LEARNING,
            
            'rl': cls.RL_SPECIALIZED,
            'reinforcement': cls.RL_SPECIALIZED,
            'agent': cls.RL_SPECIALIZED,
            
            'forex': cls.MARKET_SPECIFIC,
            'stock': cls.MARKET_SPECIFIC,
            'crypto': cls.MARKET_SPECIFIC,
            
            'exp': cls.EXPERIMENTAL,
            'research': cls.EXPERIMENTAL,
            'novel': cls.EXPERIMENTAL,
        }
        
        if category_str in aliases:
            return aliases[category_str]
        
        raise ValueError(f"Unknown reward category: {category_str}")
    
    def is_suitable_for_market(self, market_type: str) -> bool:
        """检查是否适合指定市场类型"""
        return market_type in self.compatible_markets
    
    def is_suitable_for_granularity(self, granularity: str) -> bool:
        """检查是否适合指定时间粒度"""
        return granularity in self.suitable_granularities
    
    def get_complexity_score(self) -> int:
        """获取复杂度评分 (1-10)"""
        complexity_scores = {
            'low': 2,
            'medium': 5,
            'high': 8,
            'very_high': 10
        }
        return complexity_scores[self.characteristics.complexity_level]
    
    def get_recommended_for_experience(self, experience_level: str) -> bool:
        """检查是否推荐给指定经验水平"""
        experience_mapping = {
            'beginner': ['beginner'],
            'intermediate': ['beginner', 'intermediate'],
            'advanced': ['intermediate', 'advanced'],
            'expert': ['advanced', 'expert']
        }
        
        suitable_levels = experience_mapping.get(experience_level, [])
        return self.characteristics.suitable_experience in suitable_levels


# 分类特征定义
_CATEGORY_CHARACTERISTICS = {
    RewardCategory.BASIC: CategoryCharacteristics(
        complexity_level='low',
        computation_cost='low',
        data_requirements='minimal',
        suitable_experience='beginner',
        risk_awareness='basic',
        market_adaptability='fixed',
        interpretability='high',
        stability='stable'
    ),
    
    RewardCategory.RISK_ADJUSTED: CategoryCharacteristics(
        complexity_level='medium',
        computation_cost='medium',
        data_requirements='moderate',
        suitable_experience='intermediate',
        risk_awareness='advanced',
        market_adaptability='adaptive',
        interpretability='high',
        stability='stable'
    ),
    
    RewardCategory.TREND_FOLLOWING: CategoryCharacteristics(
        complexity_level='medium',
        computation_cost='medium',
        data_requirements='moderate',
        suitable_experience='intermediate',
        risk_awareness='moderate',
        market_adaptability='adaptive',
        interpretability='medium',
        stability='moderate'
    ),
    
    RewardCategory.MEAN_REVERSION: CategoryCharacteristics(
        complexity_level='medium',
        computation_cost='medium',
        data_requirements='moderate',
        suitable_experience='intermediate',
        risk_awareness='moderate',
        market_adaptability='adaptive',
        interpretability='medium',
        stability='moderate'
    ),
    
    RewardCategory.MOMENTUM: CategoryCharacteristics(
        complexity_level='medium',
        computation_cost='medium',
        data_requirements='moderate',
        suitable_experience='intermediate',
        risk_awareness='moderate',
        market_adaptability='dynamic',
        interpretability='medium',
        stability='volatile'
    ),
    
    RewardCategory.VOLATILITY_AWARE: CategoryCharacteristics(
        complexity_level='high',
        computation_cost='high',
        data_requirements='extensive',
        suitable_experience='advanced',
        risk_awareness='advanced',
        market_adaptability='dynamic',
        interpretability='medium',
        stability='moderate'
    ),
    
    RewardCategory.MULTI_FACTOR: CategoryCharacteristics(
        complexity_level='high',
        computation_cost='high',
        data_requirements='extensive',
        suitable_experience='advanced',
        risk_awareness='advanced',
        market_adaptability='dynamic',
        interpretability='low',
        stability='moderate'
    ),
    
    RewardCategory.ML_ENHANCED: CategoryCharacteristics(
        complexity_level='high',
        computation_cost='high',
        data_requirements='extensive',
        suitable_experience='advanced',
        risk_awareness='advanced',
        market_adaptability='dynamic',
        interpretability='low',
        stability='moderate'
    ),
    
    RewardCategory.DEEP_LEARNING: CategoryCharacteristics(
        complexity_level='very_high',
        computation_cost='high',
        data_requirements='extensive',
        suitable_experience='expert',
        risk_awareness='advanced',
        market_adaptability='dynamic',
        interpretability='low',
        stability='volatile'
    ),
    
    RewardCategory.RL_SPECIALIZED: CategoryCharacteristics(
        complexity_level='very_high',
        computation_cost='high',
        data_requirements='extensive',
        suitable_experience='expert',
        risk_awareness='advanced',
        market_adaptability='dynamic',
        interpretability='low',
        stability='volatile'
    ),
    
    RewardCategory.MARKET_SPECIFIC: CategoryCharacteristics(
        complexity_level='high',
        computation_cost='medium',
        data_requirements='moderate',
        suitable_experience='advanced',
        risk_awareness='advanced',
        market_adaptability='fixed',
        interpretability='medium',
        stability='stable'
    ),
    
    RewardCategory.EXPERIMENTAL: CategoryCharacteristics(
        complexity_level='very_high',
        computation_cost='high',
        data_requirements='extensive',
        suitable_experience='expert',
        risk_awareness='advanced',
        market_adaptability='dynamic',
        interpretability='low',
        stability='volatile'
    )
}

# 显示名称
_DISPLAY_NAMES = {
    RewardCategory.BASIC: "基础奖励",
    RewardCategory.RISK_ADJUSTED: "风险调整",
    RewardCategory.TREND_FOLLOWING: "趋势跟随", 
    RewardCategory.MEAN_REVERSION: "均值回归",
    RewardCategory.MOMENTUM: "动量策略",
    RewardCategory.VOLATILITY_AWARE: "波动率感知",
    RewardCategory.MULTI_FACTOR: "多因子模型",
    RewardCategory.ML_ENHANCED: "机器学习增强",
    RewardCategory.DEEP_LEARNING: "深度学习",
    RewardCategory.RL_SPECIALIZED: "强化学习专用",
    RewardCategory.MARKET_SPECIFIC: "市场特定",
    RewardCategory.EXPERIMENTAL: "实验性"
}

# 详细描述
_DESCRIPTIONS = {
    RewardCategory.BASIC: "基础的收益计算，简单直观，适合初学者理解和使用",
    RewardCategory.RISK_ADJUSTED: "考虑风险因素的奖励，平衡收益和风险，提供更稳定的策略",
    RewardCategory.TREND_FOLLOWING: "识别和跟随市场趋势，在趋势市场中表现优异",
    RewardCategory.MEAN_REVERSION: "基于价格回归均值的理论，在震荡市场中表现良好",
    RewardCategory.MOMENTUM: "捕捉价格动量，适合突破和趋势初期的交易",
    RewardCategory.VOLATILITY_AWARE: "感知市场波动率变化，动态调整策略参数",
    RewardCategory.MULTI_FACTOR: "结合多个市场因子，提供更全面的市场分析",
    RewardCategory.ML_ENHANCED: "利用机器学习技术增强传统策略，自适应市场变化",
    RewardCategory.DEEP_LEARNING: "基于深度神经网络，能够发现复杂的非线性模式",
    RewardCategory.RL_SPECIALIZED: "专为强化学习设计，优化agent的学习过程",
    RewardCategory.MARKET_SPECIFIC: "针对特定市场类型优化，在相应市场中表现最佳",
    RewardCategory.EXPERIMENTAL: "前沿研究成果，具有创新性但可能不够稳定"
}

# 典型奖励函数
_TYPICAL_FUNCTIONS = {
    RewardCategory.BASIC: [
        "simple_return", "profit_loss", "percentage_change", "absolute_return"
    ],
    RewardCategory.RISK_ADJUSTED: [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "risk_adjusted_return"
    ],
    RewardCategory.TREND_FOLLOWING: [
        "trend_strength", "directional_change", "trend_consistency", "breakout_reward"
    ],
    RewardCategory.MEAN_REVERSION: [
        "reversion_strength", "overbought_oversold", "bollinger_reversion", "rsi_reversion"
    ],
    RewardCategory.MOMENTUM: [
        "momentum_strength", "breakout_momentum", "volume_momentum", "price_momentum"
    ],
    RewardCategory.VOLATILITY_AWARE: [
        "volatility_adjusted", "garch_based", "realized_volatility", "implied_volatility"
    ],
    RewardCategory.MULTI_FACTOR: [
        "factor_weighted", "pca_based", "composite_score", "regime_dependent"
    ],
    RewardCategory.ML_ENHANCED: [
        "ensemble_reward", "feature_based", "pattern_recognition", "anomaly_detection"
    ],
    RewardCategory.DEEP_LEARNING: [
        "lstm_reward", "cnn_reward", "transformer_reward", "autoencoder_reward"
    ],
    RewardCategory.RL_SPECIALIZED: [
        "curiosity_driven", "intrinsic_motivation", "hierarchical_reward", "meta_reward"
    ],
    RewardCategory.MARKET_SPECIFIC: [
        "forex_optimized", "crypto_volatility", "stock_fundamental", "commodity_seasonal"
    ],
    RewardCategory.EXPERIMENTAL: [
        "quantum_inspired", "graph_neural", "causal_inference", "adversarial_robust"
    ]
}

# 兼容的市场类型
_COMPATIBLE_MARKETS = {
    RewardCategory.BASIC: {"forex", "stock", "crypto", "commodity", "bond", "option", "index"},
    RewardCategory.RISK_ADJUSTED: {"forex", "stock", "crypto", "commodity", "bond", "index"},
    RewardCategory.TREND_FOLLOWING: {"forex", "stock", "crypto", "commodity", "index"},
    RewardCategory.MEAN_REVERSION: {"forex", "stock", "crypto", "commodity"},
    RewardCategory.MOMENTUM: {"stock", "crypto", "commodity", "index"},
    RewardCategory.VOLATILITY_AWARE: {"forex", "stock", "crypto", "option"},
    RewardCategory.MULTI_FACTOR: {"stock", "index", "bond"},
    RewardCategory.ML_ENHANCED: {"forex", "stock", "crypto", "commodity", "index"},
    RewardCategory.DEEP_LEARNING: {"forex", "stock", "crypto"},
    RewardCategory.RL_SPECIALIZED: {"forex", "stock", "crypto", "commodity"},
    RewardCategory.MARKET_SPECIFIC: {"forex", "stock", "crypto", "commodity", "bond", "option"},
    RewardCategory.EXPERIMENTAL: {"forex", "stock", "crypto"}
}

# 适合的时间粒度
_SUITABLE_GRANULARITIES = {
    RewardCategory.BASIC: {"1min", "5min", "15min", "1h", "1d", "1w"},
    RewardCategory.RISK_ADJUSTED: {"5min", "15min", "1h", "1d", "1w"},
    RewardCategory.TREND_FOLLOWING: {"15min", "1h", "4h", "1d", "1w"},
    RewardCategory.MEAN_REVERSION: {"1min", "5min", "15min", "1h"},
    RewardCategory.MOMENTUM: {"5min", "15min", "1h", "1d"},
    RewardCategory.VOLATILITY_AWARE: {"1min", "5min", "15min", "1h", "1d"},
    RewardCategory.MULTI_FACTOR: {"1h", "1d", "1w"},
    RewardCategory.ML_ENHANCED: {"5min", "15min", "1h", "1d"},
    RewardCategory.DEEP_LEARNING: {"1min", "5min", "15min", "1h"},
    RewardCategory.RL_SPECIALIZED: {"1min", "5min", "15min", "1h", "1d"},
    RewardCategory.MARKET_SPECIFIC: {"1min", "5min", "15min", "1h", "1d"},
    RewardCategory.EXPERIMENTAL: {"1min", "5min", "15min", "1h"}
}