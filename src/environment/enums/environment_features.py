"""
环境特征枚举 - Environment Feature Enumeration

定义交易环境的各种特征，用于选择和配置适合的奖励函数。
"""

from enum import Enum, Flag, auto
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class FeatureConfiguration:
    """特征配置"""
    importance_weight: float  # 重要性权重 (0.0-1.0)
    computational_cost: str  # 计算成本: 'low', 'medium', 'high'
    data_dependency: str  # 数据依赖: 'none', 'basic', 'extensive'
    update_frequency: str  # 更新频率: 'tick', 'bar', 'session', 'daily'
    stability: str  # 稳定性: 'stable', 'moderate', 'volatile'


class EnvironmentFeature(Flag):
    """
    环境特征标志枚举
    
    使用Flag允许组合多个特征，更好地描述复杂的交易环境。
    每个特征代表环境的一个方面，影响奖励函数的选择和配置。
    """
    
    # 基础市场特征
    HIGH_LIQUIDITY = auto()  # 高流动性
    LOW_SPREAD = auto()  # 低点差
    HIGH_VOLATILITY = auto()  # 高波动性
    TRENDING_MARKET = auto()  # 趋势市场
    RANGE_BOUND = auto()  # 区间震荡
    
    # 时间特征
    REAL_TIME_DATA = auto()  # 实时数据
    HIGH_FREQUENCY = auto()  # 高频交易
    INTRADAY_TRADING = auto()  # 日内交易
    MULTI_SESSION = auto()  # 多时段交易
    OVERNIGHT_RISK = auto()  # 隔夜风险
    
    # 交易特征
    LEVERAGE_AVAILABLE = auto()  # 可用杠杆
    SHORT_SELLING = auto()  # 可卖空
    ALGORITHMIC_TRADING = auto()  # 算法交易
    MARKET_MAKING = auto()  # 做市商模式
    POSITION_LIMITS = auto()  # 仓位限制
    
    # 成本特征
    TRANSACTION_COSTS = auto()  # 交易成本
    SLIPPAGE_EFFECTS = auto()  # 滑点影响
    FUNDING_COSTS = auto()  # 资金成本
    TAX_IMPLICATIONS = auto()  # 税务影响
    
    # 风险特征
    COUNTER_PARTY_RISK = auto()  # 对手方风险
    OPERATIONAL_RISK = auto()  # 操作风险
    REGULATORY_RISK = auto()  # 监管风险
    CURRENCY_RISK = auto()  # 汇率风险
    TAIL_RISK = auto()  # 尾部风险
    
    # 信息特征
    NEWS_DRIVEN = auto()  # 新闻驱动
    EARNINGS_SENSITIVE = auto()  # 财报敏感
    MACRO_SENSITIVE = auto()  # 宏观敏感
    TECHNICAL_DRIVEN = auto()  # 技术分析驱动
    SENTIMENT_DRIVEN = auto()  # 情绪驱动
    
    # 高级特征
    CROSS_ASSET_CORRELATION = auto()  # 跨资产相关性
    REGIME_SWITCHING = auto()  # 制度转换
    SEASONAL_PATTERNS = auto()  # 季节性模式
    MICROSTRUCTURE_EFFECTS = auto()  # 市场微观结构
    NETWORK_EFFECTS = auto()  # 网络效应
    
    @property
    def configuration(self) -> FeatureConfiguration:
        """获取特征配置"""
        return _FEATURE_CONFIGURATIONS.get(self, _DEFAULT_CONFIGURATION)
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        return _DISPLAY_NAMES.get(self, self.name.title())
    
    @property
    def description(self) -> str:
        """获取特征描述"""
        return _DESCRIPTIONS.get(self, "环境特征")
    
    @classmethod
    def get_common_combinations(cls) -> Dict[str, 'EnvironmentFeature']:
        """获取常见的特征组合"""
        return {
            "forex_scalping": (
                cls.HIGH_LIQUIDITY | cls.LOW_SPREAD | cls.HIGH_FREQUENCY |
                cls.REAL_TIME_DATA | cls.LEVERAGE_AVAILABLE | cls.TRANSACTION_COSTS
            ),
            
            "stock_day_trading": (
                cls.HIGH_LIQUIDITY | cls.INTRADAY_TRADING | cls.ALGORITHMIC_TRADING |
                cls.TRANSACTION_COSTS | cls.SLIPPAGE_EFFECTS | cls.NEWS_DRIVEN
            ),
            
            "crypto_swing": (
                cls.HIGH_VOLATILITY | cls.MULTI_SESSION | cls.SENTIMENT_DRIVEN |
                cls.OVERNIGHT_RISK | cls.REGULATORY_RISK
            ),
            
            "institutional_portfolio": (
                cls.POSITION_LIMITS | cls.TRANSACTION_COSTS | cls.TAX_IMPLICATIONS |
                cls.CROSS_ASSET_CORRELATION | cls.REGIME_SWITCHING
            ),
            
            "market_making": (
                cls.HIGH_FREQUENCY | cls.MARKET_MAKING | cls.MICROSTRUCTURE_EFFECTS |
                cls.REAL_TIME_DATA | cls.SLIPPAGE_EFFECTS
            ),
            
            "trend_following": (
                cls.TRENDING_MARKET | cls.TECHNICAL_DRIVEN | cls.CROSS_ASSET_CORRELATION |
                cls.OVERNIGHT_RISK | cls.LEVERAGE_AVAILABLE
            ),
            
            "mean_reversion": (
                cls.RANGE_BOUND | cls.HIGH_FREQUENCY | cls.MICROSTRUCTURE_EFFECTS |
                cls.TECHNICAL_DRIVEN | cls.TRANSACTION_COSTS
            )
        }
    
    def get_compatible_rewards(self) -> List[str]:
        """获取兼容的奖励函数类型"""
        compatible = []
        
        # 基于特征组合推荐奖励函数
        if self & EnvironmentFeature.HIGH_FREQUENCY:
            compatible.extend(["scalping", "market_making", "tick_based"])
        
        if self & EnvironmentFeature.TRENDING_MARKET:
            compatible.extend(["trend_following", "momentum", "breakout"])
        
        if self & EnvironmentFeature.RANGE_BOUND:
            compatible.extend(["mean_reversion", "oscillator_based", "support_resistance"])
        
        if self & EnvironmentFeature.HIGH_VOLATILITY:
            compatible.extend(["volatility_adjusted", "risk_parity", "adaptive_sizing"])
        
        if self & EnvironmentFeature.LEVERAGE_AVAILABLE:
            compatible.extend(["leverage_aware", "margin_optimized", "capital_efficient"])
        
        if self & EnvironmentFeature.NEWS_DRIVEN:
            compatible.extend(["event_driven", "sentiment_based", "news_momentum"])
        
        return list(set(compatible))  # 去重
    
    def calculate_complexity_score(self) -> float:
        """计算环境复杂度评分"""
        total_weight = 0.0
        active_features = []
        
        # 遍历所有单独的特征
        for feature in EnvironmentFeature:
            if feature & self:  # 如果该特征被激活
                active_features.append(feature)
                total_weight += feature.configuration.importance_weight
        
        # 基于激活特征数量和权重计算复杂度
        if not active_features:
            return 0.0
        
        feature_count_factor = min(len(active_features) / 10.0, 1.0)  # 标准化到0-1
        weight_factor = min(total_weight / 5.0, 1.0)  # 假设5为最大合理权重
        
        return (feature_count_factor + weight_factor) / 2.0
    
    def get_recommended_granularities(self) -> Set[str]:
        """获取推荐的时间粒度"""
        granularities = set()
        
        if self & EnvironmentFeature.HIGH_FREQUENCY:
            granularities.update(["1s", "5s", "15s", "30s", "1min"])
        
        if self & EnvironmentFeature.INTRADAY_TRADING:
            granularities.update(["1min", "5min", "15min", "30min", "1h"])
        
        if self & EnvironmentFeature.OVERNIGHT_RISK:
            granularities.update(["1h", "4h", "1d"])
        
        if self & EnvironmentFeature.SEASONAL_PATTERNS:
            granularities.update(["1d", "1w", "1M"])
        
        # 如果没有特定要求，返回中等频率
        if not granularities:
            granularities.update(["5min", "15min", "1h", "1d"])
        
        return granularities


# 默认配置
_DEFAULT_CONFIGURATION = FeatureConfiguration(
    importance_weight=0.5,
    computational_cost='medium',
    data_dependency='basic',
    update_frequency='bar',
    stability='moderate'
)

# 特征配置定义
_FEATURE_CONFIGURATIONS = {
    EnvironmentFeature.HIGH_LIQUIDITY: FeatureConfiguration(0.8, 'low', 'basic', 'tick', 'stable'),
    EnvironmentFeature.LOW_SPREAD: FeatureConfiguration(0.7, 'low', 'basic', 'tick', 'stable'),
    EnvironmentFeature.HIGH_VOLATILITY: FeatureConfiguration(0.9, 'medium', 'basic', 'tick', 'volatile'),
    EnvironmentFeature.TRENDING_MARKET: FeatureConfiguration(0.8, 'medium', 'extensive', 'bar', 'moderate'),
    EnvironmentFeature.RANGE_BOUND: FeatureConfiguration(0.7, 'medium', 'extensive', 'bar', 'stable'),
    
    EnvironmentFeature.REAL_TIME_DATA: FeatureConfiguration(0.9, 'high', 'extensive', 'tick', 'volatile'),
    EnvironmentFeature.HIGH_FREQUENCY: FeatureConfiguration(0.9, 'high', 'extensive', 'tick', 'volatile'),
    EnvironmentFeature.INTRADAY_TRADING: FeatureConfiguration(0.6, 'medium', 'basic', 'bar', 'moderate'),
    EnvironmentFeature.MULTI_SESSION: FeatureConfiguration(0.5, 'low', 'basic', 'session', 'stable'),
    EnvironmentFeature.OVERNIGHT_RISK: FeatureConfiguration(0.7, 'low', 'basic', 'daily', 'volatile'),
    
    EnvironmentFeature.LEVERAGE_AVAILABLE: FeatureConfiguration(0.8, 'low', 'none', 'bar', 'stable'),
    EnvironmentFeature.SHORT_SELLING: FeatureConfiguration(0.6, 'low', 'none', 'bar', 'stable'),
    EnvironmentFeature.ALGORITHMIC_TRADING: FeatureConfiguration(0.7, 'high', 'extensive', 'tick', 'moderate'),
    EnvironmentFeature.MARKET_MAKING: FeatureConfiguration(0.9, 'high', 'extensive', 'tick', 'volatile'),
    EnvironmentFeature.POSITION_LIMITS: FeatureConfiguration(0.5, 'low', 'basic', 'daily', 'stable'),
    
    EnvironmentFeature.TRANSACTION_COSTS: FeatureConfiguration(0.8, 'medium', 'basic', 'bar', 'stable'),
    EnvironmentFeature.SLIPPAGE_EFFECTS: FeatureConfiguration(0.7, 'medium', 'extensive', 'tick', 'volatile'),
    EnvironmentFeature.FUNDING_COSTS: FeatureConfiguration(0.5, 'low', 'basic', 'daily', 'stable'),
    EnvironmentFeature.TAX_IMPLICATIONS: FeatureConfiguration(0.3, 'low', 'basic', 'daily', 'stable'),
    
    EnvironmentFeature.COUNTER_PARTY_RISK: FeatureConfiguration(0.6, 'low', 'basic', 'daily', 'moderate'),
    EnvironmentFeature.OPERATIONAL_RISK: FeatureConfiguration(0.4, 'low', 'basic', 'daily', 'stable'),
    EnvironmentFeature.REGULATORY_RISK: FeatureConfiguration(0.5, 'low', 'basic', 'daily', 'moderate'),
    EnvironmentFeature.CURRENCY_RISK: FeatureConfiguration(0.6, 'medium', 'basic', 'bar', 'volatile'),
    EnvironmentFeature.TAIL_RISK: FeatureConfiguration(0.8, 'high', 'extensive', 'bar', 'volatile'),
    
    EnvironmentFeature.NEWS_DRIVEN: FeatureConfiguration(0.7, 'medium', 'extensive', 'bar', 'volatile'),
    EnvironmentFeature.EARNINGS_SENSITIVE: FeatureConfiguration(0.6, 'low', 'basic', 'daily', 'volatile'),
    EnvironmentFeature.MACRO_SENSITIVE: FeatureConfiguration(0.7, 'medium', 'basic', 'daily', 'moderate'),
    EnvironmentFeature.TECHNICAL_DRIVEN: FeatureConfiguration(0.8, 'medium', 'extensive', 'bar', 'moderate'),
    EnvironmentFeature.SENTIMENT_DRIVEN: FeatureConfiguration(0.6, 'high', 'extensive', 'bar', 'volatile'),
    
    EnvironmentFeature.CROSS_ASSET_CORRELATION: FeatureConfiguration(0.8, 'high', 'extensive', 'bar', 'moderate'),
    EnvironmentFeature.REGIME_SWITCHING: FeatureConfiguration(0.9, 'high', 'extensive', 'daily', 'volatile'),
    EnvironmentFeature.SEASONAL_PATTERNS: FeatureConfiguration(0.5, 'medium', 'extensive', 'daily', 'stable'),
    EnvironmentFeature.MICROSTRUCTURE_EFFECTS: FeatureConfiguration(0.9, 'high', 'extensive', 'tick', 'volatile'),
    EnvironmentFeature.NETWORK_EFFECTS: FeatureConfiguration(0.7, 'high', 'extensive', 'bar', 'moderate'),
}

# 显示名称
_DISPLAY_NAMES = {
    EnvironmentFeature.HIGH_LIQUIDITY: "高流动性",
    EnvironmentFeature.LOW_SPREAD: "低点差",
    EnvironmentFeature.HIGH_VOLATILITY: "高波动性",
    EnvironmentFeature.TRENDING_MARKET: "趋势市场",
    EnvironmentFeature.RANGE_BOUND: "区间震荡",
    
    EnvironmentFeature.REAL_TIME_DATA: "实时数据",
    EnvironmentFeature.HIGH_FREQUENCY: "高频交易",
    EnvironmentFeature.INTRADAY_TRADING: "日内交易",
    EnvironmentFeature.MULTI_SESSION: "多时段交易",
    EnvironmentFeature.OVERNIGHT_RISK: "隔夜风险",
    
    EnvironmentFeature.LEVERAGE_AVAILABLE: "可用杠杆",
    EnvironmentFeature.SHORT_SELLING: "可卖空",
    EnvironmentFeature.ALGORITHMIC_TRADING: "算法交易",
    EnvironmentFeature.MARKET_MAKING: "做市商模式",
    EnvironmentFeature.POSITION_LIMITS: "仓位限制",
    
    EnvironmentFeature.TRANSACTION_COSTS: "交易成本",
    EnvironmentFeature.SLIPPAGE_EFFECTS: "滑点影响",
    EnvironmentFeature.FUNDING_COSTS: "资金成本",
    EnvironmentFeature.TAX_IMPLICATIONS: "税务影响",
    
    EnvironmentFeature.COUNTER_PARTY_RISK: "对手方风险",
    EnvironmentFeature.OPERATIONAL_RISK: "操作风险",
    EnvironmentFeature.REGULATORY_RISK: "监管风险",
    EnvironmentFeature.CURRENCY_RISK: "汇率风险",
    EnvironmentFeature.TAIL_RISK: "尾部风险",
    
    EnvironmentFeature.NEWS_DRIVEN: "新闻驱动",
    EnvironmentFeature.EARNINGS_SENSITIVE: "财报敏感",
    EnvironmentFeature.MACRO_SENSITIVE: "宏观敏感",
    EnvironmentFeature.TECHNICAL_DRIVEN: "技术分析驱动",
    EnvironmentFeature.SENTIMENT_DRIVEN: "情绪驱动",
    
    EnvironmentFeature.CROSS_ASSET_CORRELATION: "跨资产相关性",
    EnvironmentFeature.REGIME_SWITCHING: "制度转换",
    EnvironmentFeature.SEASONAL_PATTERNS: "季节性模式",
    EnvironmentFeature.MICROSTRUCTURE_EFFECTS: "市场微观结构",
    EnvironmentFeature.NETWORK_EFFECTS: "网络效应",
}

# 特征描述
_DESCRIPTIONS = {
    EnvironmentFeature.HIGH_LIQUIDITY: "市场具有高流动性，交易执行容易，价格冲击小",
    EnvironmentFeature.LOW_SPREAD: "买卖价差小，交易成本低",
    EnvironmentFeature.HIGH_VOLATILITY: "价格波动剧烈，风险和机会并存",
    EnvironmentFeature.TRENDING_MARKET: "市场具有明显趋势，适合趋势跟随策略",
    EnvironmentFeature.RANGE_BOUND: "价格在区间内震荡，适合均值回归策略",
    
    EnvironmentFeature.REAL_TIME_DATA: "实时数据流，延迟极低",
    EnvironmentFeature.HIGH_FREQUENCY: "高频交易环境，需要快速响应",
    EnvironmentFeature.INTRADAY_TRADING: "日内交易，不持有隔夜仓位",
    EnvironmentFeature.MULTI_SESSION: "跨越多个交易时段",
    EnvironmentFeature.OVERNIGHT_RISK: "存在隔夜风险，需要风险管理",
    
    EnvironmentFeature.LEVERAGE_AVAILABLE: "可以使用杠杆放大仓位",
    EnvironmentFeature.SHORT_SELLING: "允许卖空操作",
    EnvironmentFeature.ALGORITHMIC_TRADING: "算法交易环境，需要优化执行",
    EnvironmentFeature.MARKET_MAKING: "做市商模式，提供流动性获取价差",
    EnvironmentFeature.POSITION_LIMITS: "存在仓位限制，需要风险控制",
    
    EnvironmentFeature.TRANSACTION_COSTS: "存在交易成本，影响策略收益",
    EnvironmentFeature.SLIPPAGE_EFFECTS: "存在滑点，影响实际执行价格",
    EnvironmentFeature.FUNDING_COSTS: "存在资金成本，影响持仓收益",
    EnvironmentFeature.TAX_IMPLICATIONS: "需要考虑税务影响",
    
    EnvironmentFeature.NEWS_DRIVEN: "价格受新闻事件驱动",
    EnvironmentFeature.TECHNICAL_DRIVEN: "适合技术分析方法",
    EnvironmentFeature.SENTIMENT_DRIVEN: "受市场情绪影响较大",
}