"""
市场类型枚举 - Market Type Enumeration

定义不同的交易市场类型，每种市场类型具有不同的特征和适合的奖励函数。
"""

from enum import Enum, auto
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class MarketCharacteristics:
    """市场特征"""
    volatility_range: tuple  # 波动率范围 (低, 高)
    liquidity_level: str  # 流动性水平: 'low', 'medium', 'high'
    trading_hours: str  # 交易时间: '24/7', '5days', 'custom'
    typical_spreads: tuple  # 典型点差范围
    leverage_available: tuple  # 可用杠杆范围
    minimum_position_size: float  # 最小仓位大小
    currency_base: str  # 基础货币
    regulatory_environment: str  # 监管环境
    correlation_factors: List[str]  # 相关因子


class MarketType(Enum):
    """
    市场类型枚举
    
    定义了不同的交易市场类型，每种类型都有其独特的特征，
    这些特征影响奖励函数的选择和参数配置。
    """
    
    # 外汇市场
    FOREX = "forex"
    
    # 股票市场
    STOCK = "stock"
    
    # 加密货币市场
    CRYPTO = "crypto"
    
    # 商品期货市场
    COMMODITY = "commodity"
    
    # 债券市场
    BOND = "bond"
    
    # 期权市场
    OPTION = "option"
    
    # 指数市场
    INDEX = "index"
    
    @property
    def characteristics(self) -> MarketCharacteristics:
        """获取市场特征"""
        return _MARKET_CHARACTERISTICS[self]
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        return _DISPLAY_NAMES[self]
    
    @property
    def typical_instruments(self) -> List[str]:
        """获取典型交易工具"""
        return _TYPICAL_INSTRUMENTS[self]
    
    @property
    def compatible_granularities(self) -> Set[str]:
        """获取兼容的时间粒度"""
        return _COMPATIBLE_GRANULARITIES[self]
    
    @property
    def recommended_rewards(self) -> List[str]:
        """获取推荐的奖励函数"""
        return _RECOMMENDED_REWARDS[self]
    
    @classmethod
    def from_string(cls, market_str: str) -> 'MarketType':
        """从字符串创建MarketType"""
        market_str = market_str.lower().strip()
        
        # 直接匹配
        for market_type in cls:
            if market_type.value == market_str:
                return market_type
        
        # 别名匹配
        aliases = {
            'fx': cls.FOREX,
            'currency': cls.FOREX,
            'currencies': cls.FOREX,
            'foreign_exchange': cls.FOREX,
            
            'stocks': cls.STOCK,
            'equity': cls.STOCK,
            'equities': cls.STOCK,
            'shares': cls.STOCK,
            
            'cryptocurrency': cls.CRYPTO,
            'cryptocurrencies': cls.CRYPTO,
            'digital': cls.CRYPTO,
            'bitcoin': cls.CRYPTO,
            
            'commodities': cls.COMMODITY,
            'futures': cls.COMMODITY,
            'gold': cls.COMMODITY,
            'oil': cls.COMMODITY,
            
            'bonds': cls.BOND,
            'treasury': cls.BOND,
            'fixed_income': cls.BOND,
            
            'options': cls.OPTION,
            'derivatives': cls.OPTION,
            
            'indices': cls.INDEX,
            'indexes': cls.INDEX,
            'etf': cls.INDEX,
        }
        
        if market_str in aliases:
            return aliases[market_str]
        
        raise ValueError(f"Unknown market type: {market_str}")
    
    def is_compatible_with_granularity(self, granularity: str) -> bool:
        """检查是否与指定时间粒度兼容"""
        return granularity in self.compatible_granularities
    
    def get_risk_level(self) -> str:
        """获取风险级别"""
        risk_levels = {
            MarketType.FOREX: 'medium',
            MarketType.STOCK: 'medium', 
            MarketType.CRYPTO: 'high',
            MarketType.COMMODITY: 'high',
            MarketType.BOND: 'low',
            MarketType.OPTION: 'very_high',
            MarketType.INDEX: 'medium'
        }
        return risk_levels[self]


# 市场特征定义
_MARKET_CHARACTERISTICS = {
    MarketType.FOREX: MarketCharacteristics(
        volatility_range=(0.5, 3.0),  # 日波动率百分比
        liquidity_level='high',
        trading_hours='24/5',  # 24小时，周一到周五
        typical_spreads=(0.1, 5.0),  # 点差范围（点）
        leverage_available=(1, 500),
        minimum_position_size=0.01,  # 0.01手
        currency_base='various',
        regulatory_environment='global',
        correlation_factors=['interest_rates', 'economic_indicators', 'geopolitical_events']
    ),
    
    MarketType.STOCK: MarketCharacteristics(
        volatility_range=(1.0, 5.0),
        liquidity_level='high',
        trading_hours='market_hours',  # 交易所开放时间
        typical_spreads=(0.01, 0.1),  # 相对价格的百分比
        leverage_available=(1, 4),
        minimum_position_size=1.0,  # 1股
        currency_base='local',
        regulatory_environment='national',
        correlation_factors=['sector_performance', 'market_sentiment', 'earnings', 'economic_data']
    ),
    
    MarketType.CRYPTO: MarketCharacteristics(
        volatility_range=(3.0, 20.0),
        liquidity_level='medium',
        trading_hours='24/7',
        typical_spreads=(0.05, 1.0),
        leverage_available=(1, 100),
        minimum_position_size=0.0001,  # 很小的最小单位
        currency_base='crypto',
        regulatory_environment='evolving',
        correlation_factors=['tech_adoption', 'regulatory_news', 'market_sentiment', 'bitcoin_dominance']
    ),
    
    MarketType.COMMODITY: MarketCharacteristics(
        volatility_range=(2.0, 8.0),
        liquidity_level='medium',
        trading_hours='market_hours',
        typical_spreads=(0.1, 2.0),
        leverage_available=(1, 20),
        minimum_position_size=1.0,  # 1合约
        currency_base='usd',
        regulatory_environment='national',
        correlation_factors=['supply_demand', 'weather', 'geopolitical_events', 'currency_strength']
    ),
    
    MarketType.BOND: MarketCharacteristics(
        volatility_range=(0.2, 2.0),
        liquidity_level='high',
        trading_hours='market_hours',
        typical_spreads=(0.01, 0.25),
        leverage_available=(1, 10),
        minimum_position_size=1000.0,  # 面值
        currency_base='local',
        regulatory_environment='strict',
        correlation_factors=['interest_rates', 'inflation', 'credit_risk', 'economic_growth']
    ),
    
    MarketType.OPTION: MarketCharacteristics(
        volatility_range=(5.0, 50.0),
        liquidity_level='medium',
        trading_hours='market_hours',
        typical_spreads=(0.05, 2.0),
        leverage_available=(1, 10),
        minimum_position_size=1.0,  # 1合约
        currency_base='local',
        regulatory_environment='strict',
        correlation_factors=['underlying_volatility', 'time_decay', 'interest_rates', 'market_sentiment']
    ),
    
    MarketType.INDEX: MarketCharacteristics(
        volatility_range=(0.8, 4.0),
        liquidity_level='high',
        trading_hours='market_hours',
        typical_spreads=(0.01, 0.2),
        leverage_available=(1, 5),
        minimum_position_size=0.1,  # 0.1单位
        currency_base='local',
        regulatory_environment='national',
        correlation_factors=['constituent_performance', 'market_breadth', 'economic_indicators', 'sector_rotation']
    )
}

# 显示名称
_DISPLAY_NAMES = {
    MarketType.FOREX: "外汇市场",
    MarketType.STOCK: "股票市场", 
    MarketType.CRYPTO: "加密货币市场",
    MarketType.COMMODITY: "商品期货市场",
    MarketType.BOND: "债券市场",
    MarketType.OPTION: "期权市场",
    MarketType.INDEX: "指数市场"
}

# 典型交易工具
_TYPICAL_INSTRUMENTS = {
    MarketType.FOREX: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"],
    MarketType.STOCK: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
    MarketType.CRYPTO: ["BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "BNBUSD", "SOLUSD", "XRPUSD"],
    MarketType.COMMODITY: ["GOLD", "SILVER", "CRUDE_OIL", "NATURAL_GAS", "WHEAT", "CORN", "COPPER"],
    MarketType.BOND: ["US10Y", "US30Y", "DE10Y", "UK10Y", "JP10Y"],
    MarketType.OPTION: ["SPY_OPTIONS", "QQQ_OPTIONS", "VIX_OPTIONS"],
    MarketType.INDEX: ["SPX", "NDX", "DJI", "FTSE", "DAX", "NIKKEI", "HSI"]
}

# 兼容的时间粒度
_COMPATIBLE_GRANULARITIES = {
    MarketType.FOREX: {"1min", "5min", "15min", "1h", "4h", "1d"},
    MarketType.STOCK: {"1min", "5min", "15min", "1h", "1d", "1w"},
    MarketType.CRYPTO: {"1min", "5min", "15min", "1h", "4h", "1d", "1w"},
    MarketType.COMMODITY: {"5min", "15min", "1h", "1d", "1w"},
    MarketType.BOND: {"1h", "1d", "1w", "1M"},
    MarketType.OPTION: {"1min", "5min", "15min", "1h", "1d"},
    MarketType.INDEX: {"1min", "5min", "15min", "1h", "1d", "1w"}
}

# 推荐的奖励函数
_RECOMMENDED_REWARDS = {
    MarketType.FOREX: ["forex_optimized", "pip_based", "risk_adjusted", "trend_following"],
    MarketType.STOCK: ["simple_return", "risk_adjusted", "dividend_adjusted", "sector_relative"],
    MarketType.CRYPTO: ["volatility_adjusted", "momentum_based", "fear_greed_index", "hash_rate_weighted"],
    MarketType.COMMODITY: ["supply_demand_based", "seasonal_adjusted", "storage_cost_adjusted"],
    MarketType.BOND: ["duration_adjusted", "yield_curve_based", "credit_spread_weighted"],
    MarketType.OPTION: ["gamma_hedged", "theta_decay_aware", "vega_adjusted", "volatility_smile_based"],
    MarketType.INDEX: ["market_cap_weighted", "equal_weighted", "factor_tilted", "momentum_based"]
}