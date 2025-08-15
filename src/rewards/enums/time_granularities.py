"""
时间粒度枚举 - Time Granularity Enumeration

定义不同的时间粒度，每种粒度适合不同的交易策略和奖励函数。
"""

from enum import Enum
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import datetime


@dataclass
class GranularityCharacteristics:
    """时间粒度特征"""
    duration_seconds: int  # 持续时间（秒）
    data_points_per_day: int  # 每日数据点数量
    noise_level: str  # 噪声水平: 'low', 'medium', 'high'
    trend_reliability: str  # 趋势可靠性: 'low', 'medium', 'high'
    strategy_type: str  # 适合的策略类型
    minimum_position_duration: str  # 最小持仓时间建议
    risk_level: str  # 风险级别
    data_requirements: str  # 数据需求
    computational_complexity: str  # 计算复杂度


class TimeGranularity(Enum):
    """
    时间粒度枚举
    
    定义了不同的时间粒度，每种粒度都有其特定的特征，
    影响奖励函数的选择、参数配置和策略适用性。
    """
    
    # 超短期粒度
    TICK = "tick"  # 逐笔数据
    SECOND_1 = "1s"  # 1秒
    SECOND_5 = "5s"  # 5秒
    SECOND_15 = "15s"  # 15秒
    SECOND_30 = "30s"  # 30秒
    
    # 短期粒度
    MINUTE_1 = "1min"  # 1分钟
    MINUTE_5 = "5min"  # 5分钟
    MINUTE_15 = "15min"  # 15分钟
    MINUTE_30 = "30min"  # 30分钟
    
    # 中期粒度
    HOUR_1 = "1h"  # 1小时
    HOUR_4 = "4h"  # 4小时
    HOUR_8 = "8h"  # 8小时
    HOUR_12 = "12h"  # 12小时
    
    # 长期粒度
    DAY_1 = "1d"  # 1天
    DAY_3 = "3d"  # 3天
    WEEK_1 = "1w"  # 1周
    WEEK_2 = "2w"  # 2周
    MONTH_1 = "1M"  # 1月
    MONTH_3 = "3M"  # 3月
    
    @property
    def characteristics(self) -> GranularityCharacteristics:
        """获取时间粒度特征"""
        return _GRANULARITY_CHARACTERISTICS[self]
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        return _DISPLAY_NAMES[self]
    
    @property
    def category(self) -> str:
        """获取分类"""
        return _CATEGORIES[self]
    
    @property
    def compatible_markets(self) -> Set[str]:
        """获取兼容的市场类型"""
        return _COMPATIBLE_MARKETS[self]
    
    @property
    def recommended_rewards(self) -> List[str]:
        """获取推荐的奖励函数"""
        return _RECOMMENDED_REWARDS[self]
    
    @classmethod
    def from_string(cls, granularity_str: str) -> 'TimeGranularity':
        """从字符串创建TimeGranularity"""
        granularity_str = granularity_str.lower().strip()
        
        # 直接匹配
        for granularity in cls:
            if granularity.value == granularity_str:
                return granularity
        
        # 别名匹配
        aliases = {
            # 分钟别名
            '1m': cls.MINUTE_1,
            '5m': cls.MINUTE_5,
            '15m': cls.MINUTE_15,
            '30m': cls.MINUTE_30,
            'min1': cls.MINUTE_1,
            'min5': cls.MINUTE_5,
            'min15': cls.MINUTE_15,
            'min30': cls.MINUTE_30,
            
            # 小时别名
            '1hour': cls.HOUR_1,
            '4hour': cls.HOUR_4,
            '1hr': cls.HOUR_1,
            '4hr': cls.HOUR_4,
            'h1': cls.HOUR_1,
            'h4': cls.HOUR_4,
            
            # 日别名
            '1day': cls.DAY_1,
            'daily': cls.DAY_1,
            'day': cls.DAY_1,
            'd1': cls.DAY_1,
            
            # 周别名
            '1week': cls.WEEK_1,
            'weekly': cls.WEEK_1,
            'week': cls.WEEK_1,
            'w1': cls.WEEK_1,
            
            # 月别名
            '1month': cls.MONTH_1,
            'monthly': cls.MONTH_1,
            'month': cls.MONTH_1,
            'm1': cls.MONTH_1,
            
            # 秒别名
            '1second': cls.SECOND_1,
            '1sec': cls.SECOND_1,
            's1': cls.SECOND_1,
        }
        
        if granularity_str in aliases:
            return aliases[granularity_str]
        
        raise ValueError(f"Unknown time granularity: {granularity_str}")
    
    def get_duration_timedelta(self) -> datetime.timedelta:
        """获取时间间隔"""
        return datetime.timedelta(seconds=self.characteristics.duration_seconds)
    
    def is_compatible_with_market(self, market_type: str) -> bool:
        """检查是否与指定市场类型兼容"""
        return market_type in self.compatible_markets
    
    def get_data_points_for_period(self, days: int) -> int:
        """获取指定天数内的数据点数量"""
        return self.characteristics.data_points_per_day * days
    
    def is_high_frequency(self) -> bool:
        """是否为高频数据"""
        return self.characteristics.duration_seconds < 300  # 小于5分钟
    
    def is_intraday(self) -> bool:
        """是否为日内数据"""
        return self.characteristics.duration_seconds < 86400  # 小于1天
    
    def get_suitable_history_length(self) -> int:
        """获取适合的历史数据长度（数据点数）"""
        if self.is_high_frequency():
            return 200  # 高频数据用较少历史
        elif self.is_intraday():
            return 100  # 日内数据用中等历史
        else:
            return 50   # 日级以上数据用较少历史


# 时间粒度特征定义
_GRANULARITY_CHARACTERISTICS = {
    TimeGranularity.TICK: GranularityCharacteristics(
        duration_seconds=0,  # 不定时间
        data_points_per_day=100000,  # 估算
        noise_level='very_high',
        trend_reliability='very_low',
        strategy_type='market_making',
        minimum_position_duration='seconds',
        risk_level='very_high',
        data_requirements='massive',
        computational_complexity='very_high'
    ),
    
    TimeGranularity.SECOND_1: GranularityCharacteristics(
        duration_seconds=1,
        data_points_per_day=86400,
        noise_level='very_high',
        trend_reliability='very_low',
        strategy_type='high_frequency',
        minimum_position_duration='seconds',
        risk_level='very_high',
        data_requirements='high',
        computational_complexity='very_high'
    ),
    
    TimeGranularity.SECOND_5: GranularityCharacteristics(
        duration_seconds=5,
        data_points_per_day=17280,
        noise_level='high',
        trend_reliability='low',
        strategy_type='scalping',
        minimum_position_duration='seconds',
        risk_level='high',
        data_requirements='high',
        computational_complexity='high'
    ),
    
    TimeGranularity.SECOND_15: GranularityCharacteristics(
        duration_seconds=15,
        data_points_per_day=5760,
        noise_level='high',
        trend_reliability='low',
        strategy_type='scalping',
        minimum_position_duration='minutes',
        risk_level='high',
        data_requirements='medium',
        computational_complexity='high'
    ),
    
    TimeGranularity.SECOND_30: GranularityCharacteristics(
        duration_seconds=30,
        data_points_per_day=2880,
        noise_level='medium_high',
        trend_reliability='low',
        strategy_type='scalping',
        minimum_position_duration='minutes',
        risk_level='high',
        data_requirements='medium',
        computational_complexity='medium_high'
    ),
    
    TimeGranularity.MINUTE_1: GranularityCharacteristics(
        duration_seconds=60,
        data_points_per_day=1440,
        noise_level='medium_high',
        trend_reliability='low_medium',
        strategy_type='scalping',
        minimum_position_duration='minutes',
        risk_level='medium_high',
        data_requirements='medium',
        computational_complexity='medium_high'
    ),
    
    TimeGranularity.MINUTE_5: GranularityCharacteristics(
        duration_seconds=300,
        data_points_per_day=288,
        noise_level='medium',
        trend_reliability='medium',
        strategy_type='day_trading',
        minimum_position_duration='hours',
        risk_level='medium',
        data_requirements='medium',
        computational_complexity='medium'
    ),
    
    TimeGranularity.MINUTE_15: GranularityCharacteristics(
        duration_seconds=900,
        data_points_per_day=96,
        noise_level='medium',
        trend_reliability='medium',
        strategy_type='day_trading',
        minimum_position_duration='hours',
        risk_level='medium',
        data_requirements='medium',
        computational_complexity='medium'
    ),
    
    TimeGranularity.MINUTE_30: GranularityCharacteristics(
        duration_seconds=1800,
        data_points_per_day=48,
        noise_level='medium_low',
        trend_reliability='medium',
        strategy_type='swing_trading',
        minimum_position_duration='hours',
        risk_level='medium',
        data_requirements='low_medium',
        computational_complexity='medium'
    ),
    
    TimeGranularity.HOUR_1: GranularityCharacteristics(
        duration_seconds=3600,
        data_points_per_day=24,
        noise_level='medium_low',
        trend_reliability='medium_high',
        strategy_type='swing_trading',
        minimum_position_duration='days',
        risk_level='medium_low',
        data_requirements='low_medium',
        computational_complexity='low_medium'
    ),
    
    TimeGranularity.HOUR_4: GranularityCharacteristics(
        duration_seconds=14400,
        data_points_per_day=6,
        noise_level='low',
        trend_reliability='high',
        strategy_type='swing_trading',
        minimum_position_duration='days',
        risk_level='medium_low',
        data_requirements='low',
        computational_complexity='low_medium'
    ),
    
    TimeGranularity.HOUR_8: GranularityCharacteristics(
        duration_seconds=28800,
        data_points_per_day=3,
        noise_level='low',
        trend_reliability='high',
        strategy_type='position_trading',
        minimum_position_duration='days',
        risk_level='low_medium',
        data_requirements='low',
        computational_complexity='low'
    ),
    
    TimeGranularity.HOUR_12: GranularityCharacteristics(
        duration_seconds=43200,
        data_points_per_day=2,
        noise_level='low',
        trend_reliability='high',
        strategy_type='position_trading',
        minimum_position_duration='weeks',
        risk_level='low_medium',
        data_requirements='low',
        computational_complexity='low'
    ),
    
    TimeGranularity.DAY_1: GranularityCharacteristics(
        duration_seconds=86400,
        data_points_per_day=1,
        noise_level='low',
        trend_reliability='high',
        strategy_type='position_trading',
        minimum_position_duration='weeks',
        risk_level='low',
        data_requirements='low',
        computational_complexity='low'
    ),
    
    TimeGranularity.DAY_3: GranularityCharacteristics(
        duration_seconds=259200,
        data_points_per_day=1/3,
        noise_level='very_low',
        trend_reliability='very_high',
        strategy_type='long_term_investing',
        minimum_position_duration='months',
        risk_level='low',
        data_requirements='very_low',
        computational_complexity='very_low'
    ),
    
    TimeGranularity.WEEK_1: GranularityCharacteristics(
        duration_seconds=604800,
        data_points_per_day=1/7,
        noise_level='very_low',
        trend_reliability='very_high',
        strategy_type='long_term_investing',
        minimum_position_duration='months',
        risk_level='low',
        data_requirements='very_low',
        computational_complexity='very_low'
    ),
    
    TimeGranularity.WEEK_2: GranularityCharacteristics(
        duration_seconds=1209600,
        data_points_per_day=1/14,
        noise_level='very_low',
        trend_reliability='very_high',
        strategy_type='long_term_investing',
        minimum_position_duration='quarters',
        risk_level='very_low',
        data_requirements='very_low',
        computational_complexity='very_low'
    ),
    
    TimeGranularity.MONTH_1: GranularityCharacteristics(
        duration_seconds=2592000,  # 30天近似
        data_points_per_day=1/30,
        noise_level='very_low',
        trend_reliability='very_high',
        strategy_type='strategic_investing',
        minimum_position_duration='years',
        risk_level='very_low',
        data_requirements='very_low',
        computational_complexity='very_low'
    ),
    
    TimeGranularity.MONTH_3: GranularityCharacteristics(
        duration_seconds=7776000,  # 90天近似
        data_points_per_day=1/90,
        noise_level='very_low',
        trend_reliability='very_high',
        strategy_type='strategic_investing',
        minimum_position_duration='years',
        risk_level='very_low',
        data_requirements='very_low',
        computational_complexity='very_low'
    )
}

# 显示名称
_DISPLAY_NAMES = {
    TimeGranularity.TICK: "逐笔",
    TimeGranularity.SECOND_1: "1秒",
    TimeGranularity.SECOND_5: "5秒",
    TimeGranularity.SECOND_15: "15秒", 
    TimeGranularity.SECOND_30: "30秒",
    TimeGranularity.MINUTE_1: "1分钟",
    TimeGranularity.MINUTE_5: "5分钟",
    TimeGranularity.MINUTE_15: "15分钟",
    TimeGranularity.MINUTE_30: "30分钟",
    TimeGranularity.HOUR_1: "1小时",
    TimeGranularity.HOUR_4: "4小时",
    TimeGranularity.HOUR_8: "8小时",
    TimeGranularity.HOUR_12: "12小时",
    TimeGranularity.DAY_1: "日线",
    TimeGranularity.DAY_3: "3日线",
    TimeGranularity.WEEK_1: "周线",
    TimeGranularity.WEEK_2: "双周线",
    TimeGranularity.MONTH_1: "月线",
    TimeGranularity.MONTH_3: "季线"
}

# 分类
_CATEGORIES = {
    TimeGranularity.TICK: "ultra_high_frequency",
    TimeGranularity.SECOND_1: "ultra_high_frequency",
    TimeGranularity.SECOND_5: "ultra_high_frequency",
    TimeGranularity.SECOND_15: "high_frequency",
    TimeGranularity.SECOND_30: "high_frequency",
    TimeGranularity.MINUTE_1: "high_frequency",
    TimeGranularity.MINUTE_5: "short_term",
    TimeGranularity.MINUTE_15: "short_term",
    TimeGranularity.MINUTE_30: "short_term",
    TimeGranularity.HOUR_1: "medium_term",
    TimeGranularity.HOUR_4: "medium_term",
    TimeGranularity.HOUR_8: "medium_term",
    TimeGranularity.HOUR_12: "medium_term",
    TimeGranularity.DAY_1: "long_term",
    TimeGranularity.DAY_3: "long_term",
    TimeGranularity.WEEK_1: "very_long_term",
    TimeGranularity.WEEK_2: "very_long_term",
    TimeGranularity.MONTH_1: "strategic",
    TimeGranularity.MONTH_3: "strategic"
}

# 兼容的市场类型
_COMPATIBLE_MARKETS = {
    TimeGranularity.TICK: {"forex", "stock", "crypto"},
    TimeGranularity.SECOND_1: {"forex", "crypto"},
    TimeGranularity.SECOND_5: {"forex", "crypto"},
    TimeGranularity.SECOND_15: {"forex", "crypto"},
    TimeGranularity.SECOND_30: {"forex", "crypto"},
    TimeGranularity.MINUTE_1: {"forex", "stock", "crypto", "commodity"},
    TimeGranularity.MINUTE_5: {"forex", "stock", "crypto", "commodity", "index"},
    TimeGranularity.MINUTE_15: {"forex", "stock", "crypto", "commodity", "index"},
    TimeGranularity.MINUTE_30: {"forex", "stock", "crypto", "commodity", "index"},
    TimeGranularity.HOUR_1: {"forex", "stock", "crypto", "commodity", "bond", "index"},
    TimeGranularity.HOUR_4: {"forex", "crypto", "commodity"},
    TimeGranularity.HOUR_8: {"forex", "crypto"},
    TimeGranularity.HOUR_12: {"forex", "crypto"},
    TimeGranularity.DAY_1: {"forex", "stock", "crypto", "commodity", "bond", "option", "index"},
    TimeGranularity.DAY_3: {"stock", "commodity", "bond", "index"},
    TimeGranularity.WEEK_1: {"stock", "commodity", "bond", "index"},
    TimeGranularity.WEEK_2: {"stock", "bond", "index"},
    TimeGranularity.MONTH_1: {"stock", "bond", "index"},
    TimeGranularity.MONTH_3: {"stock", "bond", "index"}
}

# 推荐的奖励函数
_RECOMMENDED_REWARDS = {
    TimeGranularity.TICK: ["market_making", "arbitrage", "ultra_short_momentum"],
    TimeGranularity.SECOND_1: ["scalping", "spread_capture", "microstructure"],
    TimeGranularity.SECOND_5: ["scalping", "mean_reversion", "volume_weighted"],
    TimeGranularity.SECOND_15: ["scalping", "momentum", "volatility_breakout"],
    TimeGranularity.SECOND_30: ["scalping", "trend_following", "support_resistance"],
    TimeGranularity.MINUTE_1: ["scalping", "trend_following", "breakout"],
    TimeGranularity.MINUTE_5: ["day_trading", "momentum", "mean_reversion"],
    TimeGranularity.MINUTE_15: ["day_trading", "trend_following", "pattern_recognition"],
    TimeGranularity.MINUTE_30: ["swing_start", "trend_continuation", "reversal_pattern"],
    TimeGranularity.HOUR_1: ["swing_trading", "trend_following", "indicator_based"],
    TimeGranularity.HOUR_4: ["position_start", "macro_trend", "cycle_analysis"],
    TimeGranularity.HOUR_8: ["position_trading", "fundamental_trend", "seasonal"],
    TimeGranularity.HOUR_12: ["position_trading", "macro_economic", "long_trend"],
    TimeGranularity.DAY_1: ["position_trading", "fundamental_analysis", "value_investing"],
    TimeGranularity.DAY_3: ["long_term_trend", "economic_cycle", "sector_rotation"],
    TimeGranularity.WEEK_1: ["strategic_allocation", "macro_trend", "business_cycle"],
    TimeGranularity.WEEK_2: ["portfolio_rebalancing", "asset_allocation", "risk_parity"],
    TimeGranularity.MONTH_1: ["strategic_investing", "fundamental_value", "dividend_growth"],
    TimeGranularity.MONTH_3: ["asset_allocation", "macro_strategy", "long_term_value"]
}