"""
风险配置枚举 - Risk Profile Enumeration

定义不同的风险配置级别，用于选择适合风险偏好的奖励函数。
"""

from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RiskConfiguration:
    """风险配置"""
    max_drawdown: float  # 最大回撤限制 (百分比)
    position_sizing: str  # 仓位大小策略: 'conservative', 'moderate', 'aggressive'
    stop_loss_required: bool  # 是否需要止损
    max_leverage: float  # 最大杠杆倍数
    diversification_requirement: str  # 分散化需求: 'high', 'medium', 'low'
    volatility_tolerance: str  # 波动率容忍度: 'low', 'medium', 'high'
    tail_risk_protection: bool  # 是否需要尾部风险保护
    correlation_limit: float  # 相关性限制
    var_limit: float  # VaR限制 (百分比)
    stress_test_requirement: bool  # 是否需要压力测试


class RiskProfile(Enum):
    """
    风险配置枚举
    
    定义不同的风险偏好级别，每个级别都有相应的风险控制参数，
    影响奖励函数的选择和风险管理策略的配置。
    """
    
    # 极保守型
    ULTRA_CONSERVATIVE = "ultra_conservative"
    
    # 保守型
    CONSERVATIVE = "conservative"
    
    # 稳健型
    MODERATE = "moderate"
    
    # 平衡型
    BALANCED = "balanced"
    
    # 成长型
    GROWTH = "growth"
    
    # 积极型
    AGGRESSIVE = "aggressive"
    
    # 极积极型
    ULTRA_AGGRESSIVE = "ultra_aggressive"
    
    # 风险中性
    RISK_NEUTRAL = "risk_neutral"
    
    @property
    def configuration(self) -> RiskConfiguration:
        """获取风险配置"""
        return _RISK_CONFIGURATIONS[self]
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        return _DISPLAY_NAMES[self]
    
    @property
    def description(self) -> str:
        """获取详细描述"""
        return _DESCRIPTIONS[self]
    
    @property
    def suitable_markets(self) -> List[str]:
        """获取适合的市场类型"""
        return _SUITABLE_MARKETS[self]
    
    @property
    def recommended_rewards(self) -> List[str]:
        """获取推荐的奖励函数"""
        return _RECOMMENDED_REWARDS[self]
    
    @property
    def risk_metrics(self) -> Dict[str, float]:
        """获取风险指标"""
        return _RISK_METRICS[self]
    
    @classmethod
    def from_string(cls, profile_str: str) -> 'RiskProfile':
        """从字符串创建RiskProfile"""
        profile_str = profile_str.lower().strip()
        
        # 直接匹配
        for profile in cls:
            if profile.value == profile_str:
                return profile
        
        # 别名匹配
        aliases = {
            'ultra_safe': cls.ULTRA_CONSERVATIVE,
            'very_safe': cls.ULTRA_CONSERVATIVE,
            'minimum_risk': cls.ULTRA_CONSERVATIVE,
            
            'safe': cls.CONSERVATIVE,
            'low_risk': cls.CONSERVATIVE,
            'capital_preservation': cls.CONSERVATIVE,
            
            'moderate_risk': cls.MODERATE,
            'medium_risk': cls.MODERATE,
            'steady': cls.MODERATE,
            
            'balanced_risk': cls.BALANCED,
            'medium': cls.BALANCED,
            'diversified': cls.BALANCED,
            
            'growth_oriented': cls.GROWTH,
            'higher_return': cls.GROWTH,
            'equity_like': cls.GROWTH,
            
            'high_risk': cls.AGGRESSIVE,
            'aggressive_growth': cls.AGGRESSIVE,
            'maximum_return': cls.AGGRESSIVE,
            
            'ultra_high': cls.ULTRA_AGGRESSIVE,
            'maximum_risk': cls.ULTRA_AGGRESSIVE,
            'speculative': cls.ULTRA_AGGRESSIVE,
            
            'neutral': cls.RISK_NEUTRAL,
            'market_neutral': cls.RISK_NEUTRAL,
            'benchmark': cls.RISK_NEUTRAL,
        }
        
        if profile_str in aliases:
            return aliases[profile_str]
        
        raise ValueError(f"Unknown risk profile: {profile_str}")
    
    @classmethod
    def from_metrics(cls, sharpe_ratio: float, max_drawdown: float, 
                    volatility: float) -> 'RiskProfile':
        """从历史风险指标推断风险配置"""
        # 基于历史表现推断风险偏好
        risk_score = 0
        
        # Sharpe比率评分
        if sharpe_ratio > 2.0:
            risk_score += 0
        elif sharpe_ratio > 1.0:
            risk_score += 1
        elif sharpe_ratio > 0.5:
            risk_score += 2
        else:
            risk_score += 3
        
        # 最大回撤评分
        if max_drawdown < 0.05:  # 5%以下
            risk_score += 0
        elif max_drawdown < 0.10:  # 5-10%
            risk_score += 1
        elif max_drawdown < 0.20:  # 10-20%
            risk_score += 2
        else:  # 20%以上
            risk_score += 3
        
        # 波动率评分
        if volatility < 0.10:  # 10%以下
            risk_score += 0
        elif volatility < 0.20:  # 10-20%
            risk_score += 1
        elif volatility < 0.30:  # 20-30%
            risk_score += 2
        else:  # 30%以上
            risk_score += 3
        
        # 根据总分映射到风险配置
        if risk_score <= 2:
            return cls.ULTRA_CONSERVATIVE
        elif risk_score <= 4:
            return cls.CONSERVATIVE
        elif risk_score <= 6:
            return cls.MODERATE
        elif risk_score <= 8:
            return cls.BALANCED
        elif risk_score <= 10:
            return cls.GROWTH
        elif risk_score <= 12:
            return cls.AGGRESSIVE
        else:
            return cls.ULTRA_AGGRESSIVE
    
    def is_compatible_with_market(self, market_type: str) -> bool:
        """检查是否与市场类型兼容"""
        return market_type in self.suitable_markets
    
    def get_position_size_multiplier(self) -> float:
        """获取仓位大小乘数"""
        multipliers = {
            'conservative': 0.5,
            'moderate': 1.0,
            'aggressive': 2.0
        }
        return multipliers[self.configuration.position_sizing]
    
    def validate_strategy_risk(self, expected_return: float, expected_volatility: float) -> bool:
        """验证策略风险是否符合配置"""
        config = self.configuration
        
        # 检查波动率
        if config.volatility_tolerance == 'low' and expected_volatility > 0.15:
            return False
        elif config.volatility_tolerance == 'medium' and expected_volatility > 0.30:
            return False
        elif config.volatility_tolerance == 'high' and expected_volatility > 0.50:
            return False
        
        # 检查预期夏普比率
        if expected_volatility > 0:
            expected_sharpe = expected_return / expected_volatility
            min_sharpe = self.risk_metrics.get('min_sharpe_ratio', 0.0)
            if expected_sharpe < min_sharpe:
                return False
        
        return True


# 风险配置定义
_RISK_CONFIGURATIONS = {
    RiskProfile.ULTRA_CONSERVATIVE: RiskConfiguration(
        max_drawdown=0.02,  # 2%
        position_sizing='conservative',
        stop_loss_required=True,
        max_leverage=1.0,
        diversification_requirement='high',
        volatility_tolerance='low',
        tail_risk_protection=True,
        correlation_limit=0.3,
        var_limit=0.01,  # 1%
        stress_test_requirement=True
    ),
    
    RiskProfile.CONSERVATIVE: RiskConfiguration(
        max_drawdown=0.05,  # 5%
        position_sizing='conservative',
        stop_loss_required=True,
        max_leverage=1.5,
        diversification_requirement='high',
        volatility_tolerance='low',
        tail_risk_protection=True,
        correlation_limit=0.4,
        var_limit=0.02,  # 2%
        stress_test_requirement=True
    ),
    
    RiskProfile.MODERATE: RiskConfiguration(
        max_drawdown=0.08,  # 8%
        position_sizing='moderate',
        stop_loss_required=True,
        max_leverage=2.0,
        diversification_requirement='medium',
        volatility_tolerance='medium',
        tail_risk_protection=True,
        correlation_limit=0.5,
        var_limit=0.03,  # 3%
        stress_test_requirement=True
    ),
    
    RiskProfile.BALANCED: RiskConfiguration(
        max_drawdown=0.12,  # 12%
        position_sizing='moderate',
        stop_loss_required=False,
        max_leverage=3.0,
        diversification_requirement='medium',
        volatility_tolerance='medium',
        tail_risk_protection=False,
        correlation_limit=0.6,
        var_limit=0.05,  # 5%
        stress_test_requirement=False
    ),
    
    RiskProfile.GROWTH: RiskConfiguration(
        max_drawdown=0.18,  # 18%
        position_sizing='moderate',
        stop_loss_required=False,
        max_leverage=5.0,
        diversification_requirement='low',
        volatility_tolerance='high',
        tail_risk_protection=False,
        correlation_limit=0.7,
        var_limit=0.08,  # 8%
        stress_test_requirement=False
    ),
    
    RiskProfile.AGGRESSIVE: RiskConfiguration(
        max_drawdown=0.25,  # 25%
        position_sizing='aggressive',
        stop_loss_required=False,
        max_leverage=10.0,
        diversification_requirement='low',
        volatility_tolerance='high',
        tail_risk_protection=False,
        correlation_limit=0.8,
        var_limit=0.12,  # 12%
        stress_test_requirement=False
    ),
    
    RiskProfile.ULTRA_AGGRESSIVE: RiskConfiguration(
        max_drawdown=0.40,  # 40%
        position_sizing='aggressive',
        stop_loss_required=False,
        max_leverage=20.0,
        diversification_requirement='low',
        volatility_tolerance='high',
        tail_risk_protection=False,
        correlation_limit=1.0,
        var_limit=0.20,  # 20%
        stress_test_requirement=False
    ),
    
    RiskProfile.RISK_NEUTRAL: RiskConfiguration(
        max_drawdown=0.10,  # 10%
        position_sizing='moderate',
        stop_loss_required=False,
        max_leverage=1.0,
        diversification_requirement='high',
        volatility_tolerance='medium',
        tail_risk_protection=True,
        correlation_limit=0.5,
        var_limit=0.05,  # 5%
        stress_test_requirement=True
    )
}

# 显示名称
_DISPLAY_NAMES = {
    RiskProfile.ULTRA_CONSERVATIVE: "极保守型",
    RiskProfile.CONSERVATIVE: "保守型",
    RiskProfile.MODERATE: "稳健型",
    RiskProfile.BALANCED: "平衡型",
    RiskProfile.GROWTH: "成长型",
    RiskProfile.AGGRESSIVE: "积极型",
    RiskProfile.ULTRA_AGGRESSIVE: "极积极型",
    RiskProfile.RISK_NEUTRAL: "风险中性"
}

# 详细描述
_DESCRIPTIONS = {
    RiskProfile.ULTRA_CONSERVATIVE: "极度保守，优先保本，可接受极低收益换取资本安全",
    RiskProfile.CONSERVATIVE: "保守稳健，注重资本保护，追求稳定的低风险收益",
    RiskProfile.MODERATE: "稳健投资，平衡风险与收益，适合长期投资",
    RiskProfile.BALANCED: "平衡配置，适中的风险水平，追求稳定增长",
    RiskProfile.GROWTH: "成长导向，可承受较高风险以获取更高收益",
    RiskProfile.AGGRESSIVE: "积极进取，追求高收益，可承受显著风险",
    RiskProfile.ULTRA_AGGRESSIVE: "极度激进，追求最高收益，可承受极高风险",
    RiskProfile.RISK_NEUTRAL: "风险中性，不特别偏好风险或规避风险"
}

# 适合的市场类型
_SUITABLE_MARKETS = {
    RiskProfile.ULTRA_CONSERVATIVE: ["bond", "index"],
    RiskProfile.CONSERVATIVE: ["bond", "stock", "index"],
    RiskProfile.MODERATE: ["stock", "bond", "index", "commodity"],
    RiskProfile.BALANCED: ["stock", "index", "commodity", "forex"],
    RiskProfile.GROWTH: ["stock", "index", "commodity", "forex"],
    RiskProfile.AGGRESSIVE: ["stock", "forex", "commodity", "crypto"],
    RiskProfile.ULTRA_AGGRESSIVE: ["forex", "crypto", "option", "commodity"],
    RiskProfile.RISK_NEUTRAL: ["stock", "bond", "index", "forex"]
}

# 推荐的奖励函数
_RECOMMENDED_REWARDS = {
    RiskProfile.ULTRA_CONSERVATIVE: [
        "capital_preservation", "minimum_variance", "bond_like", "defensive"
    ],
    RiskProfile.CONSERVATIVE: [
        "risk_adjusted", "downside_protection", "low_volatility", "dividend_focused"
    ],
    RiskProfile.MODERATE: [
        "balanced_return", "moderate_risk", "steady_growth", "quality_focused"
    ],
    RiskProfile.BALANCED: [
        "risk_parity", "balanced_allocation", "diversified", "target_volatility"
    ],
    RiskProfile.GROWTH: [
        "growth_oriented", "momentum", "trend_following", "sector_rotation"
    ],
    RiskProfile.AGGRESSIVE: [
        "high_return", "leverage_enhanced", "momentum_strong", "breakout"
    ],
    RiskProfile.ULTRA_AGGRESSIVE: [
        "maximum_return", "ultra_momentum", "speculative", "high_leverage"
    ],
    RiskProfile.RISK_NEUTRAL: [
        "market_neutral", "arbitrage", "relative_value", "hedge_fund_like"
    ]
}

# 风险指标
_RISK_METRICS = {
    RiskProfile.ULTRA_CONSERVATIVE: {
        'target_volatility': 0.05,
        'max_correlation': 0.3,
        'min_sharpe_ratio': 1.5,
        'target_return': 0.03
    },
    RiskProfile.CONSERVATIVE: {
        'target_volatility': 0.08,
        'max_correlation': 0.4,
        'min_sharpe_ratio': 1.2,
        'target_return': 0.05
    },
    RiskProfile.MODERATE: {
        'target_volatility': 0.12,
        'max_correlation': 0.5,
        'min_sharpe_ratio': 1.0,
        'target_return': 0.08
    },
    RiskProfile.BALANCED: {
        'target_volatility': 0.15,
        'max_correlation': 0.6,
        'min_sharpe_ratio': 0.8,
        'target_return': 0.10
    },
    RiskProfile.GROWTH: {
        'target_volatility': 0.20,
        'max_correlation': 0.7,
        'min_sharpe_ratio': 0.6,
        'target_return': 0.12
    },
    RiskProfile.AGGRESSIVE: {
        'target_volatility': 0.30,
        'max_correlation': 0.8,
        'min_sharpe_ratio': 0.4,
        'target_return': 0.15
    },
    RiskProfile.ULTRA_AGGRESSIVE: {
        'target_volatility': 0.50,
        'max_correlation': 1.0,
        'min_sharpe_ratio': 0.2,
        'target_return': 0.25
    },
    RiskProfile.RISK_NEUTRAL: {
        'target_volatility': 0.10,
        'max_correlation': 0.5,
        'min_sharpe_ratio': 1.0,
        'target_return': 0.06
    }
}