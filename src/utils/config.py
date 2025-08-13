"""
配置管理工具
提供统一的配置读取和管理
"""

import os
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """数据管理配置"""
    cache_dir: str = "data_cache"
    cache_expiry_hours: int = 24
    max_retries: int = 5  # 增加重试次数以应对限流
    retry_delay: float = 2.0  # 增加基础延迟
    request_timeout: float = 10.0
    

@dataclass
class FeatureConfig:
    """特征工程配置 - 汇率交易专用增强版"""
    
    # 基础移动平均线
    sma_periods: list = None
    ema_periods: list = None
    
    # 经典动量指标
    rsi_periods: list = None
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # 布林带参数
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # ADX趋势强度指标 (汇率交易核心)
    adx_period: int = 14
    adx_threshold: float = 25.0  # ADX>25表示强趋势
    
    # ATR波动性指标
    atr_periods: list = None
    
    # 随机指标
    stochastic_k_period: int = 14
    stochastic_d_period: int = 3
    stochastic_smooth: int = 3
    
    # 威廉指标
    williams_r_period: int = 14
    
    # CCI商品通道指数
    cci_period: int = 20
    cci_constant: float = 0.015
    
    # Parabolic SAR
    sar_acceleration: float = 0.02
    sar_max_acceleration: float = 0.2
    
    # 一目均衡表参数
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    
    # 多时间框架分析
    timeframes: list = None
    
    # 统计窗口
    statistical_windows: list = None
    
    # 汇率专用参数
    pip_decimal_places: int = 4  # EURUSD是4位小数
    volatility_windows: list = None
    
    # 支撑阻力位参数
    support_resistance_window: int = 50
    support_resistance_strength: int = 3
    
    # 突破信号参数
    breakout_lookback: int = 20
    breakout_threshold: float = 1.5  # ATR的倍数
    
    def __post_init__(self):
        # 基础移动平均线 - 汇率交易优化
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 200]  # 增加短期和长期
        if self.ema_periods is None:
            self.ema_periods = [8, 13, 21, 34, 55]   # 斐波那契周期
        
        # RSI多周期
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21]
        
        # ATR多周期波动性
        if self.atr_periods is None:
            self.atr_periods = [14, 21]
        
        # 多时间框架 (分钟级别)
        if self.timeframes is None:
            self.timeframes = ['5min', '15min', '1h', '4h', '1d']
        
        # 统计分析窗口
        if self.statistical_windows is None:
            self.statistical_windows = [5, 10, 20, 50]
        
        # 波动性分析窗口
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50]


@dataclass
class TradingConfig:
    """交易配置"""
    initial_balance: float = 10000.0
    commission: float = 0.01  # 调整为1%，避免精度问题
    max_position_size: float = 1.0
    observation_window: int = 50


@dataclass
class RewardConfig:
    """奖励函数配置"""
    # 奖励函数类型
    reward_type: str = "risk_adjusted"
    
    # 通用参数
    initial_balance: Optional[float] = None  # None表示使用trading.initial_balance
    
    # RiskAdjustedReward 专用参数
    risk_free_rate: float = 0.02
    window_size: int = 50
    
    # SimpleReturnReward 专用参数  
    step_weight: float = 1.0
    total_weight: float = 0.1
    
    # ProfitLossReward 专用参数
    min_trade_threshold: float = 0.001
    profit_bonus: float = 2.0
    loss_penalty: float = 1.5
    consecutive_loss_penalty: float = 0.5
    win_rate_bonus: float = 0.1
    
    # DiversifiedReward 专用参数
    weights: Optional[Dict[str, float]] = None
    volatility_window: int = 20
    drawdown_threshold: float = 0.1
    trading_cost: float = 0.001
    
    def __post_init__(self):
        # 设置DiversifiedReward的默认权重
        if self.weights is None:
            self.weights = {
                'return': 0.4,
                'risk': 0.2,  
                'stability': 0.15,
                'efficiency': 0.15,
                'drawdown': 0.1
            }
    
    def get_reward_params(self, reward_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定奖励函数类型的参数
        
        Args:
            reward_type: 奖励函数类型，None表示使用配置的类型
            
        Returns:
            Dict[str, Any]: 奖励函数参数字典
        """
        target_type = reward_type or self.reward_type
        
        # 基础参数
        params = {}
        if self.initial_balance is not None:
            params['initial_balance'] = self.initial_balance
        
        # 根据奖励函数类型添加特定参数
        if target_type in ['risk_adjusted', 'default', 'sharpe']:
            params.update({
                'risk_free_rate': self.risk_free_rate,
                'window_size': self.window_size
            })
            
        elif target_type in ['simple_return', 'simple', 'basic']:
            params.update({
                'step_weight': self.step_weight,
                'total_weight': self.total_weight
            })
            
        elif target_type in ['profit_loss', 'pnl']:
            params.update({
                'min_trade_threshold': self.min_trade_threshold,
                'profit_bonus': self.profit_bonus,
                'loss_penalty': self.loss_penalty,
                'consecutive_loss_penalty': self.consecutive_loss_penalty,
                'win_rate_bonus': self.win_rate_bonus
            })
            
        elif target_type in ['diversified', 'comprehensive', 'multi']:
            params.update({
                'weights': self.weights,
                'risk_free_rate': self.risk_free_rate,
                'volatility_window': self.volatility_window,
                'drawdown_threshold': self.drawdown_threshold,
                'trading_cost': self.trading_cost
            })
        
        return params


class Config:
    """
    统一配置管理器
    支持从环境变量、配置文件加载配置
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，支持JSON格式
        """
        self._config = {}
        
        # 加载默认配置
        self.data = DataConfig()
        self.feature = FeatureConfig()
        self.trading = TradingConfig()
        self.reward = RewardConfig()
        
        # 从配置文件加载
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # 从环境变量加载
        self.load_from_env()
    
    def load_from_file(self, config_file: str) -> None:
        """
        从配置文件加载配置
        
        Args:
            config_file: 配置文件路径
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 更新数据配置
            if 'data' in file_config:
                for key, value in file_config['data'].items():
                    if hasattr(self.data, key):
                        setattr(self.data, key, value)
            
            # 更新特征配置
            if 'feature' in file_config:
                for key, value in file_config['feature'].items():
                    if hasattr(self.feature, key):
                        setattr(self.feature, key, value)
            
            # 更新交易配置
            if 'trading' in file_config:
                for key, value in file_config['trading'].items():
                    if hasattr(self.trading, key):
                        setattr(self.trading, key, value)
            
            # 更新奖励配置
            if 'reward' in file_config:
                for key, value in file_config['reward'].items():
                    if hasattr(self.reward, key):
                        setattr(self.reward, key, value)
                        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
    
    def load_from_env(self) -> None:
        """从环境变量加载配置"""
        # 数据配置环境变量
        if os.getenv('DATA_CACHE_DIR'):
            self.data.cache_dir = os.getenv('DATA_CACHE_DIR')
        if os.getenv('DATA_CACHE_EXPIRY_HOURS'):
            self.data.cache_expiry_hours = int(os.getenv('DATA_CACHE_EXPIRY_HOURS'))
        
        # 交易配置环境变量
        if os.getenv('INITIAL_BALANCE'):
            self.trading.initial_balance = float(os.getenv('INITIAL_BALANCE'))
        if os.getenv('COMMISSION'):
            self.trading.commission = float(os.getenv('COMMISSION'))
        
        # 奖励配置环境变量
        if os.getenv('REWARD_TYPE'):
            self.reward.reward_type = os.getenv('REWARD_TYPE')
        if os.getenv('RISK_FREE_RATE'):
            self.reward.risk_free_rate = float(os.getenv('RISK_FREE_RATE'))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        self._config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        Returns:
            配置字典
        """
        return {
            'data': asdict(self.data),
            'feature': asdict(self.feature),
            'trading': asdict(self.trading),
            'reward': asdict(self.reward),
            'custom': self._config
        }
    
    def save_to_file(self, config_file: str) -> None:
        """
        保存配置到文件
        
        Args:
            config_file: 配置文件路径
        """
        config_dict = self.to_dict()
        
        # 确保目录存在
        config_dir = os.path.dirname(config_file)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def create_reward_function(self, reward_type: Optional[str] = None):
        """
        根据配置创建奖励函数实例
        
        Args:
            reward_type: 奖励函数类型，None表示使用配置的类型
            
        Returns:
            BaseRewardScheme: 奖励函数实例
        """
        # 延迟导入避免循环依赖
        from ..environment.rewards import create_reward_function
        
        target_type = reward_type or self.reward.reward_type
        
        # 获取奖励函数参数
        reward_params = self.reward.get_reward_params(target_type)
        
        # 如果没有设置initial_balance，使用trading配置的值
        if 'initial_balance' not in reward_params:
            reward_params['initial_balance'] = self.trading.initial_balance
        
        return create_reward_function(target_type, **reward_params) 