"""
Forex Feature Engineer
外汇专用特征工程器

Purpose: 专门为EURUSD外汇交易设计的特征工程系统
解决历史实验中特征工程质量低下的问题

Key Features:
- 基于外汇交易理论设计的高质量特征
- 考虑外汇市场24/5交易特性
- 数值稳定性和计算效率优化
- 渐进式特征选择支持 (3→5→10特征)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, time

from ..utils.logger import setup_logger


class ForexFeatureEngineer:
    """
    专为EURUSD外汇交易优化的特征工程器
    
    基于外汇交易理论和实践经验，设计高质量的交易特征
    解决历史实验中特征质量低下导致的性能问题
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = setup_logger("ForexFeatureEngineer")
        
        # 外汇市场参数
        self.pip_value = 0.0001  # EURUSD点值
        self.typical_spreads = {
            'asian_session': 1.8,    # 亚洲时段点差
            'london_session': 1.2,   # 伦敦时段点差
            'ny_session': 1.0,       # 纽约时段点差
            'overlap_session': 0.8   # 重叠时段点差
        }
        
        # 交易时段定义 (UTC时间)
        self.trading_sessions = {
            'asian': {'start': 0, 'end': 8},      # 0:00-8:00 UTC
            'london': {'start': 8, 'end': 16},    # 8:00-16:00 UTC  
            'new_york': {'start': 13, 'end': 21}, # 13:00-21:00 UTC
            'overlap_london_ny': {'start': 13, 'end': 16}  # 重叠时段
        }
        
        # 特征集配置
        self.feature_sets = {
            'core_3': ['price_momentum', 'volatility_regime', 'trend_strength'],
            'basic_5': ['price_momentum', 'volatility_regime', 'trend_strength', 
                       'session_activity', 'support_resistance_distance'],
            'enhanced_10': ['price_momentum', 'volatility_regime', 'trend_strength',
                           'session_activity', 'support_resistance_distance',
                           'currency_strength', 'breakout_probability', 
                           'mean_reversion_signal', 'volume_confirmation', 'market_microstructure']
        }
        
        self.logger.info("ForexFeatureEngineer初始化完成")
    
    def create_features(self, data: pd.DataFrame, feature_set: str = 'core_3') -> pd.DataFrame:
        """
        创建外汇专用特征集
        
        Args:
            data: OHLCV数据
            feature_set: 特征集类型 ('core_3', 'basic_5', 'enhanced_10')
            
        Returns:
            features: 特征DataFrame
        """
        
        if feature_set not in self.feature_sets:
            raise ValueError(f"不支持的特征集: {feature_set}. 支持: {list(self.feature_sets.keys())}")
        
        self.logger.info(f"创建{feature_set}特征集，包含{len(self.feature_sets[feature_set])}个特征")
        
        features = {}
        feature_list = self.feature_sets[feature_set]
        
        # 基础价格数据预处理
        data = self._preprocess_data(data)
        
        # 逐个创建特征
        for feature_name in feature_list:
            try:
                feature_values = self._calculate_feature(data, feature_name)
                features[feature_name] = feature_values
                self.logger.debug(f"创建特征: {feature_name}")
            except Exception as e:
                self.logger.error(f"创建特征{feature_name}失败: {e}")
                # 创建零值特征作为占位符
                features[feature_name] = np.zeros(len(data))
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features, index=data.index)
        
        # 数据清理和验证
        features_df = self._clean_features(features_df)
        
        self.logger.info(f"特征创建完成: {features_df.shape}")
        return features_df
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理OHLCV数据"""
        data = data.copy()
        
        # 确保必需列存在
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"缺失列: {missing_columns}")
            # 创建缺失列的近似值
            for col in missing_columns:
                if col == 'Volume':
                    data[col] = 1000000  # 默认成交量
                else:
                    data[col] = data['Close'] if 'Close' in data.columns else 1.0
        
        # 计算基础衍生数据
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['price_range'] = data['High'] - data['Low']
        
        return data
    
    def _calculate_feature(self, data: pd.DataFrame, feature_name: str) -> np.ndarray:
        """计算单个特征"""
        
        if feature_name == 'price_momentum':
            return self._calculate_price_momentum(data)
        
        elif feature_name == 'volatility_regime':
            return self._calculate_volatility_regime(data)
        
        elif feature_name == 'trend_strength':
            return self._calculate_trend_strength(data)
        
        elif feature_name == 'session_activity':
            return self._calculate_session_activity(data)
        
        elif feature_name == 'support_resistance_distance':
            return self._calculate_support_resistance_distance(data)
        
        elif feature_name == 'currency_strength':
            return self._calculate_currency_strength(data)
        
        elif feature_name == 'breakout_probability':
            return self._calculate_breakout_probability(data)
        
        elif feature_name == 'mean_reversion_signal':
            return self._calculate_mean_reversion_signal(data)
        
        elif feature_name == 'volume_confirmation':
            return self._calculate_volume_confirmation(data)
        
        elif feature_name == 'market_microstructure':
            return self._calculate_market_microstructure(data)
        
        else:
            raise ValueError(f"未知特征: {feature_name}")
    
    def _calculate_price_momentum(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        计算价格动量特征
        基于RSI改进，适应外汇市场特性
        """
        close_prices = data['Close']
        
        # 计算价格变化
        delta = close_prices.diff()
        
        # 分离涨跌
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # 计算平均涨跌
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        
        # 避免除零
        avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        momentum = 100 - (100 / (1 + rs))
        
        # 标准化到[-1, 1]范围
        momentum_normalized = (momentum - 50) / 50
        
        return momentum_normalized.fillna(0).values
    
    def _calculate_volatility_regime(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """
        计算波动性状态特征
        识别低波动/高波动状态，对外汇风险管理重要
        """
        # 计算真实波动范围 (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        # 计算历史波动率
        returns = data['Close'].pct_change()
        historical_vol = returns.rolling(window=period).std() * np.sqrt(252)  # 年化
        
        # 结合ATR和历史波动率
        combined_vol = atr / data['Close'] + historical_vol
        
        # 计算波动率的相对位置 (0-1范围)
        vol_min = combined_vol.rolling(window=100).min()
        vol_max = combined_vol.rolling(window=100).max()
        
        vol_regime = (combined_vol - vol_min) / (vol_max - vol_min + 1e-10)
        
        # 转换为[-1, 1]范围, -1表示低波动，1表示高波动
        vol_regime_normalized = (vol_regime * 2) - 1
        
        return vol_regime_normalized.fillna(0).values
    
    def _calculate_trend_strength(self, data: pd.DataFrame, period: int = 21) -> np.ndarray:
        """
        计算趋势强度特征
        基于EMA斜率和趋势一致性
        """
        close_prices = data['Close']
        
        # 计算指数移动平均
        ema = close_prices.ewm(span=period).mean()
        
        # 计算EMA斜率
        ema_slope = ema.diff(5) / ema.shift(5)  # 5期斜率
        
        # 计算趋势一致性（价格相对EMA的位置稳定性）
        price_position = (close_prices - ema) / ema
        trend_consistency = 1 - price_position.rolling(window=10).std()
        
        # 结合斜率和一致性
        trend_strength = ema_slope * trend_consistency
        
        # 标准化
        trend_strength_normalized = np.tanh(trend_strength * 10)  # 使用tanh限制范围
        
        return trend_strength_normalized.fillna(0).values
    
    def _calculate_session_activity(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算交易时段活跃度特征
        外汇市场24/5特性，不同时段有不同的流动性和波动性
        """
        # 由于简化，使用基于价格波动的代理指标
        price_range = data['High'] - data['Low']
        volume = data['Volume']
        
        # 计算活跃度（价格波动*成交量）
        activity = price_range * volume
        
        # 滚动标准化
        activity_norm = (activity - activity.rolling(window=100).mean()) / (activity.rolling(window=100).std() + 1e-10)
        
        # 限制范围
        activity_norm = np.clip(activity_norm, -3, 3) / 3  # 标准化到[-1, 1]
        
        return activity_norm.fillna(0).values
    
    def _calculate_support_resistance_distance(self, data: pd.DataFrame, period: int = 50) -> np.ndarray:
        """
        计算支撑阻力位距离特征
        价格相对于近期支撑阻力位的距离
        """
        close_prices = data['Close']
        high_prices = data['High']  
        low_prices = data['Low']
        
        # 计算动态支撑阻力位
        resistance = high_prices.rolling(window=period).max()
        support = low_prices.rolling(window=period).min()
        
        # 计算价格在支撑阻力区间的相对位置
        range_size = resistance - support + 1e-10
        price_position = (close_prices - support) / range_size
        
        # 转换为距离概念：0.5表示中间位置，0表示支撑位，1表示阻力位
        # 转换为[-1, 1]范围：-1接近支撑，1接近阻力，0在中间
        distance_normalized = (price_position - 0.5) * 2
        
        return distance_normalized.fillna(0).values
    
    def _calculate_currency_strength(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """
        计算货币强度特征
        基于价格变化的相对强度（简化版本）
        """
        returns = data['Close'].pct_change()
        
        # 计算滚动平均回报
        avg_return = returns.rolling(window=period).mean()
        
        # 计算回报的稳定性
        return_stability = 1 - (returns.rolling(window=period).std() / (abs(avg_return) + 1e-10))
        
        # 货币强度 = 平均回报 * 稳定性
        currency_strength = avg_return * return_stability
        
        # 标准化
        strength_normalized = np.tanh(currency_strength * 100)
        
        return strength_normalized.fillna(0).values
    
    def _calculate_breakout_probability(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """
        计算突破概率特征
        基于价格压缩和成交量的突破信号
        """
        close_prices = data['Close']
        volume = data['Volume']
        
        # 计算价格压缩程度（波动率收缩）
        returns_std = close_prices.pct_change().rolling(window=period).std()
        compression = 1 - (returns_std / returns_std.rolling(window=100).mean())
        
        # 计算成交量相对强度
        volume_strength = volume / volume.rolling(window=period).mean()
        
        # 突破概率 = 价格压缩 * 成交量强度
        breakout_prob = compression * np.log1p(volume_strength)  # 使用log1p避免极值
        
        # 标准化到[0, 1]然后转换为[-1, 1]
        breakout_prob_norm = (breakout_prob - breakout_prob.rolling(window=100).min()) / (
            breakout_prob.rolling(window=100).max() - breakout_prob.rolling(window=100).min() + 1e-10
        )
        
        breakout_prob_final = (breakout_prob_norm * 2) - 1
        
        return breakout_prob_final.fillna(0).values
    
    def _calculate_mean_reversion_signal(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        计算均值回归信号特征
        价格偏离均值的程度，用于识别反转机会
        """
        close_prices = data['Close']
        
        # 计算移动平均
        ma = close_prices.rolling(window=period).mean()
        
        # 计算标准差
        std = close_prices.rolling(window=period).std()
        
        # 计算z-score (价格偏离程度)
        z_score = (close_prices - ma) / (std + 1e-10)
        
        # 转换为均值回归信号：极值时回归信号强
        # 使用tanh函数限制范围并增强极值信号
        reversion_signal = -np.tanh(z_score)  # 负号：偏离越大，回归信号越强
        
        return reversion_signal.fillna(0).values
    
    def _calculate_volume_confirmation(self, data: pd.DataFrame, period: int = 10) -> np.ndarray:
        """
        计算成交量确认特征
        成交量与价格变化的一致性
        """
        returns = data['Close'].pct_change()
        volume = data['Volume']
        
        # 计算成交量变化
        volume_change = volume.pct_change()
        
        # 计算价格与成交量变化的相关性（滚动）
        price_vol_correlation = returns.rolling(window=period).corr(volume_change)
        
        # 计算成交量相对强度
        volume_strength = volume / volume.rolling(window=period).mean() - 1
        
        # 成交量确认 = 相关性 * 成交量强度
        volume_confirmation = price_vol_correlation * volume_strength
        
        # 标准化
        confirmation_normalized = np.tanh(volume_confirmation)
        
        return confirmation_normalized.fillna(0).values
    
    def _calculate_market_microstructure(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算市场微观结构特征
        基于价格分布和tick数据特征的代理指标
        """
        open_prices = data['Open']
        high_prices = data['High']
        low_prices = data['Low']
        close_prices = data['Close']
        
        # 计算价格分布特征
        # 1. 开盘到收盘的价格压力
        open_close_pressure = (close_prices - open_prices) / (high_prices - low_prices + 1e-10)
        
        # 2. 价格在high-low range中的位置
        price_position_in_range = (close_prices - low_prices) / (high_prices - low_prices + 1e-10)
        
        # 3. 价格范围相对大小
        range_relative_size = (high_prices - low_prices) / close_prices
        
        # 综合微观结构信号
        microstructure = (
            open_close_pressure * 0.4 +
            (price_position_in_range - 0.5) * 2 * 0.4 +  # 转换为[-1,1]范围
            np.tanh(range_relative_size * 100) * 0.2
        )
        
        return microstructure.fillna(0).values
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """清理和验证特征数据"""
        
        # 1. 处理无穷值和NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # 2. 检查数值范围
        for col in features_df.columns:
            col_data = features_df[col]
            if col_data.abs().max() > 100:  # 异常大值
                self.logger.warning(f"特征{col}存在异常大值，进行截断处理")
                features_df[col] = np.clip(col_data, -10, 10)
        
        # 3. 数据类型转换
        features_df = features_df.astype(np.float32)  # 节省内存
        
        self.logger.info(f"特征清理完成，最终特征矩阵: {features_df.shape}")
        
        return features_df
    
    def get_feature_info(self, feature_set: str = 'core_3') -> Dict:
        """获取特征集信息"""
        
        if feature_set not in self.feature_sets:
            raise ValueError(f"不支持的特征集: {feature_set}")
        
        feature_descriptions = {
            'price_momentum': '价格动量 - 基于RSI改进的趋势跟踪指标',
            'volatility_regime': '波动性状态 - 识别市场波动性状态的指标', 
            'trend_strength': '趋势强度 - 基于EMA斜率和一致性的趋势指标',
            'session_activity': '时段活跃度 - 反映不同交易时段市场活跃程度',
            'support_resistance_distance': '支撑阻力距离 - 价格相对支撑阻力位的位置',
            'currency_strength': '货币强度 - 货币相对强弱指标',
            'breakout_probability': '突破概率 - 基于价格压缩和成交量的突破信号',
            'mean_reversion_signal': '均值回归信号 - 价格偏离均值的回归机会',
            'volume_confirmation': '成交量确认 - 成交量与价格变化的一致性',
            'market_microstructure': '市场微观结构 - 基于价格分布的微观结构特征'
        }
        
        selected_features = self.feature_sets[feature_set]
        
        return {
            'feature_set': feature_set,
            'feature_count': len(selected_features),
            'features': selected_features,
            'descriptions': {feat: feature_descriptions[feat] for feat in selected_features},
            'designed_for': 'EURUSD外汇交易',
            'theoretical_basis': '基于外汇交易理论和市场微观结构',
            'value_range': '所有特征标准化到[-1, 1]范围',
            'data_type': 'float32',
            'missing_value_handling': '前向填充后零值填充',
            'update_frequency': '与输入数据频率一致'
        }


# 使用示例和测试
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    test_data = pd.DataFrame({
        'Open': np.random.randn(1000).cumsum() + 1.1000,
        'High': np.random.randn(1000).cumsum() + 1.1020,
        'Low': np.random.randn(1000).cumsum() + 0.9980,
        'Close': np.random.randn(1000).cumsum() + 1.1010,
        'Volume': np.random.randint(100000, 1000000, 1000)
    }, index=dates)
    
    # 创建特征工程器
    engineer = ForexFeatureEngineer()
    
    # 测试不同特征集
    for feature_set in ['core_3', 'basic_5', 'enhanced_10']:
        print(f"\n测试{feature_set}特征集:")
        
        features = engineer.create_features(test_data, feature_set)
        info = engineer.get_feature_info(feature_set)
        
        print(f"特征数量: {info['feature_count']}")
        print(f"特征列表: {info['features']}")
        print(f"数据形状: {features.shape}")
        print(f"数值范围: [{features.min().min():.3f}, {features.max().max():.3f}]")
        print(f"缺失值: {features.isnull().sum().sum()}")