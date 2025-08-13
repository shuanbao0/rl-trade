"""
特征工程模块
负责技术指标计算、统计特征生成、数据预处理等功能

主要功能:
1. 技术指标计算 (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV)
2. 统计特征生成 (收益率, 滚动统计, 多时间窗口)
3. 数据预处理 (标准化, 缺失值处理)
4. 特征选择和输出格式化
5. 性能监控和日志记录
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler
import warnings

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


class FeatureEngineer:
    """
    特征工程器
    
    负责从原始价格数据生成技术指标和统计特征
    采用模块化设计，便于扩展和维护
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化特征工程器
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or Config()
        
        # 初始化日志
        self.logger = setup_logger(
            name="FeatureEngineer",
            level="INFO",
            log_file=get_default_log_file("feature_engineer")
        )
        
        # 初始化标准化器
        self.scaler = MinMaxScaler()
        self._scaler_fitted = False
        
        # 特征配置
        self.feature_config = self.config.feature
        
        # 定义特征列名
        self.feature_columns = []
        self._setup_feature_columns()
        
        self.logger.info("FeatureEngineer initialized successfully")
    
    def _setup_feature_columns(self) -> None:
        """设置汇率交易专用特征列名称"""
        self.feature_columns = []
        
        # 基础价格特征 (保留原始价格信息)
        self.feature_columns.extend(['Open', 'High', 'Low', 'Close'])
        # 注意：汇率通常没有成交量，但保留以兼容股票数据
        self.feature_columns.append('Volume')
        
        # === 趋势指标系统 ===
        # SMA移动平均线 (多周期趋势)
        for period in self.feature_config.sma_periods:
            self.feature_columns.append(f'SMA_{period}')
        
        # EMA指数移动平均线 (更敏感的趋势指标)
        for period in self.feature_config.ema_periods:
            self.feature_columns.append(f'EMA_{period}')
        
        # MACD趋势动量
        self.feature_columns.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
        
        # ADX趋势强度 (汇率交易核心指标)
        self.feature_columns.extend([f'ADX_{self.feature_config.adx_period}', 
                                   f'DI_Plus_{self.feature_config.adx_period}', 
                                   f'DI_Minus_{self.feature_config.adx_period}'])
        
        # === 动量振荡器系统 ===
        # RSI相对强弱指数 (多周期)
        for period in self.feature_config.rsi_periods:
            self.feature_columns.append(f'RSI_{period}')
        
        # 随机指标
        self.feature_columns.extend([f'Stochastic_K_{self.feature_config.stochastic_k_period}', 
                                   f'Stochastic_D_{self.feature_config.stochastic_d_period}'])
        
        # 威廉指标
        self.feature_columns.append(f'Williams_R_{self.feature_config.williams_r_period}')
        
        # CCI商品通道指数
        self.feature_columns.append(f'CCI_{self.feature_config.cci_period}')
        
        # ROC价格变化率
        self.feature_columns.extend(['ROC_12', 'ROC_25'])
        
        # === 波动性指标系统 ===
        # 布林带
        bb_period = self.feature_config.bollinger_period
        self.feature_columns.extend([
            f'BB_Upper_{bb_period}',
            f'BB_Lower_{bb_period}', 
            f'BB_Width_{bb_period}',
            f'BB_Position_{bb_period}',
            f'BB_Squeeze_{bb_period}'
        ])
        
        # ATR平均真实波动范围 (多周期)
        for period in self.feature_config.atr_periods:
            self.feature_columns.append(f'ATR_{period}')
        
        # 真实波动率
        for window in self.feature_config.volatility_windows:
            self.feature_columns.extend([
                f'Volatility_{window}d',
                f'Realized_Vol_{window}d'
            ])
        
        # === 汇率专用技术指标 ===
        # Parabolic SAR
        self.feature_columns.append('Parabolic_SAR')
        
        # 一目均衡表
        self.feature_columns.extend([
            f'Ichimoku_Tenkan_{self.feature_config.ichimoku_tenkan}',
            f'Ichimoku_Kijun_{self.feature_config.ichimoku_kijun}',
            f'Ichimoku_Senkou_A',
            f'Ichimoku_Senkou_B_{self.feature_config.ichimoku_senkou_b}',
            'Ichimoku_Cloud_Direction'
        ])
        
        # === 统计特征系统 ===
        for window in self.feature_config.statistical_windows:
            self.feature_columns.extend([
                f'Return_{window}d',         # 简单收益率
                f'LogReturn_{window}d',      # 对数收益率 (汇率更适合)
                f'Mean_{window}d',           # 滚动均值
                f'Std_{window}d',            # 滚动标准差
                f'Skewness_{window}d',       # 偏度 (分布形状)
                f'Kurtosis_{window}d',       # 峰度 (尾部风险)
                f'Sharpe_Ratio_{window}d'    # 滚动夏普比率
            ])
        
        # === 市场结构特征 ===
        # 价格位置和通道
        for window in [5, 10, 20, 50]:
            self.feature_columns.extend([
                f'Price_Position_{window}d',      # 价格在区间中的位置
                f'High_Low_Ratio_{window}d',      # 高低点比率
                f'Close_Position_{window}d'       # 收盘价位置
            ])
        
        # 支撑阻力位
        self.feature_columns.extend([
            f'Support_Level_{self.feature_config.support_resistance_window}',
            f'Resistance_Level_{self.feature_config.support_resistance_window}',
            f'Distance_to_Support',
            f'Distance_to_Resistance'
        ])
        
        # 突破信号
        self.feature_columns.extend([
            f'Breakout_Signal_{self.feature_config.breakout_lookback}',
            f'Breakdown_Signal_{self.feature_config.breakout_lookback}',
            f'Range_Breakout_Strength'
        ])
        
        # === 动量和加速度 ===
        for window in [10, 20]:
            self.feature_columns.extend([
                f'Momentum_{window}d',
                f'Acceleration_{window}d',
                f'Price_Velocity_{window}d'
            ])
        
        # === 风险度量指标 ===
        self.feature_columns.extend([
            'Current_Drawdown',              # 当前回撤
            'Max_Drawdown_20d',              # 最大回撤
            'Underwater_Duration',           # 水下时间
            'Pain_Index_20d',                # 痛苦指数
            'Calmar_Ratio_50d',              # Calmar比率
            'Sterling_Ratio_50d'             # Sterling比率
        ])
        
        # === 价格行为模式 ===
        self.feature_columns.extend([
            'Candle_Body_Size',              # 实体大小
            'Upper_Shadow_Size',             # 上影线大小  
            'Lower_Shadow_Size',             # 下影线大小
            'Doji_Signal',                   # 十字星信号
            'Hammer_Signal',                 # 锤子线信号
            'Shooting_Star_Signal',          # 流星信号
            'Engulfing_Pattern'              # 吞噬形态
        ])
        
        # === 时间特征 (汇率市场时段分析) ===
        self.feature_columns.extend([
            'Hour_of_Day',                   # 小时 (交易时段)
            'Day_of_Week',                   # 星期 (周期模式)
            'Is_Asian_Session',              # 亚洲交易时段
            'Is_European_Session',           # 欧洲交易时段
            'Is_US_Session',                 # 美国交易时段
            'Session_Overlap'                # 交易时段重叠
        ])
        
        # === 成交量指标 (如果有成交量数据) ===
        self.feature_columns.extend([
            'OBV',                           # 能量潮
            'Volume_Ratio_5d',               # 成交量比率
            'Volume_Trend_20d'               # 成交量趋势
        ])
        
        self.logger.info(f"设置了 {len(self.feature_columns)} 个汇率交易专用特征")
        self.logger.debug(f"特征类别分布: 趋势({len([f for f in self.feature_columns if 'SMA' in f or 'EMA' in f or 'MACD' in f or 'ADX' in f])}), "
                         f"动量({len([f for f in self.feature_columns if 'RSI' in f or 'Stochastic' in f or 'Williams' in f])}), "
                         f"波动性({len([f for f in self.feature_columns if 'ATR' in f or 'BB_' in f or 'Volatility' in f])}), "
                         f"统计({len([f for f in self.feature_columns if 'Return' in f or 'Sharpe' in f])})")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据 - 主要入口方法
        
        Args:
            data: 原始OHLCV数据
            
        Returns:
            包含所有特征的DataFrame
            
        Raises:
            ValueError: 当输入数据无效时
            Exception: 当特征计算失败时
        """
        start_time = time.time()
        self.logger.info(f"Starting feature preparation for {len(data)} records")
        
        try:
            # 输入验证
            if data is None or data.empty:
                raise ValueError("Input data is None or empty")
            
            # Check for essential price columns
            required_price_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_price_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for Volume column and handle forex data
            has_volume = 'Volume' in data.columns
            if not has_volume:
                self.logger.warning("Volume column not found - creating dummy volume for forex data")
                processed_data = data.copy()
                processed_data['Volume'] = 0  # Create dummy volume for forex
            else:
                # Check if volume is meaningful (not all zeros)
                volume_sum = data['Volume'].sum()
                if volume_sum == 0:
                    self.logger.info("Volume data is all zeros (forex data detected)")
                processed_data = data.copy()
            
            # 计算技术指标
            self.logger.debug("Calculating technical indicators")
            processed_data = self.calculate_technical_indicators(processed_data)
            
            # 计算统计特征
            self.logger.debug("Calculating statistical features")
            processed_data = self.calculate_statistical_features(processed_data)
            
            # 数据预处理和标准化
            self.logger.debug("Preprocessing data")
            processed_data = self.preprocess_data(processed_data, fit_scaler=not self._scaler_fitted)
            
            # 特征选择
            feature_data = self._select_features(processed_data)
            
            duration = time.time() - start_time
            self.logger.info(
                f"Feature preparation completed. Generated {feature_data.shape[1]} features "
                f"from {len(data)} records in {duration:.2f} seconds"
            )
            
            return feature_data
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: OHLCV数据
            
        Returns:
            包含技术指标的数据
        """
        result = data.copy()
        
        try:
            # 趋势指标
            self._calculate_trend_indicators(result)
            
            # 动量指标
            self._calculate_momentum_indicators(result)
            
            # 波动性指标
            self._calculate_volatility_indicators(result)
            
            # 成交量指标
            self._calculate_volume_indicators(result)
            
            # === 汇率专用高级技术指标 ===
            # ADX和方向性指标
            self._calculate_adx_indicators(result)
            
            # Parabolic SAR
            self._calculate_parabolic_sar(result)
            
            # 一目均衡表
            self._calculate_ichimoku_indicators(result)
            
            # 市场结构指标
            self._calculate_market_structure(result)
            
            # 价格行为模式
            self._calculate_price_patterns(result)
            
            # 时间特征
            self._calculate_time_features(result)
            
            self.logger.debug("All technical indicators calculated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate technical indicators: {str(e)}")
            raise
        
        return result
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> None:
        """计算趋势指标 - 汇率交易增强版"""
        # Simple Moving Average (SMA) - 多周期趋势
        for period in self.feature_config.sma_periods:
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        
        # Exponential Moving Average (EMA) - 斐波那契周期
        for period in self.feature_config.ema_periods:
            data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        
        # MACD - 趋势动量系统
        ema_fast = data['Close'].ewm(span=self.feature_config.macd_fast).mean()
        ema_slow = data['Close'].ewm(span=self.feature_config.macd_slow).mean()
        data['MACD'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=self.feature_config.macd_signal).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> None:
        """计算动量指标 - 汇率交易增强版"""
        # RSI (Relative Strength Index) - 多周期分析
        delta = data['Close'].diff()
        for period in self.feature_config.rsi_periods:
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator - 配置化参数
        k_period = self.feature_config.stochastic_k_period
        d_period = self.feature_config.stochastic_d_period
        smooth = self.feature_config.stochastic_smooth
        
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()
        k_raw = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
        data[f'Stochastic_K_{k_period}'] = k_raw.rolling(window=smooth).mean()
        data[f'Stochastic_D_{d_period}'] = data[f'Stochastic_K_{k_period}'].rolling(window=d_period).mean()
        
        # Williams %R
        wr_period = self.feature_config.williams_r_period
        highest_high_wr = data['High'].rolling(window=wr_period).max()
        lowest_low_wr = data['Low'].rolling(window=wr_period).min()
        data[f'Williams_R_{wr_period}'] = -100 * (highest_high_wr - data['Close']) / (highest_high_wr - lowest_low_wr)
        
        # CCI (Commodity Channel Index)
        cci_period = self.feature_config.cci_period
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=cci_period).mean()
        mean_dev = typical_price.rolling(window=cci_period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        data[f'CCI_{cci_period}'] = (typical_price - sma_tp) / (self.feature_config.cci_constant * mean_dev)
        
        # ROC (Rate of Change) - 价格动量
        data['ROC_12'] = ((data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12)) * 100
        data['ROC_25'] = ((data['Close'] - data['Close'].shift(25)) / data['Close'].shift(25)) * 100
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> None:
        """计算波动性指标 - 汇率交易增强版"""
        # Bollinger Bands - 增强版
        period = self.feature_config.bollinger_period
        std_dev = self.feature_config.bollinger_std
        
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        
        data[f'BB_Upper_{period}'] = sma + (std * std_dev)
        data[f'BB_Lower_{period}'] = sma - (std * std_dev)
        data[f'BB_Width_{period}'] = data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}']
        data[f'BB_Position_{period}'] = (data['Close'] - data[f'BB_Lower_{period}']) / data[f'BB_Width_{period}']
        
        # Bollinger Band Squeeze (布林带收缩)
        data[f'BB_Squeeze_{period}'] = data[f'BB_Width_{period}'] / sma
        
        # Average True Range (ATR) - 多周期
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        for period in self.feature_config.atr_periods:
            data[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        
        # 实际波动率 (多时间窗口)
        for window in self.feature_config.volatility_windows:
            # 简单波动率
            returns = data['Close'].pct_change()
            data[f'Volatility_{window}d'] = returns.rolling(window=window).std()
            
            # 实现波动率 (已实现波动率)
            data[f'Realized_Vol_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252)
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> None:
        """计算成交量指标"""
        # Check if volume data is meaningful
        volume_sum = data['Volume'].sum()
        
        if volume_sum == 0:
            # For forex data with zero volume, create a simple price-based OBV
            self.logger.debug("Creating price-based OBV for forex data (zero volume)")
            obv = np.zeros(len(data))
            obv[0] = 1  # Start with 1 instead of 0 volume
            
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] + 1  # Use +1 for up days
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] - 1  # Use -1 for down days  
                else:
                    obv[i] = obv[i-1]
        else:
            # Traditional OBV calculation for stock data
            obv = np.zeros(len(data))
            obv[0] = data['Volume'].iloc[0]
            
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] + data['Volume'].iloc[i]
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] - data['Volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
        
        data['OBV'] = obv
    
    def calculate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算统计特征
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            包含统计特征的数据
        """
        result = data.copy()
        
        try:
            # 收益率特征
            self._calculate_return_features(result)
            
            # 滚动统计特征
            self._calculate_rolling_features(result)
            
            # 价格位置特征
            self._calculate_position_features(result)
            
            # === 汇率专用统计特征 ===
            # 动量和加速度
            self._calculate_momentum_acceleration(result)
            
            # 风险度量指标
            self._calculate_risk_metrics(result)
            
            self.logger.debug("All statistical features calculated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate statistical features: {str(e)}")
            raise
        
        return result
    
    def _calculate_return_features(self, data: pd.DataFrame) -> None:
        """计算收益率特征 - 汇率交易增强版"""
        windows = self.feature_config.statistical_windows
        
        for window in windows:
            # 简单收益率
            data[f'Return_{window}d'] = data['Close'].pct_change(periods=window)
            
            # 对数收益率 (汇率更适合)
            data[f'LogReturn_{window}d'] = np.log(data['Close'] / data['Close'].shift(window))
            
            # 波动率 (滚动标准差)
            returns = data['Close'].pct_change()
            data[f'Volatility_{window}d'] = returns.rolling(window=window).std()
            
            # 偏度 (分布形状) 
            data[f'Skewness_{window}d'] = returns.rolling(window=window).skew()
            
            # 峰度 (尾部风险)
            data[f'Kurtosis_{window}d'] = returns.rolling(window=window).kurt()
            
            # 滚动夏普比率 (假设无风险利率为0)
            data[f'Sharpe_Ratio_{window}d'] = data[f'Return_{window}d'] / data[f'Volatility_{window}d']
    
    def _calculate_rolling_features(self, data: pd.DataFrame) -> None:
        """计算滚动统计特征 - 汇率交易增强版"""
        windows = self.feature_config.statistical_windows
        
        for window in windows:
            # 滚动均值
            data[f'Mean_{window}d'] = data['Close'].rolling(window=window).mean()
            
            # 滚动标准差
            data[f'Std_{window}d'] = data['Close'].rolling(window=window).std()
    
    def _calculate_position_features(self, data: pd.DataFrame) -> None:
        """计算价格位置特征"""
        # 价格在历史范围内的位置
        data['Price_Position_5d'] = (
            (data['Close'] - data['Low'].rolling(5).min()) /
            (data['High'].rolling(5).max() - data['Low'].rolling(5).min())
        )
        
        data['Price_Position_20d'] = (
            (data['Close'] - data['Low'].rolling(20).min()) /
            (data['High'].rolling(20).max() - data['Low'].rolling(20).min())
        )
        
        # 成交量比率
        data['Volume_Ratio_5d'] = data['Volume'] / data['Volume'].rolling(5).mean()
    
    def preprocess_data(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 要预处理的数据
            fit_scaler: 是否拟合标准化器
            
        Returns:
            预处理后的数据
        """
        result = data.copy()
        
        try:
            # 处理无穷大值
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # 处理缺失值
            self._handle_missing_values(result)
            
            # 特征选择 - 只保留有效的特征列
            available_features = [col for col in self.feature_columns if col in result.columns]
            result = result[available_features]
            
            # 数据标准化
            if fit_scaler:
                # 只对数值型列进行标准化，排除可能的分类特征
                numeric_columns = result.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) > 0:
                    result[numeric_columns] = self.scaler.fit_transform(result[numeric_columns])
                    self._scaler_fitted = True
                    self.logger.debug(f"Fitted scaler on {len(numeric_columns)} numeric columns")
            else:
                if self._scaler_fitted:
                    numeric_columns = result.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        result[numeric_columns] = self.scaler.transform(result[numeric_columns])
                        self.logger.debug(f"Applied scaler to {len(numeric_columns)} numeric columns")
            
            self.logger.debug("Data preprocessing completed")
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise
        
        return result
    
    def _handle_missing_values(self, data: pd.DataFrame) -> None:
        """处理缺失值"""
        # 统计缺失值
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            self.logger.debug(f"Handling {missing_count} missing values")
            
            # 前向填充
            data.ffill(inplace=True)
            
            # 后向填充（处理开头的NaN）
            data.bfill(inplace=True)
            
            # 如果还有NaN，用0填充
            remaining_nan = data.isnull().sum().sum()
            if remaining_nan > 0:
                data.fillna(0, inplace=True)
                self.logger.warning(f"Filled {remaining_nan} remaining NaN values with 0")
    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        特征选择
        
        Args:
            data: 包含所有特征的数据
            
        Returns:
            选择后的特征数据
        """
        # 只保留在特征列表中的列
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        if len(available_features) == 0:
            raise ValueError("No valid features found in data")
        
        selected_data = data[available_features].copy()
        
        # 移除方差为0的特征（常数特征）
        variance = selected_data.var()
        constant_features = variance[variance == 0].index.tolist()
        
        if constant_features:
            self.logger.warning(f"Removing constant features: {constant_features}")
            selected_data = selected_data.drop(columns=constant_features)
        
        self.logger.debug(f"Selected {len(selected_data.columns)} features from {len(available_features)} available")
        
        return selected_data
    
    def get_feature_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征重要性 (基于方差)
        
        Args:
            data: 特征数据
            
        Returns:
            特征重要性DataFrame
        """
        try:
            variance = data.var().sort_values(ascending=False)
            importance = variance / variance.sum()
            
            importance_df = pd.DataFrame({
                'feature': importance.index,
                'variance': variance.values,
                'importance': importance.values
            })
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Failed to calculate feature importance: {str(e)}")
            raise
    
    def get_feature_stats(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        获取特征统计信息
        
        Args:
            data: 特征数据
            
        Returns:
            特征统计信息字典
        """
        try:
            stats = {
                'feature_count': len(data.columns),
                'sample_count': len(data),
                'missing_values': data.isnull().sum().sum(),
                'dtypes': data.dtypes.value_counts().to_dict(),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'feature_names': data.columns.tolist()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get feature stats: {str(e)}")
            raise
    
    def save_scaler(self, filepath: str) -> None:
        """
        保存标准化器到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            import joblib
            joblib.dump(self.scaler, filepath)
            self.logger.info(f"Scaler saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save scaler: {str(e)}")
            raise
    
    def load_scaler(self, filepath: str) -> None:
        """
        从文件加载标准化器
        
        Args:
            filepath: 文件路径
        """
        try:
            import joblib
            self.scaler = joblib.load(filepath)
            self._scaler_fitted = True
            self.logger.info(f"Scaler loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {str(e)}")
            raise
    
    def reset_scaler(self) -> None:
        """重置标准化器"""
        self.scaler = MinMaxScaler()
        self._scaler_fitted = False
        self.logger.info("Scaler reset")
    
    # === 汇率专用高级技术指标计算方法 ===
    
    def _calculate_adx_indicators(self, data: pd.DataFrame) -> None:
        """计算ADX趋势强度和方向性指标 - 汇率交易核心"""
        period = self.feature_config.adx_period
        
        # 计算True Range
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # 计算方向性移动
        plus_dm = np.where(
            (data['High'] - data['High'].shift()) > (data['Low'].shift() - data['Low']),
            np.maximum(data['High'] - data['High'].shift(), 0),
            0
        )
        minus_dm = np.where(
            (data['Low'].shift() - data['Low']) > (data['High'] - data['High'].shift()),
            np.maximum(data['Low'].shift() - data['Low'], 0),
            0
        )
        
        # 平滑处理
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).sum() / 
                         pd.Series(true_range).rolling(window=period).sum())
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).sum() / 
                          pd.Series(true_range).rolling(window=period).sum())
        
        # 计算ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        data[f'ADX_{period}'] = adx
        data[f'DI_Plus_{period}'] = plus_di
        data[f'DI_Minus_{period}'] = minus_di
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame) -> None:
        """计算Parabolic SAR - 趋势跟踪指标"""
        af = self.feature_config.sar_acceleration
        max_af = self.feature_config.sar_max_acceleration
        
        length = len(data)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1 for uptrend, -1 for downtrend
        af_current = af
        
        # 初始化
        sar[0] = data['Low'].iloc[0]
        trend[0] = 1
        ep = data['High'].iloc[0]  # Extreme Point
        
        for i in range(1, length):
            # 更新SAR
            sar[i] = sar[i-1] + af_current * (ep - sar[i-1])
            
            # 检查趋势反转
            if trend[i-1] == 1:  # 上升趋势
                if data['Low'].iloc[i] <= sar[i]:
                    # 趋势反转到下降
                    trend[i] = -1
                    sar[i] = ep
                    ep = data['Low'].iloc[i]
                    af_current = af
                else:
                    trend[i] = 1
                    if data['High'].iloc[i] > ep:
                        ep = data['High'].iloc[i]
                        af_current = min(af_current + af, max_af)
                    sar[i] = min(sar[i], data['Low'].iloc[i-1], data['Low'].iloc[i-2] if i > 1 else data['Low'].iloc[i-1])
            else:  # 下降趋势
                if data['High'].iloc[i] >= sar[i]:
                    # 趋势反转到上升
                    trend[i] = 1
                    sar[i] = ep
                    ep = data['High'].iloc[i]
                    af_current = af
                else:
                    trend[i] = -1
                    if data['Low'].iloc[i] < ep:
                        ep = data['Low'].iloc[i]
                        af_current = min(af_current + af, max_af)
                    sar[i] = max(sar[i], data['High'].iloc[i-1], data['High'].iloc[i-2] if i > 1 else data['High'].iloc[i-1])
        
        data['Parabolic_SAR'] = sar
    
    def _calculate_ichimoku_indicators(self, data: pd.DataFrame) -> None:
        """计算一目均衡表指标 - 日式技术分析"""
        tenkan_period = self.feature_config.ichimoku_tenkan
        kijun_period = self.feature_config.ichimoku_kijun
        senkou_b_period = self.feature_config.ichimoku_senkou_b
        
        # Tenkan-sen (转换线)
        tenkan_high = data['High'].rolling(window=tenkan_period).max()
        tenkan_low = data['Low'].rolling(window=tenkan_period).min()
        data[f'Ichimoku_Tenkan_{tenkan_period}'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (基准线)
        kijun_high = data['High'].rolling(window=kijun_period).max()
        kijun_low = data['Low'].rolling(window=kijun_period).min()
        data[f'Ichimoku_Kijun_{kijun_period}'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (先行跨度A)
        data['Ichimoku_Senkou_A'] = (data[f'Ichimoku_Tenkan_{tenkan_period}'] + data[f'Ichimoku_Kijun_{kijun_period}']) / 2
        
        # Senkou Span B (先行跨度B)
        senkou_b_high = data['High'].rolling(window=senkou_b_period).max()
        senkou_b_low = data['Low'].rolling(window=senkou_b_period).min()
        data[f'Ichimoku_Senkou_B_{senkou_b_period}'] = (senkou_b_high + senkou_b_low) / 2
        
        # 云的方向 (Cloud Direction)
        data['Ichimoku_Cloud_Direction'] = np.where(
            data['Ichimoku_Senkou_A'] > data[f'Ichimoku_Senkou_B_{senkou_b_period}'], 1, -1
        )
    
    def _calculate_market_structure(self, data: pd.DataFrame) -> None:
        """计算市场结构特征 - 支撑阻力位和突破信号"""
        # 支撑阻力位
        window = self.feature_config.support_resistance_window
        strength = self.feature_config.support_resistance_strength
        
        # 简化的支撑阻力计算
        data[f'Support_Level_{window}'] = data['Low'].rolling(window=window).min()
        data[f'Resistance_Level_{window}'] = data['High'].rolling(window=window).max()
        
        # 距离支撑阻力位的距离
        data['Distance_to_Support'] = (data['Close'] - data[f'Support_Level_{window}']) / data['Close']
        data['Distance_to_Resistance'] = (data[f'Resistance_Level_{window}'] - data['Close']) / data['Close']
        
        # 突破信号
        lookback = self.feature_config.breakout_lookback
        threshold = self.feature_config.breakout_threshold
        
        # 基于ATR的突破强度
        atr = data[f'ATR_{self.feature_config.atr_periods[0]}']
        
        # 向上突破信号
        data[f'Breakout_Signal_{lookback}'] = np.where(
            data['Close'] > data['High'].rolling(window=lookback).max().shift(1) + (atr * threshold),
            1, 0
        )
        
        # 向下突破信号  
        data[f'Breakdown_Signal_{lookback}'] = np.where(
            data['Close'] < data['Low'].rolling(window=lookback).min().shift(1) - (atr * threshold),
            1, 0
        )
        
        # 区间突破强度
        range_size = data['High'].rolling(window=lookback).max() - data['Low'].rolling(window=lookback).min()
        data['Range_Breakout_Strength'] = (data['Close'] - data['Close'].shift(1)) / range_size
        
        # 价格位置和通道特征
        for window in [5, 10, 20, 50]:
            high_max = data['High'].rolling(window=window).max()
            low_min = data['Low'].rolling(window=window).min()
            
            data[f'Price_Position_{window}d'] = (data['Close'] - low_min) / (high_max - low_min)
            data[f'High_Low_Ratio_{window}d'] = data['High'] / data['Low']
            data[f'Close_Position_{window}d'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
    
    def _calculate_price_patterns(self, data: pd.DataFrame) -> None:
        """计算价格行为模式 - K线形态分析"""
        # 蜡烛图组件
        data['Candle_Body_Size'] = np.abs(data['Close'] - data['Open'])
        data['Upper_Shadow_Size'] = data['High'] - np.maximum(data['Open'], data['Close'])
        data['Lower_Shadow_Size'] = np.minimum(data['Open'], data['Close']) - data['Low']
        
        # 归一化蜡烛组件
        total_range = data['High'] - data['Low']
        data['Candle_Body_Size'] = data['Candle_Body_Size'] / total_range
        data['Upper_Shadow_Size'] = data['Upper_Shadow_Size'] / total_range
        data['Lower_Shadow_Size'] = data['Lower_Shadow_Size'] / total_range
        
        # K线形态识别
        # 十字星 (Doji)
        body_threshold = 0.1  # 实体小于总区间的10%
        data['Doji_Signal'] = np.where(data['Candle_Body_Size'] < body_threshold, 1, 0)
        
        # 锤子线 (Hammer)
        data['Hammer_Signal'] = np.where(
            (data['Lower_Shadow_Size'] > 2 * data['Candle_Body_Size']) &
            (data['Upper_Shadow_Size'] < data['Candle_Body_Size']), 1, 0
        )
        
        # 流星 (Shooting Star)
        data['Shooting_Star_Signal'] = np.where(
            (data['Upper_Shadow_Size'] > 2 * data['Candle_Body_Size']) &
            (data['Lower_Shadow_Size'] < data['Candle_Body_Size']), 1, 0
        )
        
        # 吞噬形态 (简化版)
        data['Engulfing_Pattern'] = np.where(
            (data['Candle_Body_Size'] > data['Candle_Body_Size'].shift(1) * 1.5), 1, 0
        )
    
    def _calculate_time_features(self, data: pd.DataFrame) -> None:
        """计算时间特征 - 汇率市场时段分析"""
        if data.index.dtype == 'datetime64[ns]' or hasattr(data.index, 'hour'):
            # 小时特征 (0-23)
            data['Hour_of_Day'] = data.index.hour
            
            # 星期特征 (0-6, 0=Monday)
            data['Day_of_Week'] = data.index.dayofweek
            
            # 交易时段识别 (UTC时间)
            # 亚洲时段: 23:00-08:00 UTC
            data['Is_Asian_Session'] = np.where(
                (data['Hour_of_Day'] >= 23) | (data['Hour_of_Day'] <= 8), 1, 0
            )
            
            # 欧洲时段: 07:00-16:00 UTC  
            data['Is_European_Session'] = np.where(
                (data['Hour_of_Day'] >= 7) & (data['Hour_of_Day'] <= 16), 1, 0
            )
            
            # 美国时段: 13:00-22:00 UTC
            data['Is_US_Session'] = np.where(
                (data['Hour_of_Day'] >= 13) & (data['Hour_of_Day'] <= 22), 1, 0
            )
            
            # 交易时段重叠
            data['Session_Overlap'] = (
                data['Is_Asian_Session'] + 
                data['Is_European_Session'] + 
                data['Is_US_Session']
            )
        else:
            # 如果没有时间索引，填充默认值
            data['Hour_of_Day'] = 12  # 默认中午
            data['Day_of_Week'] = 2   # 默认周三
            data['Is_Asian_Session'] = 0
            data['Is_European_Session'] = 1  # 默认欧洲时段
            data['Is_US_Session'] = 0
            data['Session_Overlap'] = 1
    
    def _calculate_momentum_acceleration(self, data: pd.DataFrame) -> None:
        """计算动量和加速度特征 - 汇率交易专用"""
        for window in [10, 20]:
            # 价格动量 (价格变化率)
            data[f'Momentum_{window}d'] = data['Close'] / data['Close'].shift(window) - 1
            
            # 价格加速度 (动量的变化率)
            momentum = data[f'Momentum_{window}d']
            data[f'Acceleration_{window}d'] = momentum - momentum.shift(window//2)
            
            # 价格速度 (对数差分的移动平均)
            log_returns = np.log(data['Close'] / data['Close'].shift(1))
            data[f'Price_Velocity_{window}d'] = log_returns.rolling(window=window).mean()
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> None:
        """计算风险度量指标 - 汇率交易专用"""
        # 计算累积收益率
        returns = data['Close'].pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        
        # 当前回撤 (从最高点的跌幅)
        running_max = cumulative_returns.expanding().max()
        data['Current_Drawdown'] = (cumulative_returns / running_max - 1) * 100
        
        # 最大回撤 (20日)
        data['Max_Drawdown_20d'] = data['Current_Drawdown'].rolling(window=20).min()
        
        # 水下时间 (连续回撤天数)
        underwater = (data['Current_Drawdown'] < 0).astype(int)
        underwater_groups = underwater.ne(underwater.shift()).cumsum()
        data['Underwater_Duration'] = underwater.groupby(underwater_groups).cumsum()
        data['Underwater_Duration'] = data['Underwater_Duration'] * underwater
        
        # 痛苦指数 (平均回撤深度)
        data['Pain_Index_20d'] = abs(data['Current_Drawdown'].rolling(window=20).mean())
        
        # Calmar比率 (20日平均收益 / 最大回撤)
        avg_return_20d = returns.rolling(window=20).mean() * 252  # 年化
        max_dd_20d = abs(data['Max_Drawdown_20d']) / 100
        data['Calmar_Ratio_50d'] = avg_return_20d / max_dd_20d.replace(0, np.nan)
        
        # Sterling比率 (20日收益 / 平均回撤)
        avg_dd_20d = data['Pain_Index_20d'] / 100
        data['Sterling_Ratio_50d'] = avg_return_20d / avg_dd_20d.replace(0, np.nan)
    
    def _calculate_volume_enhancements(self, data: pd.DataFrame) -> None:
        """计算增强的成交量特征 - 汇率交易适配"""
        # 成交量比率 (如果有成交量数据)
        if 'Volume' in data.columns and data['Volume'].sum() > 0:
            data['Volume_Ratio_5d'] = data['Volume'] / data['Volume'].rolling(5).mean()
            data['Volume_Trend_20d'] = data['Volume'].rolling(20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
            )
        else:
            # 为汇率数据创建基于价格的"成交量"代理
            price_change = abs(data['Close'].pct_change())
            data['Volume_Ratio_5d'] = price_change / price_change.rolling(5).mean()
            data['Volume_Trend_20d'] = price_change.rolling(20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
            ) 