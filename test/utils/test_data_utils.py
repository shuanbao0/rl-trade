"""
测试数据工具模块
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.utils.data_utils import (
    validate_dataframe, 
    clean_market_data,
    calculate_returns,
    normalize_prices,
    detect_outliers,
    fill_missing_data,
    resample_data
)


class TestDataUtils:
    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 102,
            'Volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_validate_dataframe_valid(self):
        """测试有效数据框验证"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        result = validate_dataframe(self.test_df, required_columns)
        assert result is True
    
    def test_validate_dataframe_missing_columns(self):
        """测试缺失列的数据框验证"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Missing']
        result = validate_dataframe(self.test_df, required_columns)
        assert result is False
    
    def test_validate_dataframe_empty(self):
        """测试空数据框验证"""
        empty_df = pd.DataFrame()
        result = validate_dataframe(empty_df, ['Close'])
        assert result is False
    
    def test_clean_market_data_basic(self):
        """测试基础市场数据清洗"""
        # 添加一些脏数据
        dirty_df = self.test_df.copy()
        dirty_df.loc[10, 'Close'] = np.inf  # 无穷值
        dirty_df.loc[20, 'Volume'] = -100   # 负值
        dirty_df.loc[30, 'High'] = np.nan   # NaN值
        
        cleaned_df = clean_market_data(dirty_df)
        
        # 验证清洗结果
        assert not np.isinf(cleaned_df['Close']).any()
        assert not (cleaned_df['Volume'] < 0).any()
        assert not cleaned_df['High'].isna().any()
    
    def test_calculate_returns_simple(self):
        """测试简单收益率计算"""
        prices = pd.Series([100, 105, 102, 108, 110])
        returns = calculate_returns(prices, method='simple')
        
        expected_returns = prices.pct_change().fillna(0)
        pd.testing.assert_series_equal(returns, expected_returns)
    
    def test_calculate_returns_log(self):
        """测试对数收益率计算"""
        prices = pd.Series([100, 105, 102, 108, 110])
        returns = calculate_returns(prices, method='log')
        
        expected_returns = np.log(prices / prices.shift(1)).fillna(0)
        pd.testing.assert_series_equal(returns, expected_returns)
    
    def test_normalize_prices_minmax(self):
        """测试MinMax价格归一化"""
        prices = pd.Series([100, 150, 80, 200, 120])
        normalized = normalize_prices(prices, method='minmax')
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(prices)
    
    def test_normalize_prices_zscore(self):
        """测试Z-score价格归一化"""
        prices = pd.Series([100, 150, 80, 200, 120])
        normalized = normalize_prices(prices, method='zscore')
        
        # Z-score归一化后均值应接近0，标准差接近1
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1
    
    def test_detect_outliers_iqr(self):
        """测试IQR方法异常值检测"""
        # 创建包含异常值的数据
        data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
        outliers = detect_outliers(data, method='iqr')
        
        # 验证100被识别为异常值
        assert 100 in data[outliers].values
        assert len(outliers[outliers]) >= 1
    
    def test_detect_outliers_zscore(self):
        """测试Z-score方法异常值检测"""
        data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
        outliers = detect_outliers(data, method='zscore', threshold=2)
        
        # 验证100被识别为异常值
        assert 100 in data[outliers].values
    
    def test_fill_missing_data_forward(self):
        """测试前向填充缺失数据"""
        data = pd.Series([1, 2, np.nan, 4, np.nan, 6])
        filled = fill_missing_data(data, method='forward')
        
        expected = pd.Series([1.0, 2.0, 2.0, 4.0, 4.0, 6.0])
        pd.testing.assert_series_equal(filled, expected)
    
    def test_fill_missing_data_backward(self):
        """测试后向填充缺失数据"""
        data = pd.Series([1, 2, np.nan, 4, np.nan, 6])
        filled = fill_missing_data(data, method='backward')
        
        expected = pd.Series([1.0, 2.0, 4.0, 4.0, 6.0, 6.0])
        pd.testing.assert_series_equal(filled, expected)
    
    def test_fill_missing_data_interpolate(self):
        """测试插值填充缺失数据"""
        data = pd.Series([1, 2, np.nan, 4, np.nan, 6])
        filled = fill_missing_data(data, method='interpolate')
        
        # 验证插值结果合理
        assert not filled.isna().any()
        assert filled.iloc[2] == 3.0  # (2+4)/2
        assert filled.iloc[4] == 5.0  # (4+6)/2
    
    def test_resample_data_daily_to_weekly(self):
        """测试数据重采样：日频到周频"""
        # 创建日频数据
        dates = pd.date_range(start='2023-01-01', periods=14, freq='D')
        daily_data = pd.DataFrame({
            'Date': dates,
            'Close': range(1, 15),
            'Volume': range(100, 1500, 100)
        })
        daily_data.set_index('Date', inplace=True)
        
        weekly_data = resample_data(daily_data, freq='W')
        
        # 验证重采样结果
        assert len(weekly_data) < len(daily_data)
        assert 'Close' in weekly_data.columns
        assert 'Volume' in weekly_data.columns
    
    def test_resample_data_with_aggregation(self):
        """测试带聚合函数的数据重采样"""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Price': range(1, 11),
            'Volume': range(10, 101, 10)
        })
        data.set_index('Date', inplace=True)
        
        # 自定义聚合规则
        agg_rules = {
            'Price': 'last',
            'Volume': 'sum'
        }
        
        resampled = resample_data(data, freq='2D', agg_rules=agg_rules)
        
        # 验证聚合结果
        assert len(resampled) == 5  # 10天按2天聚合
        assert resampled['Volume'].sum() > data['Volume'].sum() * 0.9  # 考虑精度误差