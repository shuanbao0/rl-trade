"""
测试数据管理器模块
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data import DataManager
from src.data.sources.base import DataSource, DataPeriod
from src.utils.config import Config


class TestDataManager:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.data_manager = DataManager(
            config=self.config, 
            data_source_type=DataSource.YFINANCE
        )
    
    def test_data_manager_initialization(self):
        """测试数据管理器初始化"""
        assert self.data_manager.config is not None
    
    @patch('yfinance.Ticker')
    def test_get_stock_data(self, mock_yf_ticker):
        """测试获取股票数据"""
        # 模拟yfinance返回数据，需要足够的记录（至少10条）和日期索引
        import pandas as pd
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=20), periods=20, freq='D')
        mock_data = pd.DataFrame({
            'Open': range(100, 120),
            'High': range(105, 125), 
            'Low': range(95, 115),
            'Close': range(103, 123),
            'Volume': range(1000, 3000, 100)
        }, index=dates)
        
        # 模拟yfinance.Ticker对象和其history方法
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_yf_ticker.return_value = mock_ticker_instance
        
        result = self.data_manager.get_stock_data('AAPL', period='1d')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_data_validation(self):
        """测试数据验证"""
        # 测试有效数据，需要足够的记录和正确的日期索引
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=15), periods=15, freq='D')
        valid_data = pd.DataFrame({
            'Open': range(100, 115), 
            'High': range(105, 120), 
            'Low': range(95, 110), 
            'Close': range(103, 118), 
            'Volume': range(1000, 2500, 100)
        }, index=dates)
        
        validation_result = self.data_manager.validate_data(valid_data)
        assert validation_result.is_valid is True
    
    def test_dataperiod_enum_support(self):
        """测试DataPeriod枚举支持"""
        # 测试获取支持的周期
        supported_periods = self.data_manager.get_supported_periods()
        assert len(supported_periods) > 0
        assert DataPeriod.YEAR_1 in supported_periods
        assert DataPeriod.MONTH_1 in supported_periods
        
        # 测试周期信息获取
        period_info = self.data_manager.get_period_info(DataPeriod.YEAR_1)
        assert period_info['enum_value'] == DataPeriod.YEAR_1
        assert period_info['string_value'] == '1y'
        assert period_info['display_name'] == '1 Year'
        assert period_info['days'] == 365
        
        # 测试字符串周期信息获取
        period_info_str = self.data_manager.get_period_info('2y')
        assert period_info_str['enum_value'] == DataPeriod.YEAR_2
        assert period_info_str['days'] == 730
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_with_dataperiod_enum(self, mock_yf_ticker):
        """测试使用DataPeriod枚举获取股票数据"""
        from datetime import datetime, timedelta
        
        # 模拟数据
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Open': range(100, 130),
            'High': range(105, 135), 
            'Low': range(95, 125),
            'Close': range(103, 133),
            'Volume': range(1000, 4000, 100)
        }, index=dates)
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_yf_ticker.return_value = mock_ticker_instance
        
        # 使用DataPeriod枚举获取数据
        result = self.data_manager.get_stock_data('AAPL', period=DataPeriod.MONTH_1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_convert_period_to_date_range(self):
        """测试周期到日期范围转换"""
        from datetime import datetime
        
        # 测试DataPeriod枚举转换
        date_range = self.data_manager.convert_period_to_date_range(DataPeriod.MONTH_1)
        assert date_range.duration_days == 30
        assert isinstance(date_range.start_date, datetime)
        assert isinstance(date_range.end_date, datetime)
        
        # 测试字符串周期转换
        date_range_str = self.data_manager.convert_period_to_date_range('1y')
        assert date_range_str.duration_days == 365
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_by_date_range(self, mock_yf_ticker):
        """测试按日期范围获取股票数据"""
        from datetime import datetime, timedelta
        
        # 模拟数据
        dates = pd.date_range(start=datetime(2023, 1, 1), end=datetime(2023, 6, 30), freq='D')
        mock_data = pd.DataFrame({
            'Open': range(100, 100 + len(dates)),
            'High': range(105, 105 + len(dates)), 
            'Low': range(95, 95 + len(dates)),
            'Close': range(103, 103 + len(dates)),
            'Volume': range(1000, 1000 + len(dates) * 100, 100)
        }, index=dates)
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_yf_ticker.return_value = mock_ticker_instance
        
        # 使用日期范围获取数据
        result = self.data_manager.get_stock_data_by_date_range(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-06-30'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_get_date_range_estimation(self):
        """测试日期范围下载估算"""
        estimation = self.data_manager.get_date_range_estimation(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            interval='1d'
        )
        
        assert 'date_range' in estimation
        assert 'validation' in estimation
        assert 'recommendations' in estimation
        assert estimation['date_range']['duration_days'] == 365