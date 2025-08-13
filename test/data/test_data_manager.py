"""
测试数据管理器模块
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.data_manager import DataManager
from src.utils.config import Config


class TestDataManager:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.data_manager = DataManager(config=self.config)
    
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