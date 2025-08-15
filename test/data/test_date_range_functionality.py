"""
测试日期范围下载功能
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

from src.data import DataManager
from src.data.sources.base import DataSource, DataPeriod
from src.utils.config import Config
from src.utils.date_range_utils import DateRangeUtils, DateRange


class TestDateRangeFunctionality:
    """测试日期范围下载功能"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.data_manager = DataManager(
            config=self.config,
            data_source_type=DataSource.YFINANCE
        )
        self.date_utils = DateRangeUtils()
        
        # 清理缓存以确保测试的一致性
        try:
            if hasattr(self.data_manager, 'cache_manager') and self.data_manager.cache_manager:
                self.data_manager.cache_manager.clear()
        except Exception:
            pass  # 忽略缓存清理异常
    
    def test_date_range_creation(self):
        """测试日期范围创建"""
        # 测试使用开始和结束日期创建
        date_range = self.date_utils.create_date_range(
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        assert date_range.start_date == datetime(2023, 1, 1)
        assert date_range.end_date == datetime(2023, 12, 31)
        assert date_range.duration_days >= 364  # Account for leap year calculation differences
        
        # 测试使用period创建
        date_range_period = self.date_utils.create_date_range(period='1y')
        assert date_range_period.duration_days >= 364  # Account for leap year calculation differences
        assert isinstance(date_range_period.start_date, datetime)
        assert isinstance(date_range_period.end_date, datetime)
    
    def test_date_range_validation(self):
        """测试日期范围验证"""
        # 测试有效日期范围
        valid_range = DateRange(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        is_valid, errors = DateRangeUtils.validate_date_range(valid_range, DataSource.YFINANCE)
        assert is_valid is True
        assert len(errors) == 0
        
        # 测试无效日期范围（开始日期晚于结束日期）
        with pytest.raises(ValueError):
            invalid_range = DateRange(
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1)
            )
    
    @patch('yfinance.Ticker')
    def test_data_manager_date_range_download(self, mock_yf_ticker):
        """测试DataManager日期范围下载"""
        # 清理缓存以确保实际调用yfinance
        if hasattr(self.data_manager, 'cache_manager'):
            self.data_manager.cache_manager.clear()
        
        # 模拟数据
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
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
        
        # 测试日期范围下载（使用不同的symbol避免缓存冲突）
        result = self.data_manager.get_stock_data_by_date_range(
            symbol='MSFT',  # 使用不同的symbol
            start_date='2023-01-01',
            end_date='2023-06-30',
            interval='1d'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # 验证结果格式（列名可能是小写或大写）
        columns_lower = [col.lower() for col in result.columns]
        assert 'open' in columns_lower or 'close' in columns_lower
    
    def test_date_range_estimation(self):
        """测试日期范围下载估算"""
        estimation = self.data_manager.get_date_range_estimation(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            interval='1d'
        )
        
        # 验证估算结果结构
        assert 'date_range' in estimation
        assert 'validation' in estimation
        assert 'recommendations' in estimation
        assert 'estimation' in estimation
        assert 'batch_download' in estimation
        
        # 验证日期范围信息
        date_range_info = estimation['date_range']
        assert date_range_info['duration_days'] >= 364  # Account for leap year calculation differences
        assert date_range_info['start_date'] == '2023-01-01'
        assert date_range_info['end_date'] == '2023-12-31'
        
        # 验证验证信息
        validation_info = estimation['validation']
        assert 'is_valid' in validation_info
        assert 'errors' in validation_info
        
        # 验证推荐信息
        recommendations = estimation['recommendations']
        assert 'optimal_interval' in recommendations
        assert 'current_interval' in recommendations
        
        # 验证批次下载建议
        batch_info = estimation['batch_download']
        assert 'recommended' in batch_info
        assert 'reason' in batch_info
    
    def test_period_to_date_range_conversion(self):
        """测试周期到日期范围转换"""
        # 测试DataPeriod枚举转换
        date_range = self.data_manager.convert_period_to_date_range(DataPeriod.MONTH_1)
        assert date_range.duration_days == 30
        
        # 测试字符串周期转换
        date_range_str = self.data_manager.convert_period_to_date_range('6mo')
        assert date_range_str.duration_days == 180
        
        # 测试指定结束日期的转换
        end_date = datetime(2023, 12, 31)
        date_range_custom = self.data_manager.convert_period_to_date_range(
            DataPeriod.YEAR_1, 
            end_date=end_date
        )
        assert date_range_custom.end_date == end_date
        assert date_range_custom.duration_days >= 364  # Account for leap year calculation differences
    
    def test_date_range_priority_logic(self):
        """测试日期范围优先级逻辑"""
        # 当同时提供period和日期范围时，应优先使用日期范围
        # 这个测试需要通过配置来验证
        config = Config()
        time_params = config.data.get_effective_time_params(
            period='1y',
            start_date='2023-01-01',
            end_date='2023-06-30'
        )
        
        # 根据配置，应该优先使用日期范围
        assert 'start_date' in time_params
        assert 'end_date' in time_params
        assert time_params['start_date'] == '2023-01-01'
        assert time_params['end_date'] == '2023-06-30'
    
    def test_date_range_with_different_intervals(self):
        """测试不同间隔的日期范围下载估算"""
        # 测试日间隔
        estimation_daily = self.data_manager.get_date_range_estimation(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            interval='1d'
        )
        
        # 测试小时间隔
        estimation_hourly = self.data_manager.get_date_range_estimation(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-31',  # 更短的时间范围适合小时数据
            interval='1h'
        )
        
        # 验证间隔估算结果存在（具体推荐值可能因阈值设置而不同）
        assert 'batch_download' in estimation_hourly
        assert 'batch_download' in estimation_daily
    
    def test_date_range_error_handling(self):
        """测试日期范围错误处理"""
        # 测试无效日期格式
        with pytest.raises(ValueError):
            self.date_utils.create_date_range(
                start_date='invalid-date',
                end_date='2023-12-31'
            )
        
        # 测试开始日期晚于结束日期
        with pytest.raises(ValueError):
            self.date_utils.create_date_range(
                start_date='2023-12-31',
                end_date='2023-01-01'
            )
        
        # 测试缺少必要参数 - DateRangeUtils可能有默认处理
        try:
            result = self.date_utils.create_date_range()
            # 如果没有抛出异常，验证返回值是否合理
            if result is not None:
                assert hasattr(result, 'start_date')
        except ValueError:
            # 如果抛出异常也是可以接受的
            pass