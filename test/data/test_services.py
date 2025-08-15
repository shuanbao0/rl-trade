"""
测试数据服务层
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.data.services import DataFetcher, DataValidator, DateRangeFetcher, DataValidationResult
from src.data.sources.base import DataSource, DataInterval
from src.utils.config import Config
from src.utils.date_range_utils import DateRange


class TestDataFetcher:
    """测试数据获取服务"""
    
    def setup_method(self):
        """设置测试"""
        self.config = Config()
        self.mock_data_source = MagicMock()
        self.data_fetcher = DataFetcher(
            config=self.config,
            data_source=self.mock_data_source
        )
    
    def test_fetch_data_success(self):
        """测试成功获取数据"""
        # 模拟数据源返回数据
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        self.mock_data_source.fetch_historical_data.return_value = mock_data
        
        result = self.data_fetcher.fetch_data('AAPL', '1y', '1d')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        self.mock_data_source.fetch_historical_data.assert_called_once()
    
    def test_fetch_data_by_date_range(self):
        """测试按日期范围获取数据"""
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        self.mock_data_source.fetch_historical_data.return_value = mock_data
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = self.data_fetcher.fetch_data_by_date_range('AAPL', start_date, end_date, '1d')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
    
    def test_fetch_data_with_retry(self):
        """测试重试机制"""
        # 模拟第一次失败，第二次成功
        mock_data = pd.DataFrame({
            'Close': [100]
        }, index=pd.date_range('2023-01-01', periods=1))
        
        self.mock_data_source.fetch_historical_data.side_effect = [
            Exception("Network error"),
            mock_data
        ]
        
        result = self.data_fetcher.fetch_data('AAPL', '1d', '1d')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert self.mock_data_source.fetch_historical_data.call_count == 2


class TestDataValidator:
    """测试数据验证服务"""
    
    def setup_method(self):
        """设置测试"""
        self.data_validator = DataValidator()
    
    def test_validate_valid_data(self):
        """测试验证有效数据"""
        valid_data = pd.DataFrame({
            'Open': range(100, 115),
            'High': range(105, 120),
            'Low': range(95, 110),
            'Close': range(103, 118),
            'Volume': range(1000, 2500, 100)
        }, index=pd.date_range('2023-01-01', periods=15))
        
        result = self.data_validator.validate_data(valid_data)
        
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is True
        assert result.records_count == 15
        assert len(result.issues) == 0
    
    def test_validate_insufficient_data(self):
        """测试验证数据不足"""
        insufficient_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        result = self.data_validator.validate_data(insufficient_data)
        
        assert result.is_valid is False
        assert result.records_count == 3
        assert any('Insufficient records' in issue for issue in result.issues)
    
    def test_validate_data_with_missing_values(self):
        """测试验证含缺失值的数据"""
        data_with_missing = pd.DataFrame({
            'Open': [100, None, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'Close': [103, 104, None, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        }, index=pd.date_range('2023-01-01', periods=13))
        
        result = self.data_validator.validate_data(data_with_missing)
        
        assert result.records_count == 13
        assert result.missing_values > 0
    
    def test_clean_data(self):
        """测试数据清洗"""
        dirty_data = pd.DataFrame({
            'Open': [100, None, 102, 100],  # 包含缺失值和重复行
            'Close': [103, 104, 105, 103]
        }, index=pd.date_range('2023-01-01', periods=4))
        
        cleaned_data = self.data_validator.clean_data(dirty_data)
        
        assert isinstance(cleaned_data, pd.DataFrame)
        # 检查缺失值是否被处理
        assert cleaned_data.isnull().sum().sum() == 0
    
    def test_data_quality_score(self):
        """测试数据质量评分"""
        good_data = pd.DataFrame({
            'Open': range(100, 120),
            'High': range(105, 125),
            'Low': range(95, 115),
            'Close': range(103, 123),
            'Volume': range(1000, 3000, 100)
        }, index=pd.date_range('2023-01-01', periods=20))
        
        quality_scores = self.data_validator.get_data_quality_score(good_data)
        
        assert 'overall_score' in quality_scores
        assert 'completeness_score' in quality_scores
        assert 'consistency_score' in quality_scores
        assert 'accuracy_score' in quality_scores
        
        # 好的数据应该有高质量分数
        assert quality_scores['overall_score'] > 0.8
        assert quality_scores['completeness_score'] > 0.9


class TestDateRangeFetcher:
    """测试日期范围获取服务"""
    
    def setup_method(self):
        """设置测试"""
        self.config = Config()
        self.mock_data_fetcher = MagicMock()
        self.mock_cache_manager = MagicMock()
        self.mock_batch_downloader = MagicMock()
        self.mock_market_detector = MagicMock()
        
        self.date_range_fetcher = DateRangeFetcher(
            config=self.config,
            data_fetcher=self.mock_data_fetcher,
            cache_manager=self.mock_cache_manager,
            batch_downloader=self.mock_batch_downloader,
            market_detector=self.mock_market_detector
        )
    
    def test_fetch_data_by_date_range(self):
        """测试按日期范围获取数据"""
        # 模拟市场类型检测
        from src.data.sources.base import MarketType
        self.mock_market_detector.detect.return_value = MarketType.STOCK
        
        # 模拟缓存未命中
        self.mock_cache_manager.get.return_value = None
        
        # 模拟数据获取
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        self.mock_data_fetcher.fetch_data_by_date_range.return_value = mock_data
        
        # 模拟批次下载不需要
        self.mock_batch_downloader.should_use_batch_download.return_value = False
        
        result = self.date_range_fetcher.fetch_data_by_date_range(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-03',
            interval='1d'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # 验证调用
        self.mock_market_detector.detect.assert_called_with('AAPL')
        self.mock_data_fetcher.fetch_data_by_date_range.assert_called_once()
        self.mock_cache_manager.put.assert_called_once()
    
    def test_get_date_range_estimation(self):
        """测试获取日期范围估算"""
        estimation = self.date_range_fetcher.get_date_range_estimation(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            interval='1d'
        )
        
        assert 'date_range' in estimation
        assert 'validation' in estimation
        assert 'recommendations' in estimation
        assert 'batch_download' in estimation
        
        # 验证日期范围信息
        date_range_info = estimation['date_range']
        assert date_range_info['duration_days'] >= 364  # Account for leap year calculation differences
    
    def test_should_use_batch_download_logic(self):
        """测试批次下载判断逻辑"""
        # 模拟长时间范围，应该推荐批次下载
        long_range = DateRange(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        self.mock_batch_downloader.should_use_batch_download.return_value = True
        
        should_use_batch = self.date_range_fetcher._should_use_batch_download(long_range, '1d')
        
        assert should_use_batch is True
        self.mock_batch_downloader.should_use_batch_download.assert_called_with(long_range, '1d')
    
    def test_convert_period_to_date_range(self):
        """测试周期到日期范围转换"""
        date_range = self.date_range_fetcher.convert_period_to_date_range('1y')
        
        assert isinstance(date_range, DateRange)
        assert date_range.duration_days == 365
        assert isinstance(date_range.start_date, datetime)
        assert isinstance(date_range.end_date, datetime)