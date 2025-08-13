"""
测试实时数据管理器模块
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from src.realtime.real_time_data_manager import RealTimeDataManager, MarketData, DataFeed
from src.utils.config import Config


class TestMarketData:
    def test_market_data_creation(self):
        """测试市场数据创建"""
        data = MarketData(
            symbol="AAPL",
            price=150.0,
            volume=1000,
            timestamp=datetime.now(),
            bid=149.5,
            ask=150.5
        )
        
        assert data.symbol == "AAPL"
        assert data.price == 150.0
        assert data.volume == 1000
        assert data.bid == 149.5
        assert data.ask == 150.5


class TestDataFeed:
    def test_data_feed_creation(self):
        """测试数据源创建"""
        feed = DataFeed(
            name="test_feed",
            url="ws://test.example.com",
            symbols=["AAPL", "MSFT"],
            reconnect_interval=5,
            max_reconnect_attempts=10
        )
        
        assert feed.name == "test_feed"
        assert feed.url == "ws://test.example.com"
        assert "AAPL" in feed.symbols
        assert "MSFT" in feed.symbols
        assert feed.reconnect_interval == 5
        assert feed.max_reconnect_attempts == 10


class TestRealTimeDataManager:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.data_manager = RealTimeDataManager(
            buffer_size=1000,
            max_latency_ms=100.0
        )
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        if hasattr(self.data_manager, 'stop'):
            self.data_manager.stop()
    
    def test_data_manager_initialization(self):
        """测试数据管理器初始化"""
        assert self.data_manager.buffer_size == 1000
        assert self.data_manager.max_latency_ms == 100.0
        assert len(self.data_manager.data_feeds) == 0
        assert len(self.data_manager.data_buffer) == 0
    
    def test_add_data_feed(self):
        """测试添加数据源"""
        feed = DataFeed(
            name="test_feed",
            url="ws://test.com",
            symbols=["AAPL"]
        )
        
        self.data_manager.add_data_feed(feed)
        
        assert len(self.data_manager.data_feeds) == 1
        assert "test_feed" in self.data_manager.data_feeds
        assert self.data_manager.data_feeds["test_feed"] == feed
    
    def test_remove_data_feed(self):
        """测试移除数据源"""
        feed = DataFeed(name="test_feed", url="ws://test.com", symbols=["AAPL"])
        self.data_manager.add_data_feed(feed)
        
        result = self.data_manager.remove_data_feed("test_feed")
        
        assert result is True
        assert len(self.data_manager.data_feeds) == 0
        assert "test_feed" not in self.data_manager.data_feeds
    
    def test_add_data_callback(self):
        """测试添加数据回调"""
        callback = MagicMock()
        
        self.data_manager.add_data_callback(callback)
        
        assert callback in self.data_manager.data_callbacks
    
    def test_add_error_callback(self):
        """测试添加错误回调"""
        error_callback = MagicMock()
        
        self.data_manager.add_error_callback(error_callback)
        
        assert error_callback in self.data_manager.error_callbacks
    
    def test_process_market_data(self):
        """测试处理市场数据"""
        # 添加数据回调
        callback = MagicMock()
        self.data_manager.add_data_callback(callback)
        
        # 创建测试数据
        market_data = MarketData(
            symbol="AAPL",
            price=150.0,
            volume=1000,
            timestamp=datetime.now(),
            bid=149.5,
            ask=150.5
        )
        
        # 处理数据
        self.data_manager._process_market_data(market_data)
        
        # 验证数据被存储
        assert len(self.data_manager.data_buffer) == 1
        assert "AAPL" in self.data_manager.latest_data
        assert self.data_manager.latest_data["AAPL"] == market_data
        
        # 验证回调被调用
        callback.assert_called_once()
    
    def test_get_latest_data(self):
        """测试获取最新数据"""
        market_data = MarketData(
            symbol="AAPL", price=150.0, volume=1000,
            timestamp=datetime.now(), bid=149.5, ask=150.5
        )
        
        self.data_manager._process_market_data(market_data)
        
        latest = self.data_manager.get_latest_data("AAPL")
        
        assert latest == market_data
        
        # 测试不存在的符号
        no_data = self.data_manager.get_latest_data("INVALID")
        assert no_data is None
    
    def test_get_data_frame(self):
        """测试获取数据框"""
        # 添加多个数据点
        for i in range(10):
            data = MarketData(
                symbol="AAPL",
                price=150.0 + i,
                volume=1000 + i * 100,
                timestamp=datetime.now(),
                bid=149.5 + i,
                ask=150.5 + i
            )
            self.data_manager._process_market_data(data)
        
        df = self.data_manager.get_data_frame("AAPL", window_size=5)
        
        assert df is not None
        assert len(df) <= 5  # 应该返回最新的5个数据点
        assert 'Close' in df.columns
        assert 'Volume' in df.columns
    
    def test_buffer_size_limit(self):
        """测试缓冲区大小限制"""
        small_manager = RealTimeDataManager(buffer_size=3)
        
        # 添加超过缓冲区大小的数据
        for i in range(5):
            data = MarketData(
                symbol="AAPL", price=150.0 + i, volume=1000,
                timestamp=datetime.now(), bid=149.5, ask=150.5
            )
            small_manager._process_market_data(data)
        
        # 缓冲区应该不超过限制
        assert len(small_manager.data_buffer) <= 3
    
    def test_latency_tracking(self):
        """测试延迟跟踪"""
        # 创建旧数据（高延迟）
        old_timestamp = datetime.now()
        # 手动设置时间戳为过去时间进行测试
        import time
        time.sleep(0.01)  # 等待一小段时间
        
        market_data = MarketData(
            symbol="AAPL", price=150.0, volume=1000,
            timestamp=old_timestamp, bid=149.5, ask=150.5
        )
        
        self.data_manager._process_market_data(market_data)
        
        # 验证延迟统计更新
        stats = self.data_manager.get_connection_status()
        assert 'latency_ms' in stats
    
    def test_connection_status(self):
        """测试连接状态"""
        status = self.data_manager.get_connection_status()
        
        assert isinstance(status, dict)
        assert 'running' in status
        assert 'data_feeds_count' in status
        assert 'buffer_size' in status
        assert 'latency_ms' in status
    
    def test_start_stop_data_manager(self):
        """测试启动和停止数据管理器"""
        # 启动
        self.data_manager.start()
        assert self.data_manager.is_running is True
        
        # 停止
        self.data_manager.stop()
        assert self.data_manager.is_running is False
    
    def test_error_handling(self):
        """测试错误处理"""
        error_callback = MagicMock()
        self.data_manager.add_error_callback(error_callback)
        
        # 触发错误
        test_error = Exception("Test error")
        self.data_manager._handle_error("test_feed", test_error)
        
        # 验证错误回调被调用
        error_callback.assert_called_once_with("test_feed", test_error)