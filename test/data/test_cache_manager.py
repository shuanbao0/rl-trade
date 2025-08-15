#!/usr/bin/env python3
"""
测试增强缓存管理器功能
"""

import unittest
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.data.cache_manager import (
    EnhancedCacheManager, 
    CacheKey, 
    CacheEntry, 
    LRUCache,
    get_cache_manager
)
from src.data.sources.base import MarketType


class TestCacheKey(unittest.TestCase):
    """缓存键测试类"""
    
    def test_cache_key_creation(self):
        """测试缓存键创建"""
        cache_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d"
        )
        
        self.assertEqual(cache_key.source, "yfinance")
        self.assertEqual(cache_key.symbol, "AAPL")
        self.assertEqual(cache_key.market_type, MarketType.STOCK)
        self.assertEqual(cache_key.period, "1y")
        self.assertEqual(cache_key.interval, "1d")
    
    def test_cache_key_string_conversion(self):
        """测试缓存键字符串转换"""
        cache_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d"
        )
        
        key_string = cache_key.to_string()
        expected = "yfinance_AAPL_stock_1y_1d"
        self.assertEqual(key_string, expected)
    
    def test_cache_key_with_parameters(self):
        """测试带参数的缓存键"""
        cache_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d",
            parameters={"param1": "value1", "param2": "value2"}
        )
        
        key_string = cache_key.to_string()
        self.assertIn("param1=value1", key_string)
        self.assertIn("param2=value2", key_string)
    
    def test_cache_key_hash(self):
        """测试缓存键哈希"""
        cache_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d"
        )
        
        hash_key = cache_key.to_hash()
        self.assertIsInstance(hash_key, str)
        self.assertEqual(len(hash_key), 32)  # MD5 hash length
    
    def test_cache_key_from_string(self):
        """测试从字符串解析缓存键"""
        original_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d"
        )
        
        key_string = original_key.to_string()
        parsed_key = CacheKey.from_string(key_string)
        
        self.assertEqual(parsed_key.source, original_key.source)
        self.assertEqual(parsed_key.symbol, original_key.symbol)
        self.assertEqual(parsed_key.market_type, original_key.market_type)
        self.assertEqual(parsed_key.period, original_key.period)
        self.assertEqual(parsed_key.interval, original_key.interval)


class TestCacheEntry(unittest.TestCase):
    """缓存条目测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 99.0],
            'High': [102.0, 103.0, 101.0],
            'Low': [99.0, 100.0, 98.0],
            'Close': [101.0, 100.0, 102.0],
            'Volume': [1000, 1100, 900]
        })
        
        self.cache_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d"
        )
    
    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = CacheEntry(
            key=self.cache_key,
            data=self.test_data,
            timestamp=datetime.now(),
            ttl_hours=24,
            size_bytes=1000
        )
        
        self.assertEqual(entry.key, self.cache_key)
        self.assertTrue(entry.data.equals(self.test_data))
        self.assertEqual(entry.ttl_hours, 24)
        self.assertEqual(entry.size_bytes, 1000)
        self.assertEqual(entry.access_count, 0)
    
    def test_cache_entry_expiry(self):
        """测试缓存条目过期检查"""
        # 创建已过期的条目
        past_time = datetime.now() - timedelta(hours=25)
        entry = CacheEntry(
            key=self.cache_key,
            data=self.test_data,
            timestamp=past_time,
            ttl_hours=24,
            size_bytes=1000
        )
        
        self.assertTrue(entry.is_expired())
        
        # 创建未过期的条目
        recent_time = datetime.now() - timedelta(hours=1)
        entry2 = CacheEntry(
            key=self.cache_key,
            data=self.test_data,
            timestamp=recent_time,
            ttl_hours=24,
            size_bytes=1000
        )
        
        self.assertFalse(entry2.is_expired())
    
    def test_cache_entry_touch(self):
        """测试缓存条目访问更新"""
        entry = CacheEntry(
            key=self.cache_key,
            data=self.test_data,
            timestamp=datetime.now(),
            ttl_hours=24,
            size_bytes=1000
        )
        
        initial_count = entry.access_count
        entry.touch()
        
        self.assertEqual(entry.access_count, initial_count + 1)
        self.assertIsNotNone(entry.last_access)


class TestLRUCache(unittest.TestCase):
    """LRU缓存测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.cache = LRUCache(max_size=3)
        self.test_data = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        
        self.cache_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d"
        )
    
    def test_lru_cache_put_get(self):
        """测试LRU缓存存取"""
        entry = CacheEntry(
            key=self.cache_key,
            data=self.test_data,
            timestamp=datetime.now(),
            ttl_hours=24,
            size_bytes=1000
        )
        
        key_string = self.cache_key.to_string()
        
        # 存储数据
        self.cache.put(key_string, entry)
        
        # 获取数据
        retrieved_entry = self.cache.get(key_string)
        self.assertIsNotNone(retrieved_entry)
        self.assertEqual(retrieved_entry.access_count, 1)
    
    def test_lru_cache_eviction(self):
        """测试LRU缓存驱逐"""
        # 填充缓存至容量
        for i in range(4):
            key = f"test_key_{i}"
            entry = CacheEntry(
                key=self.cache_key,
                data=self.test_data,
                timestamp=datetime.now(),
                ttl_hours=24,
                size_bytes=1000
            )
            self.cache.put(key, entry)
        
        # 验证最旧的条目被驱逐
        self.assertIsNone(self.cache.get("test_key_0"))
        self.assertIsNotNone(self.cache.get("test_key_3"))
    
    def test_lru_cache_stats(self):
        """测试LRU缓存统计"""
        stats = self.cache.get_stats()
        
        self.assertIn('entries', stats)
        self.assertIn('max_size', stats)
        self.assertIn('total_size_bytes', stats)
        self.assertIn('utilization', stats)


class TestEnhancedCacheManager(unittest.TestCase):
    """增强缓存管理器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        
        config = {
            'cache_directory': self.temp_dir,
            'memory_cache_size': 10,
            'default_ttl_hours': 24
        }
        
        self.cache_manager = EnhancedCacheManager(config)
        
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 99.0],
            'High': [102.0, 103.0, 101.0],
            'Low': [99.0, 100.0, 98.0],
            'Close': [101.0, 100.0, 102.0],
            'Volume': [1000, 1100, 900]
        })
        
        self.cache_key = CacheKey(
            source="yfinance",
            symbol="AAPL",
            market_type=MarketType.STOCK,
            period="1y",
            interval="1d"
        )
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_manager_put_get(self):
        """测试缓存管理器存取"""
        # 测试缓存未命中
        cached_data = self.cache_manager.get(self.cache_key)
        self.assertIsNone(cached_data)
        
        # 存储数据
        success = self.cache_manager.put(self.cache_key, self.test_data)
        self.assertTrue(success)
        
        # 测试缓存命中
        cached_data = self.cache_manager.get(self.cache_key)
        self.assertIsNotNone(cached_data)
        self.assertTrue(self.test_data.equals(cached_data))
    
    def test_cache_manager_disk_persistence(self):
        """测试缓存管理器磁盘持久化"""
        # 存储数据
        self.cache_manager.put(self.cache_key, self.test_data)
        
        # 创建新的缓存管理器实例（模拟重启）
        config = {
            'cache_directory': self.temp_dir,
            'memory_cache_size': 10,
            'default_ttl_hours': 24
        }
        new_cache_manager = EnhancedCacheManager(config)
        
        # 应该能从磁盘加载数据
        cached_data = new_cache_manager.get(self.cache_key)
        self.assertIsNotNone(cached_data)
        self.assertTrue(self.test_data.equals(cached_data))
    
    def test_cache_manager_remove(self):
        """测试缓存管理器删除"""
        # 存储数据
        self.cache_manager.put(self.cache_key, self.test_data)
        
        # 验证数据存在
        cached_data = self.cache_manager.get(self.cache_key)
        self.assertIsNotNone(cached_data)
        
        # 删除数据
        removed = self.cache_manager.remove(self.cache_key)
        self.assertTrue(removed)
        
        # 验证数据被删除
        cached_data = self.cache_manager.get(self.cache_key)
        self.assertIsNone(cached_data)
    
    def test_cache_manager_clear(self):
        """测试缓存管理器清理"""
        # 存储多个数据
        for i in range(3):
            key = CacheKey(
                source="yfinance",
                symbol=f"TEST{i}",
                market_type=MarketType.STOCK,
                period="1y",
                interval="1d"
            )
            self.cache_manager.put(key, self.test_data)
        
        # 清理所有缓存
        self.cache_manager.clear()
        
        # 验证所有数据被清理
        for i in range(3):
            key = CacheKey(
                source="yfinance",
                symbol=f"TEST{i}",
                market_type=MarketType.STOCK,
                period="1y",
                interval="1d"
            )
            cached_data = self.cache_manager.get(key)
            self.assertIsNone(cached_data)
    
    def test_cache_manager_statistics(self):
        """测试缓存管理器统计"""
        # 存储一些数据
        self.cache_manager.put(self.cache_key, self.test_data)
        
        # 获取统计信息
        stats = self.cache_manager.get_statistics()
        
        self.assertIn('memory_cache', stats)
        self.assertIn('disk_cache', stats)
        self.assertIn('hit_rate', stats)
        self.assertIn('total_hits', stats)
        self.assertIn('total_misses', stats)
    
    def test_cache_manager_cleanup_expired(self):
        """测试缓存管理器过期清理"""
        # 创建短TTL的缓存键（0.001小时 = 3.6秒）
        short_ttl_config = {
            'cache_directory': self.temp_dir,
            'memory_cache_size': 10,
            'default_ttl_hours': 0.001,  # 非常短的TTL
            'market_specific_ttl': {
                'stock': 0.001
            }
        }
        
        short_ttl_manager = EnhancedCacheManager(short_ttl_config)
        
        # 存储数据
        short_ttl_manager.put(self.cache_key, self.test_data)
        
        # 等待过期
        import time
        time.sleep(4)  # 等待4秒
        
        # 清理过期缓存
        cleaned_count = short_ttl_manager.cleanup_expired()
        
        # 验证数据被清理
        cached_data = short_ttl_manager.get(self.cache_key)
        self.assertIsNone(cached_data)


class TestGlobalCacheManager(unittest.TestCase):
    """全局缓存管理器测试类"""
    
    def test_get_global_cache_manager(self):
        """测试获取全局缓存管理器"""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        # 应该返回同一个实例
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, EnhancedCacheManager)


if __name__ == '__main__':
    unittest.main()