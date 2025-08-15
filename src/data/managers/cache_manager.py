"""
增强缓存管理器

提供基于多维度的智能缓存策略：
1. 基于(source, symbol, market_type, period, interval)的缓存键设计
2. 分层缓存架构（内存 + 文件 + 压缩）
3. 缓存失效策略和自动清理
4. 缓存统计和监控
5. 缓存预热和批量操作
"""

import os
import pickle
import gzip
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from threading import Lock, RLock
from collections import OrderedDict, defaultdict

from ..sources.base import MarketType, DataInterval
from ...utils.logger import setup_logger


@dataclass
class CacheKey:
    """缓存键结构"""
    source: str                         # 数据源
    symbol: str                         # 交易符号
    market_type: MarketType            # 市场类型
    period: str                        # 时间周期
    interval: str                      # 数据间隔
    parameters: Dict[str, Any] = field(default_factory=dict)  # 额外参数
    
    def __post_init__(self):
        """标准化键值"""
        self.source = self.source.lower()
        self.symbol = self.symbol.upper()
        
    def to_string(self) -> str:
        """转换为字符串键"""
        base_key = f"{self.source}_{self.symbol}_{self.market_type.value}_{self.period}_{self.interval}"
        
        if self.parameters:
            # 对参数进行排序确保一致性
            param_str = "_".join(f"{k}={v}" for k, v in sorted(self.parameters.items()))
            return f"{base_key}_{param_str}"
        
        return base_key
    
    def to_hash(self) -> str:
        """转换为哈希键（用于文件名）"""
        key_str = self.to_string()
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @classmethod
    def from_string(cls, key_str: str) -> 'CacheKey':
        """从字符串解析缓存键"""
        parts = key_str.split('_')
        if len(parts) < 5:
            raise ValueError(f"Invalid cache key format: {key_str}")
        
        source = parts[0]
        symbol = parts[1]
        market_type = MarketType(parts[2])
        period = parts[3]
        interval = parts[4]
        
        parameters = {}
        if len(parts) > 5:
            # 解析参数
            for param_part in parts[5:]:
                if '=' in param_part:
                    k, v = param_part.split('=', 1)
                    parameters[k] = v
        
        return cls(source, symbol, market_type, period, interval, parameters)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: CacheKey                      # 缓存键
    data: pd.DataFrame                 # 缓存数据
    timestamp: datetime                # 缓存时间
    ttl_hours: float                   # 生存时间（小时）
    size_bytes: int                    # 数据大小
    access_count: int = 0              # 访问次数
    last_access: Optional[datetime] = None  # 最后访问时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl_hours <= 0:
            return False  # 永不过期
        
        expiry_time = self.timestamp + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def touch(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'key': self.key.to_string(),
            'timestamp': self.timestamp.isoformat(),
            'ttl_hours': self.ttl_hours,
            'size_bytes': self.size_bytes,
            'access_count': self.access_count,
            'last_access': self.last_access.isoformat() if self.last_access else None,
            'metadata': self.metadata
        }


class LRUCache:
    """线程安全的LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # 检查是否过期
                if entry.is_expired():
                    del self._cache[key]
                    return None
                
                # 移动到末尾（最近使用）
                self._cache.move_to_end(key)
                entry.touch()
                return entry
            
            return None
    
    def put(self, key: str, entry: CacheEntry):
        """放入缓存条目"""
        with self._lock:
            if key in self._cache:
                # 更新现有条目
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # 添加新条目
                self._cache[key] = entry
                
                # 检查是否超出容量
                if len(self._cache) > self.max_size:
                    # 移除最旧的条目
                    self._cache.popitem(last=False)
    
    def remove(self, key: str) -> bool:
        """移除缓存条目"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            total_access = sum(entry.access_count for entry in self._cache.values())
            
            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'total_size_bytes': total_size,
                'total_access_count': total_access,
                'utilization': len(self._cache) / self.max_size if self.max_size > 0 else 0
            }
    
    def get_keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            return list(self._cache.keys())


class EnhancedCacheManager:
    """增强缓存管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化缓存管理器
        
        Args:
            config: 缓存配置
        """
        self.config = config or self._get_default_config()
        self.logger = setup_logger("CacheManager")
        
        # 初始化缓存层
        self.memory_cache = LRUCache(self.config['memory_cache_size'])
        self.cache_dir = Path(self.config['cache_directory'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存统计
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0
        }
        self._stats_lock = Lock()
        
        # TTL配置
        self.default_ttl = self.config['default_ttl_hours']
        self.market_specific_ttl = self.config.get('market_specific_ttl', {})
        
        # 压缩配置
        self.enable_compression = self.config.get('enable_compression', True)
        self.compression_threshold = self.config.get('compression_threshold_bytes', 1024 * 1024)  # 1MB
        
        self.logger.info(f"CacheManager initialized with directory: {self.cache_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'cache_directory': 'data_cache',
            'memory_cache_size': 1000,
            'default_ttl_hours': 24,
            'market_specific_ttl': {
                'stock': 24,      # 股票数据缓存24小时
                'forex': 1,       # 外汇数据缓存1小时
                'crypto': 0.5,    # 加密货币数据缓存30分钟
                'commodities': 12,  # 商品数据缓存12小时
                'index': 24,      # 指数数据缓存24小时
                'etf': 24         # ETF数据缓存24小时
            },
            'enable_compression': True,
            'compression_threshold_bytes': 1024 * 1024,  # 1MB
            'auto_cleanup_interval_hours': 24,
            'max_disk_cache_size_gb': 10
        }
    
    def get(self, cache_key: CacheKey) -> Optional[pd.DataFrame]:
        """
        获取缓存数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据或None
        """
        key_str = cache_key.to_string()
        
        # 1. 检查内存缓存
        entry = self.memory_cache.get(key_str)
        if entry is not None:
            self._record_hit()
            self.logger.debug(f"Memory cache hit: {key_str}")
            return entry.data.copy()
        
        # 2. 检查文件缓存
        disk_data = self._load_from_disk(cache_key)
        if disk_data is not None:
            self._record_hit()
            self.logger.debug(f"Disk cache hit: {key_str}")
            
            # 加载到内存缓存
            ttl = self._get_ttl_for_market(cache_key.market_type)
            entry = CacheEntry(
                key=cache_key,
                data=disk_data,
                timestamp=datetime.now(),
                ttl_hours=ttl,
                size_bytes=self._calculate_size(disk_data)
            )
            self.memory_cache.put(key_str, entry)
            
            return disk_data.copy()
        
        # 3. 缓存未命中
        self._record_miss()
        self.logger.debug(f"Cache miss: {key_str}")
        return None
    
    def put(self, cache_key: CacheKey, data: pd.DataFrame) -> bool:
        """
        存储数据到缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
            
        Returns:
            是否成功存储
        """
        try:
            key_str = cache_key.to_string()
            data_size = self._calculate_size(data)
            ttl = self._get_ttl_for_market(cache_key.market_type)
            
            # 创建缓存条目
            entry = CacheEntry(
                key=cache_key,
                data=data,
                timestamp=datetime.now(),
                ttl_hours=ttl,
                size_bytes=data_size
            )
            
            # 1. 存储到内存缓存
            self.memory_cache.put(key_str, entry)
            
            # 2. 存储到文件缓存
            success = self._save_to_disk(cache_key, data, entry)
            
            if success:
                self.logger.debug(f"Cached data for {key_str} (size: {data_size} bytes)")
                return True
            else:
                self.logger.warning(f"Failed to cache data to disk for {key_str}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cache data for {cache_key.to_string()}: {e}")
            return False
    
    def remove(self, cache_key: CacheKey) -> bool:
        """
        移除缓存条目
        
        Args:
            cache_key: 缓存键
            
        Returns:
            是否成功移除
        """
        key_str = cache_key.to_string()
        
        # 从内存缓存移除
        memory_removed = self.memory_cache.remove(key_str)
        
        # 从文件缓存移除
        disk_removed = self._remove_from_disk(cache_key)
        
        if memory_removed or disk_removed:
            self.logger.debug(f"Removed cache entry: {key_str}")
            return True
        
        return False
    
    def clear(self, pattern: Optional[str] = None):
        """
        清理缓存
        
        Args:
            pattern: 可选的模式匹配（如果为None则清理所有）
        """
        if pattern is None:
            # 清理所有缓存
            self.memory_cache.clear()
            self._clear_disk_cache()
            self.logger.info("Cleared all cache")
        else:
            # 模式匹配清理
            self._clear_by_pattern(pattern)
            self.logger.info(f"Cleared cache matching pattern: {pattern}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._stats_lock:
            memory_stats = self.memory_cache.get_stats()
            disk_stats = self._get_disk_cache_stats()
            
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'memory_cache': memory_stats,
                'disk_cache': disk_stats,
                'hit_rate': hit_rate,
                'total_hits': self._stats['hits'],
                'total_misses': self._stats['misses'],
                'total_requests': total_requests,
                'disk_reads': self._stats['disk_reads'],
                'disk_writes': self._stats['disk_writes'],
                'evictions': self._stats['evictions']
            }
    
    def cleanup_expired(self) -> int:
        """
        清理过期缓存
        
        Returns:
            清理的条目数量
        """
        cleaned_count = 0
        
        # 清理内存缓存中的过期条目
        memory_keys = self.memory_cache.get_keys()
        for key in memory_keys:
            entry = self.memory_cache.get(key)
            if entry and entry.is_expired():
                self.memory_cache.remove(key)
                cleaned_count += 1
        
        # 清理文件缓存中的过期条目
        cleaned_count += self._cleanup_expired_disk_cache()
        
        self.logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        return cleaned_count
    
    def preload(self, cache_keys: List[CacheKey]):
        """
        预加载缓存（用于缓存预热）
        
        Args:
            cache_keys: 要预加载的缓存键列表
        """
        self.logger.info(f"Preloading {len(cache_keys)} cache entries")
        
        for cache_key in cache_keys:
            # 检查是否已在内存中
            if self.memory_cache.get(cache_key.to_string()) is None:
                # 尝试从磁盘加载
                disk_data = self._load_from_disk(cache_key)
                if disk_data is not None:
                    ttl = self._get_ttl_for_market(cache_key.market_type)
                    entry = CacheEntry(
                        key=cache_key,
                        data=disk_data,
                        timestamp=datetime.now(),
                        ttl_hours=ttl,
                        size_bytes=self._calculate_size(disk_data)
                    )
                    self.memory_cache.put(cache_key.to_string(), entry)
    
    def _get_ttl_for_market(self, market_type: MarketType) -> float:
        """获取市场特定的TTL"""
        return self.market_specific_ttl.get(market_type.value, self.default_ttl)
    
    def _calculate_size(self, data: pd.DataFrame) -> int:
        """计算数据大小（字节）"""
        return data.memory_usage(deep=True).sum()
    
    def _get_file_path(self, cache_key: CacheKey, compressed: bool = False) -> Path:
        """获取文件路径"""
        hash_key = cache_key.to_hash()
        
        # 按市场类型分目录
        market_dir = self.cache_dir / cache_key.market_type.value
        market_dir.mkdir(exist_ok=True)
        
        # 按数据源再分目录
        source_dir = market_dir / cache_key.source
        source_dir.mkdir(exist_ok=True)
        
        extension = '.pkl.gz' if compressed else '.pkl'
        return source_dir / f"{hash_key}{extension}"
    
    def _save_to_disk(self, cache_key: CacheKey, data: pd.DataFrame, entry: CacheEntry) -> bool:
        """保存数据到磁盘"""
        try:
            # 决定是否压缩
            use_compression = (
                self.enable_compression and 
                entry.size_bytes > self.compression_threshold
            )
            
            file_path = self._get_file_path(cache_key, use_compression)
            
            # 准备元数据
            metadata = {
                'cache_entry': entry.to_dict(),
                'compression': use_compression,
                'save_time': datetime.now().isoformat()
            }
            
            if use_compression:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump({'data': data, 'metadata': metadata}, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump({'data': data, 'metadata': metadata}, f)
            
            with self._stats_lock:
                self._stats['disk_writes'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save to disk: {e}")
            return False
    
    def _load_from_disk(self, cache_key: CacheKey) -> Optional[pd.DataFrame]:
        """从磁盘加载数据"""
        try:
            # 尝试压缩和非压缩文件
            for compressed in [True, False]:
                file_path = self._get_file_path(cache_key, compressed)
                
                if file_path.exists():
                    if compressed:
                        with gzip.open(file_path, 'rb') as f:
                            cached_obj = pickle.load(f)
                    else:
                        with open(file_path, 'rb') as f:
                            cached_obj = pickle.load(f)
                    
                    # 检查缓存是否过期
                    metadata = cached_obj.get('metadata', {})
                    cache_entry_dict = metadata.get('cache_entry', {})
                    
                    if cache_entry_dict:
                        ttl_hours = cache_entry_dict.get('ttl_hours', self.default_ttl)
                        save_time_str = cache_entry_dict.get('timestamp')
                        
                        if save_time_str and ttl_hours > 0:
                            save_time = datetime.fromisoformat(save_time_str)
                            if datetime.now() > save_time + timedelta(hours=ttl_hours):
                                # 缓存已过期，删除文件
                                file_path.unlink()
                                continue
                    
                    with self._stats_lock:
                        self._stats['disk_reads'] += 1
                    
                    return cached_obj['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load from disk: {e}")
            return None
    
    def _remove_from_disk(self, cache_key: CacheKey) -> bool:
        """从磁盘移除缓存文件"""
        removed = False
        
        for compressed in [True, False]:
            file_path = self._get_file_path(cache_key, compressed)
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed = True
                except Exception as e:
                    self.logger.error(f"Failed to remove cache file {file_path}: {e}")
        
        return removed
    
    def _clear_disk_cache(self):
        """清理所有磁盘缓存"""
        try:
            for file_path in self.cache_dir.rglob("*.pkl*"):
                file_path.unlink()
        except Exception as e:
            self.logger.error(f"Failed to clear disk cache: {e}")
    
    def _clear_by_pattern(self, pattern: str):
        """按模式清理缓存"""
        # 简单的模式匹配实现
        # 支持通配符 * 和 ?
        import fnmatch
        
        # 清理内存缓存
        memory_keys = self.memory_cache.get_keys()
        for key in memory_keys:
            if fnmatch.fnmatch(key, pattern):
                self.memory_cache.remove(key)
        
        # 清理文件缓存
        for file_path in self.cache_dir.rglob("*.pkl*"):
            # 尝试从文件名匹配
            if fnmatch.fnmatch(file_path.stem, pattern):
                try:
                    file_path.unlink()
                except Exception:
                    pass
    
    def _get_disk_cache_stats(self) -> Dict[str, Any]:
        """获取磁盘缓存统计"""
        total_files = 0
        total_size = 0
        
        try:
            for file_path in self.cache_dir.rglob("*.pkl*"):
                total_files += 1
                total_size += file_path.stat().st_size
        except Exception as e:
            self.logger.error(f"Failed to get disk cache stats: {e}")
        
        return {
            'files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'directory': str(self.cache_dir)
        }
    
    def _cleanup_expired_disk_cache(self) -> int:
        """清理过期的磁盘缓存文件"""
        cleaned_count = 0
        
        try:
            for file_path in self.cache_dir.rglob("*.pkl*"):
                try:
                    # 尝试加载文件检查是否过期
                    if file_path.suffix == '.gz':
                        with gzip.open(file_path, 'rb') as f:
                            cached_obj = pickle.load(f)
                    else:
                        with open(file_path, 'rb') as f:
                            cached_obj = pickle.load(f)
                    
                    metadata = cached_obj.get('metadata', {})
                    cache_entry_dict = metadata.get('cache_entry', {})
                    
                    if cache_entry_dict:
                        ttl_hours = cache_entry_dict.get('ttl_hours', self.default_ttl)
                        save_time_str = cache_entry_dict.get('timestamp')
                        
                        if save_time_str and ttl_hours > 0:
                            save_time = datetime.fromisoformat(save_time_str)
                            if datetime.now() > save_time + timedelta(hours=ttl_hours):
                                file_path.unlink()
                                cleaned_count += 1
                
                except Exception:
                    # 如果文件损坏或无法读取，也删除它
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception:
                        pass
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired disk cache: {e}")
        
        return cleaned_count
    
    def _record_hit(self):
        """记录缓存命中"""
        with self._stats_lock:
            self._stats['hits'] += 1
    
    def _record_miss(self):
        """记录缓存未命中"""
        with self._stats_lock:
            self._stats['misses'] += 1


# 全局缓存管理器实例
_global_cache_manager: Optional[EnhancedCacheManager] = None


def get_cache_manager() -> EnhancedCacheManager:
    """获取全局缓存管理器实例"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = EnhancedCacheManager()
    return _global_cache_manager


def set_cache_manager(cache_manager: EnhancedCacheManager):
    """设置全局缓存管理器实例"""
    global _global_cache_manager
    _global_cache_manager = cache_manager