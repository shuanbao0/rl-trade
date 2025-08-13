"""
奖励函数性能优化模块

提供内存使用优化、计算加速和资源管理功能，
确保奖励函数系统在生产环境中的高效运行。
"""

import gc
import sys
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from collections import deque
from functools import wraps, lru_cache
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    computation_time: float
    memory_usage: float
    cache_hit_rate: float
    operations_per_second: float

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb: float = 512.0):
        self.max_memory_mb = max_memory_mb
        self.memory_pools = {}
        self.gc_threshold = 0.8  # 80%内存使用时触发垃圾回收
        
    def get_memory_usage_mb(self) -> float:
        """获取当前内存使用量(MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # 如果没有psutil，使用sys.getsizeof的近似估算
            return sys.getsizeof(self.memory_pools) / 1024 / 1024
    
    def should_cleanup(self) -> bool:
        """判断是否需要清理内存"""
        current_usage = self.get_memory_usage_mb()
        return current_usage > (self.max_memory_mb * self.gc_threshold)
    
    def cleanup_memory(self):
        """清理内存"""
        if self.should_cleanup():
            # 清理内存池
            for pool_name in list(self.memory_pools.keys()):
                if len(self.memory_pools[pool_name]) > 100:  # 保留最近100个
                    self.memory_pools[pool_name] = deque(
                        list(self.memory_pools[pool_name])[-100:], 
                        maxlen=self.memory_pools[pool_name].maxlen
                    )
            
            # 强制垃圾回收
            gc.collect()
            
            logger.debug(f"Memory cleanup completed. Usage: {self.get_memory_usage_mb():.2f}MB")
    
    def get_memory_pool(self, pool_name: str, maxlen: int = 1000) -> deque:
        """获取或创建内存池"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = deque(maxlen=maxlen)
        return self.memory_pools[pool_name]

class ComputationCache:
    """计算结果缓存"""
    
    def __init__(self, max_cache_size: int = 10000):
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        try:
            # 将numpy数组转换为可哈希的元组
            def make_hashable(obj):
                if isinstance(obj, np.ndarray):
                    return ('ndarray', obj.shape, tuple(obj.flatten()[:10]))  # 只使用前10个元素避免过长
                elif isinstance(obj, (list, tuple)):
                    return tuple(make_hashable(item) for item in obj)
                elif isinstance(obj, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                else:
                    return obj
            
            hashable_args = make_hashable(args)
            hashable_kwargs = make_hashable(kwargs)
            
            return f"{func_name}_{hash((hashable_args, hashable_kwargs))}"
        except:
            # 如果生成哈希失败，返回基于函数名的简单键
            return f"{func_name}_{hash(str(args)[:100])}"
    
    def get(self, key: str):
        """从缓存获取结果"""
        if key in self.cache:
            self.hit_count += 1
            self.access_times[key] = time.time()
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def put(self, key: str, value):
        """将结果放入缓存"""
        if len(self.cache) >= self.max_cache_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """淘汰最近最少使用的缓存项"""
        if not self.access_times:
            return
        
        # 找到最旧的访问时间
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # 删除最旧的项
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, enable_cache: bool = True, enable_memory_management: bool = True):
        self.enable_cache = enable_cache
        self.enable_memory_management = enable_memory_management
        
        self.memory_manager = MemoryManager() if enable_memory_management else None
        self.computation_cache = ComputationCache() if enable_cache else None
        
        # 性能监控
        self.performance_history = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def cached_computation(self, func: Callable) -> Callable:
        """计算结果缓存装饰器"""
        if not self.enable_cache:
            return func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self.computation_cache.get_cache_key(func.__name__, args, kwargs)
            
            # 尝试从缓存获取
            cached_result = self.computation_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 计算结果
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # 缓存结果
            try:
                self.computation_cache.put(cache_key, result)
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
            
            # 记录性能
            self._record_performance(computation_time)
            
            return result
        
        return wrapper
    
    def memory_managed(self, func: Callable) -> Callable:
        """内存管理装饰器"""
        if not self.enable_memory_management:
            return func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 执行前检查内存
            if self.memory_manager.should_cleanup():
                self.memory_manager.cleanup_memory()
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 执行后再次检查内存
            if self.memory_manager.should_cleanup():
                self.memory_manager.cleanup_memory()
            
            return result
        
        return wrapper
    
    def optimized(self, func: Callable) -> Callable:
        """综合优化装饰器"""
        # 应用所有优化
        optimized_func = func
        
        if self.enable_memory_management:
            optimized_func = self.memory_managed(optimized_func)
        
        if self.enable_cache:
            optimized_func = self.cached_computation(optimized_func)
        
        return optimized_func
    
    def _record_performance(self, computation_time: float):
        """记录性能指标"""
        with self._lock:
            memory_usage = self.memory_manager.get_memory_usage_mb() if self.memory_manager else 0.0
            cache_hit_rate = self.computation_cache.get_hit_rate() if self.computation_cache else 0.0
            ops_per_second = 1.0 / computation_time if computation_time > 0 else 0.0
            
            metrics = PerformanceMetrics(
                computation_time=computation_time,
                memory_usage=memory_usage,
                cache_hit_rate=cache_hit_rate,
                operations_per_second=ops_per_second
            )
            
            self.performance_history.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {'status': 'no_data'}
        
        metrics_list = list(self.performance_history)
        
        avg_computation_time = np.mean([m.computation_time for m in metrics_list])
        avg_memory_usage = np.mean([m.memory_usage for m in metrics_list])
        current_cache_hit_rate = metrics_list[-1].cache_hit_rate
        avg_ops_per_second = np.mean([m.operations_per_second for m in metrics_list])
        
        return {
            'average_computation_time_ms': avg_computation_time * 1000,
            'average_memory_usage_mb': avg_memory_usage,
            'cache_hit_rate': current_cache_hit_rate,
            'average_ops_per_second': avg_ops_per_second,
            'total_measurements': len(metrics_list),
            'memory_manager_enabled': self.enable_memory_management,
            'cache_enabled': self.enable_cache
        }
    
    def optimize_reward_function(self, reward_class):
        """为奖励函数类添加优化"""
        # 优化主要的计算方法
        if hasattr(reward_class, 'get_reward'):
            reward_class.get_reward = self.optimized(reward_class.get_reward)
        
        if hasattr(reward_class, 'reward'):
            reward_class.reward = self.optimized(reward_class.reward)
        
        # 优化其他计算密集的方法
        computation_methods = [
            '_calculate_profit_reward',
            '_calculate_risk_reward', 
            '_calculate_consistency_reward',
            '_calculate_efficiency_reward',
            '_calculate_sharpe_ratio',
            '_calculate_sortino_ratio',
            '_calculate_uncertainty',
            '_calculate_curiosity',
            '_calculate_causal_effect'
        ]
        
        for method_name in computation_methods:
            if hasattr(reward_class, method_name):
                original_method = getattr(reward_class, method_name)
                optimized_method = self.optimized(original_method)
                setattr(reward_class, method_name, optimized_method)
        
        return reward_class
    
    def clear_all_caches(self):
        """清空所有缓存"""
        if self.computation_cache:
            self.computation_cache.clear()
        
        if self.memory_manager:
            self.memory_manager.memory_pools.clear()
        
        logger.info("All caches cleared")

# 全局性能优化器实例
global_optimizer = PerformanceOptimizer()

def enable_performance_optimization(reward_class):
    """启用奖励函数性能优化的装饰器"""
    return global_optimizer.optimize_reward_function(reward_class)

def get_global_performance_summary() -> Dict[str, Any]:
    """获取全局性能摘要"""
    return global_optimizer.get_performance_summary()

def clear_global_caches():
    """清空全局缓存"""
    global_optimizer.clear_all_caches()

# Numpy计算优化
class NumpyOptimizer:
    """NumPy计算优化"""
    
    @staticmethod
    def optimized_mean(arr: np.ndarray) -> float:
        """优化的均值计算"""
        if len(arr) == 0:
            return 0.0
        return np.mean(arr)
    
    @staticmethod
    def optimized_std(arr: np.ndarray) -> float:
        """优化的标准差计算"""
        if len(arr) <= 1:
            return 0.0
        return np.std(arr, ddof=1)
    
    @staticmethod
    def optimized_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """优化的夏普比率计算"""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # 日化风险收益率
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    @staticmethod
    def optimized_max_drawdown(values: np.ndarray) -> float:
        """优化的最大回撤计算"""
        if len(values) <= 1:
            return 0.0
        
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown)
    
    @staticmethod
    def rolling_calculation(arr: np.ndarray, window: int, func: Callable) -> np.ndarray:
        """滚动窗口计算优化"""
        if len(arr) < window:
            return np.array([func(arr)])
        
        result = []
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            result.append(func(window_data))
        
        return np.array(result)

# 内存使用优化的数据结构
class OptimizedHistory:
    """优化的历史数据存储"""
    
    def __init__(self, maxlen: int = 1000, dtype: np.dtype = np.float32):
        self.maxlen = maxlen
        self.dtype = dtype
        self._data = np.zeros(maxlen, dtype=dtype)
        self._length = 0
        self._index = 0
    
    def append(self, value: float):
        """添加新值"""
        self._data[self._index] = value
        self._index = (self._index + 1) % self.maxlen
        if self._length < self.maxlen:
            self._length += 1
    
    def get_data(self) -> np.ndarray:
        """获取当前数据"""
        if self._length < self.maxlen:
            return self._data[:self._length].copy()
        else:
            # 重新排列以保持时间顺序
            return np.concatenate([
                self._data[self._index:],
                self._data[:self._index]
            ])
    
    def get_recent(self, n: int) -> np.ndarray:
        """获取最近n个值"""
        data = self.get_data()
        return data[-n:] if len(data) >= n else data
    
    def __len__(self) -> int:
        return self._length

def optimize_all_reward_functions():
    """为所有奖励函数启用性能优化"""
    try:
        from . import (
            RiskAdjustedReward, SimpleReturnReward, ProfitLossReward, DiversifiedReward,
            LogSharpeReward, ReturnDrawdownReward, DynamicSortinoReward, RegimeAwareReward,
            ExpertCommitteeReward, UncertaintyAwareReward, CuriosityDrivenReward,
            SelfRewardingReward, CausalReward, LLMGuidedReward, CurriculumReward
        )
        
        reward_classes = [
            RiskAdjustedReward, SimpleReturnReward, ProfitLossReward, DiversifiedReward,
            LogSharpeReward, ReturnDrawdownReward, DynamicSortinoReward, RegimeAwareReward,
            ExpertCommitteeReward, UncertaintyAwareReward, CuriosityDrivenReward,
            SelfRewardingReward, CausalReward, LLMGuidedReward, CurriculumReward
        ]
        
        for reward_class in reward_classes:
            try:
                global_optimizer.optimize_reward_function(reward_class)
                logger.debug(f"Optimized {reward_class.__name__}")
            except Exception as e:
                logger.warning(f"Failed to optimize {reward_class.__name__}: {e}")
        
        logger.info("Performance optimization enabled for all reward functions")
        
    except Exception as e:
        logger.error(f"Failed to optimize reward functions: {e}")

if __name__ == "__main__":
    # 测试性能优化
    optimizer = PerformanceOptimizer()
    
    @optimizer.optimized
    def test_computation(data: np.ndarray) -> float:
        """测试计算函数"""
        return np.mean(data ** 2) + np.std(data)
    
    # 运行测试
    test_data = np.random.random(1000)
    
    for i in range(100):
        result = test_computation(test_data)
    
    summary = optimizer.get_performance_summary()
    print("Performance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")