"""
Reward Registry - 全局奖励函数注册中心
管理所有奖励函数的注册、发现和创建
"""

import logging
from typing import Dict, List, Type, Optional, Callable, Any
from functools import wraps
from .base_reward import BaseReward


class RewardRegistry:
    """
    全局奖励函数注册中心
    
    使用单例模式，提供统一的奖励函数管理：
    - 注册/注销奖励函数
    - 别名管理
    - 分类管理
    - 市场类型和时间粒度适配
    - 动态发现和加载
    """
    
    _instance: Optional['RewardRegistry'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 主要注册表
        self._rewards: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        
        # 市场和粒度适配
        self._market_compatibility: Dict[str, List[str]] = {}
        self._granularity_compatibility: Dict[str, List[str]] = {}
        
        # 工厂函数注册
        self._factories: Dict[str, Callable] = {}
        
        # 性能统计
        self._usage_stats: Dict[str, int] = {}
        
        # 使用统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger("RewardRegistry")
        self._initialized = True
        
        self.logger.info("奖励函数注册中心初始化完成")
    
    def register(self, 
                 reward_class: Type[BaseReward],
                 name: str = None,
                 aliases: List[str] = None,
                 category: str = "general",
                 description: str = None,
                 market_types: List[str] = None,
                 granularities: List[str] = None,
                 priority: int = 1,
                 **metadata) -> None:
        """
        注册奖励函数
        
        Args:
            reward_class: 奖励函数类
            name: 注册名称（默认使用类名）
            aliases: 别名列表
            category: 分类
            description: 描述
            market_types: 适用的市场类型
            granularities: 适用的时间粒度
            priority: 优先级（用于同名冲突解决）
            **metadata: 额外元数据
        """
        if not issubclass(reward_class, BaseReward):
            raise ValueError(f"奖励函数类必须继承自BaseReward: {reward_class}")
        
        name = name or reward_class.__name__
        
        # 检查重复注册
        if name in self._rewards:
            existing_priority = self._rewards[name].get('priority', 1)
            if priority <= existing_priority:
                self.logger.warning(f"奖励函数 {name} 已存在且优先级更高，跳过注册")
                return
        
        # 主要注册
        self._rewards[name] = {
            'class': reward_class,
            'category': category,
            'description': description or f"{reward_class.__name__} reward function",
            'market_types': market_types or ['all'],
            'granularities': granularities or ['all'],
            'priority': priority,
            'metadata': metadata
        }
        
        # 别名注册
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    self.logger.warning(f"别名 {alias} 已存在，将被覆盖")
                self._aliases[alias] = name
        
        # 分类索引
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        # 市场类型索引
        for market_type in (market_types or ['all']):
            if market_type not in self._market_compatibility:
                self._market_compatibility[market_type] = []
            if name not in self._market_compatibility[market_type]:
                self._market_compatibility[market_type].append(name)
        
        # 时间粒度索引
        for granularity in (granularities or ['all']):
            if granularity not in self._granularity_compatibility:
                self._granularity_compatibility[granularity] = []
            if name not in self._granularity_compatibility[granularity]:
                self._granularity_compatibility[granularity].append(name)
        
        # 初始化使用统计
        self._usage_stats[name] = 0
        
        self.logger.info(f"注册奖励函数: {name} "
                        f"(别名: {aliases}, 分类: {category}, "
                        f"市场: {market_types}, 粒度: {granularities})")
    
    def unregister(self, name: str) -> bool:
        """
        注销奖励函数
        
        Args:
            name: 奖励函数名称
            
        Returns:
            bool: 是否成功注销
        """
        if name not in self._rewards:
            return False
        
        # 移除主要注册
        reward_info = self._rewards.pop(name)
        
        # 移除别名
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        # 移除分类索引
        category = reward_info['category']
        if category in self._categories and name in self._categories[category]:
            self._categories[category].remove(name)
            if not self._categories[category]:
                del self._categories[category]
        
        # 移除市场类型索引
        for market_type, names in self._market_compatibility.items():
            if name in names:
                names.remove(name)
        
        # 移除时间粒度索引
        for granularity, names in self._granularity_compatibility.items():
            if name in names:
                names.remove(name)
        
        # 移除统计
        self._usage_stats.pop(name, None)
        
        self.logger.info(f"注销奖励函数: {name}")
        return True
    
    def get(self, name: str) -> Type[BaseReward]:
        """
        获取奖励函数类
        
        Args:
            name: 奖励函数名称或别名
            
        Returns:
            Type[BaseReward]: 奖励函数类
            
        Raises:
            KeyError: 当奖励函数不存在时
        """
        # 解析别名
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._rewards:
            available = list(self._rewards.keys()) + list(self._aliases.keys())
            raise KeyError(f"未注册的奖励函数: {name}. 可用函数: {available}")
        
        # 更新使用统计
        self._usage_stats[actual_name] = self._usage_stats.get(actual_name, 0) + 1
        
        return self._rewards[actual_name]['class']
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """
        获取奖励函数详细信息
        
        Args:
            name: 奖励函数名称或别名
            
        Returns:
            Dict: 奖励函数信息
        """
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._rewards:
            raise KeyError(f"未注册的奖励函数: {name}")
        
        info = self._rewards[actual_name].copy()
        info['name'] = actual_name
        info['usage_count'] = self._usage_stats.get(actual_name, 0)
        
        # 获取别名
        info['aliases'] = [alias for alias, target in self._aliases.items() if target == actual_name]
        
        return info
    
    def list_all(self) -> List[str]:
        """列出所有已注册的奖励函数"""
        return list(self._rewards.keys())
    
    def list_aliases(self) -> Dict[str, str]:
        """列出所有别名映射"""
        return self._aliases.copy()
    
    def list_by_category(self, category: str) -> List[str]:
        """
        按分类列出奖励函数
        
        Args:
            category: 分类名称
            
        Returns:
            List[str]: 该分类下的奖励函数列表
        """
        return self._categories.get(category, []).copy()
    
    def list_categories(self) -> List[str]:
        """列出所有分类"""
        return list(self._categories.keys())
    
    def list_by_market_type(self, market_type: str) -> List[str]:
        """
        按市场类型列出适用的奖励函数
        
        Args:
            market_type: 市场类型 (forex/stock/crypto/all)
            
        Returns:
            List[str]: 适用的奖励函数列表
        """
        compatible = set()
        
        # 添加通用函数（适用于所有市场）
        compatible.update(self._market_compatibility.get('all', []))
        
        # 添加特定市场函数
        compatible.update(self._market_compatibility.get(market_type, []))
        
        return list(compatible)
    
    def list_by_granularity(self, granularity: str) -> List[str]:
        """
        按时间粒度列出适用的奖励函数
        
        Args:
            granularity: 时间粒度 (1min/5min/1h/1d/all)
            
        Returns:
            List[str]: 适用的奖励函数列表
        """
        compatible = set()
        
        # 添加通用函数（适用于所有粒度）
        compatible.update(self._granularity_compatibility.get('all', []))
        
        # 添加特定粒度函数
        compatible.update(self._granularity_compatibility.get(granularity, []))
        
        return list(compatible)
    
    def find_optimal(self, 
                    market_type: str = None,
                    granularity: str = None,
                    category: str = None) -> List[str]:
        """
        查找最优奖励函数
        
        Args:
            market_type: 市场类型
            granularity: 时间粒度
            category: 分类
            
        Returns:
            List[str]: 按优先级排序的适用奖励函数列表
        """
        candidates = set(self._rewards.keys())
        
        # 按市场类型过滤
        if market_type:
            market_compatible = set(self.list_by_market_type(market_type))
            candidates &= market_compatible
        
        # 按时间粒度过滤
        if granularity:
            granularity_compatible = set(self.list_by_granularity(granularity))
            candidates &= granularity_compatible
        
        # 按分类过滤
        if category:
            category_compatible = set(self.list_by_category(category))
            candidates &= category_compatible
        
        # 按优先级和使用频率排序
        def sort_key(name):
            info = self._rewards[name]
            priority = info.get('priority', 1)
            usage = self._usage_stats.get(name, 0)
            return (-priority, -usage)  # 优先级高的在前，使用频率高的在前
        
        return sorted(candidates, key=sort_key)
    
    def register_factory(self, name: str, factory_func: Callable) -> None:
        """
        注册奖励函数工厂方法
        
        Args:
            name: 工厂名称
            factory_func: 工厂函数
        """
        self._factories[name] = factory_func
        self.logger.info(f"注册奖励函数工厂: {name}")
    
    def create_from_factory(self, factory_name: str, **kwargs):
        """
        使用工厂方法创建奖励函数
        
        Args:
            factory_name: 工厂名称
            **kwargs: 创建参数
            
        Returns:
            BaseReward: 奖励函数实例
        """
        if factory_name not in self._factories:
            raise KeyError(f"未注册的工厂方法: {factory_name}")
        
        return self._factories[factory_name](**kwargs)
    
    def get_usage_stats(self) -> Dict[str, int]:
        """获取使用统计"""
        return self._usage_stats.copy()
    
    def get_popular_rewards(self, top_n: int = 10) -> List[tuple]:
        """
        获取最受欢迎的奖励函数
        
        Args:
            top_n: 返回前N个
            
        Returns:
            List[tuple]: (名称, 使用次数) 的列表
        """
        sorted_stats = sorted(self._usage_stats.items(), key=lambda x: x[1], reverse=True)
        return sorted_stats[:top_n]
    
    def clear_stats(self):
        """清除使用统计"""
        self._usage_stats.clear()
        self.logger.info("使用统计已清除")
    
    def export_registry(self) -> Dict[str, Any]:
        """导出注册信息"""
        return {
            'rewards': self._rewards,
            'aliases': self._aliases,
            'categories': self._categories,
            'market_compatibility': self._market_compatibility,
            'granularity_compatibility': self._granularity_compatibility,
            'usage_stats': self._usage_stats
        }
    
    def __len__(self) -> int:
        """返回已注册的奖励函数数量"""
        return len(self._rewards)
    
    def __contains__(self, name: str) -> bool:
        """检查奖励函数是否已注册"""
        return name in self._rewards or name in self._aliases
    
    def __iter__(self):
        """迭代所有已注册的奖励函数"""
        return iter(self._rewards.keys())


# 装饰器：自动注册奖励函数
def register_reward(name: str = None, 
                   aliases: List[str] = None,
                   category: str = "general",
                   market_types: List[str] = None,
                   granularities: List[str] = None,
                   **kwargs):
    """
    奖励函数自动注册装饰器
    
    使用示例:
    @register_reward("my_reward", aliases=["mr"], category="custom")
    class MyReward(BaseReward):
        pass
    """
    def decorator(reward_class):
        registry = RewardRegistry()
        registry.register(
            reward_class=reward_class,
            name=name,
            aliases=aliases,
            category=category,
            market_types=market_types,
            granularities=granularities,
            **kwargs
        )
        return reward_class
    
    return decorator


# 全局注册中心实例
_global_registry = RewardRegistry()

def get_global_registry() -> RewardRegistry:
    """获取全局注册中心实例"""
    return _global_registry