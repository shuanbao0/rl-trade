"""
兼容性映射器 - 处理新旧API之间的兼容性映射
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from ..core.reward_context import RewardContext


@dataclass
class ContextMapping:
    """上下文映射规则"""
    old_attribute: str
    new_attribute: str
    converter: Optional[Callable] = None
    default_value: Any = None


@dataclass
class MethodMapping:
    """方法映射规则"""
    old_method: str
    new_method: str
    parameter_mapping: Dict[str, str] = None
    return_converter: Optional[Callable] = None


class CompatibilityMapper:
    """兼容性映射器 - 处理新旧API的兼容性"""
    
    def __init__(self):
        # 上下文属性映射
        self.context_mappings = {
            # 旧属性名 -> 新属性名
            'portfolio_value': ContextMapping('portfolio_value', 'portfolio_value'),
            'action': ContextMapping('action', 'action'),
            'current_price': ContextMapping('current_price', 'current_price'),
            'step': ContextMapping('step', 'step'),
            'timestamp': ContextMapping('timestamp', 'timestamp'),
            
            # 可能的旧属性名变体
            'portfolio_val': ContextMapping('portfolio_val', 'portfolio_value'),
            'current_step': ContextMapping('current_step', 'step'),
            'price': ContextMapping('price', 'current_price'),
            'obs': ContextMapping('obs', 'observation'),
            
            # 历史数据映射
            'portfolio_history': ContextMapping('portfolio_history', 'portfolio_history'),
            'price_history': ContextMapping('price_history', 'price_history'),
            'action_history': ContextMapping('action_history', 'action_history'),
        }
        
        # 方法映射
        self.method_mappings = {
            # 旧方法名 -> 新方法名
            'get_reward_info': MethodMapping('get_reward_info', 'get_info'),
            'compute_reward': MethodMapping('compute_reward', 'calculate'),
            'calculate_reward': MethodMapping('calculate_reward', 'calculate'),
            'get_reward': MethodMapping('get_reward', 'calculate'),
            
            # 上下文相关方法
            'get_return': MethodMapping('get_return', 'get_step_return'),
            'get_portfolio_return': MethodMapping('get_portfolio_return', 'get_return_pct'),
            'get_price_change': MethodMapping('get_price_change', 'get_price_change'),
        }
        
        # 参数映射
        self.parameter_mappings = {
            # 旧参数名 -> 新参数名
            'portfolio_val': 'portfolio_value',
            'current_step': 'step',
            'obs': 'observation',
            'reward_config': 'config',
            'env_config': 'config',
        }
        
        # 别名映射（从旧的别名系统）
        self.alias_mappings = self._create_alias_mappings()
    
    def _create_alias_mappings(self) -> Dict[str, str]:
        """创建别名映射"""
        # 这里包含了从旧系统中的150+别名映射
        # 简化版本，实际使用时需要从分析器获取完整映射
        return {
            # 基础奖励类型
            'simple': 'simple_return',
            'return': 'simple_return', 
            'profit': 'profit_loss',
            'pnl': 'profit_loss',
            'sharpe': 'sharpe_ratio',
            'sortino': 'sortino_ratio',
            
            # 风险调整类型
            'risk_adj': 'risk_adjusted',
            'risk_adjusted': 'risk_adjusted',
            'vol_adj': 'volatility_adjusted',
            
            # 外汇专用
            'fx_return': 'forex_return',
            'forex': 'forex_optimized',
            'pip_based': 'pip_optimized',
            
            # 高级类型
            'log_return': 'log_return',
            'compound': 'compound_return',
            'momentum': 'momentum_based',
            'mean_revert': 'mean_reversion',
            
            # AI增强类型
            'curiosity': 'curiosity_driven',
            'uncertainty': 'uncertainty_aware',
            'self_reward': 'self_rewarding',
            'expert': 'expert_committee',
        }
    
    def map_context(self, old_context: Any) -> RewardContext:
        """映射旧上下文到新上下文"""
        if isinstance(old_context, RewardContext):
            return old_context  # 已经是新格式
            
        # 从旧格式创建新上下文
        context_kwargs = {}
        
        # 处理字典格式的旧上下文
        if isinstance(old_context, dict):
            for old_key, value in old_context.items():
                if old_key in self.context_mappings:
                    mapping = self.context_mappings[old_key]
                    new_key = mapping.new_attribute
                    
                    # 应用转换器
                    if mapping.converter:
                        value = mapping.converter(value)
                    
                    context_kwargs[new_key] = value
                else:
                    # 尝试参数映射
                    new_key = self.parameter_mappings.get(old_key, old_key)
                    context_kwargs[new_key] = value
        
        # 处理对象格式的旧上下文
        elif hasattr(old_context, '__dict__'):
            for old_key, value in old_context.__dict__.items():
                if old_key in self.context_mappings:
                    mapping = self.context_mappings[old_key]
                    new_key = mapping.new_attribute
                    
                    if mapping.converter:
                        value = mapping.converter(value)
                    
                    context_kwargs[new_key] = value
                else:
                    new_key = self.parameter_mappings.get(old_key, old_key)
                    context_kwargs[new_key] = value
        
        # 设置默认值
        if 'portfolio_value' not in context_kwargs:
            context_kwargs['portfolio_value'] = 10000.0
        if 'action' not in context_kwargs:
            context_kwargs['action'] = 0.0
        if 'current_price' not in context_kwargs:
            context_kwargs['current_price'] = 1.0
        if 'step' not in context_kwargs:
            context_kwargs['step'] = 0
            
        return RewardContext(**context_kwargs)
    
    def map_method_call(self, old_method_name: str, *args, **kwargs) -> tuple:
        """映射旧方法调用到新方法"""
        if old_method_name in self.method_mappings:
            mapping = self.method_mappings[old_method_name]
            new_method_name = mapping.new_method
            
            # 映射参数
            if mapping.parameter_mapping:
                new_kwargs = {}
                for old_param, new_param in mapping.parameter_mapping.items():
                    if old_param in kwargs:
                        new_kwargs[new_param] = kwargs[old_param]
                    else:
                        new_kwargs[old_param] = kwargs.get(old_param)
                kwargs = new_kwargs
            
            return new_method_name, args, kwargs
        
        return old_method_name, args, kwargs
    
    def map_reward_type(self, old_reward_type: str) -> str:
        """映射奖励类型别名"""
        # 标准化输入
        old_reward_type = old_reward_type.lower().strip()
        
        # 直接映射
        if old_reward_type in self.alias_mappings:
            return self.alias_mappings[old_reward_type]
        
        # 模糊匹配
        for alias, canonical in self.alias_mappings.items():
            if alias in old_reward_type or old_reward_type in alias:
                return canonical
        
        # 关键词匹配
        if 'forex' in old_reward_type or 'fx' in old_reward_type:
            return 'forex_optimized'
        elif 'risk' in old_reward_type:
            return 'risk_adjusted'
        elif 'sharpe' in old_reward_type:
            return 'sharpe_ratio'
        elif 'return' in old_reward_type:
            return 'simple_return'
        
        # 如果找不到匹配，返回原值
        return old_reward_type
    
    def create_compatibility_wrapper(self, new_reward_class):
        """创建兼容性包装器类"""
        
        class CompatibilityWrapper(new_reward_class):
            """兼容性包装器 - 使新奖励函数兼容旧API"""
            
            def __init__(self, mapper: CompatibilityMapper, **config):
                # 映射配置参数
                mapped_config = {}
                for key, value in config.items():
                    new_key = mapper.parameter_mappings.get(key, key)
                    mapped_config[new_key] = value
                
                super().__init__(**mapped_config)
                self.mapper = mapper
            
            def compute_reward(self, old_context):
                """兼容旧的compute_reward方法"""
                new_context = self.mapper.map_context(old_context)
                return self.calculate(new_context)
            
            def calculate_reward(self, old_context):
                """兼容旧的calculate_reward方法"""
                return self.compute_reward(old_context)
            
            def get_reward(self, old_context):
                """兼容旧的get_reward方法"""
                return self.compute_reward(old_context)
            
            def get_reward_info(self):
                """兼容旧的get_reward_info方法"""
                return self.get_info()
            
            def __call__(self, old_context):
                """支持直接调用"""
                if hasattr(super(), '__call__'):
                    new_context = self.mapper.map_context(old_context)
                    return super().__call__(new_context)
                else:
                    return self.compute_reward(old_context)
        
        return CompatibilityWrapper
    
    def generate_migration_guide(self) -> str:
        """生成迁移指南"""
        guide = """
# 奖励函数迁移指南

## 主要变化

### 1. 上下文对象 (Context Object)
```python
# 旧方式
def calculate_reward(self, portfolio_val, action, price, step):
    return portfolio_val * 0.01

# 新方式  
def calculate(self, context: RewardContext) -> float:
    return context.portfolio_value * 0.01
```

### 2. 方法名称变化
```python
# 旧方法名 -> 新方法名
get_reward_info() -> get_info()
compute_reward() -> calculate()
calculate_reward() -> calculate()
```

### 3. 参数名称变化
"""
        
        guide += "```python\n# 参数映射:\n"
        for old_param, new_param in self.parameter_mappings.items():
            guide += f"'{old_param}' -> '{new_param}'\n"
        guide += "```\n\n"
        
        guide += "### 4. 奖励类型别名映射\n```python\n"
        for alias, canonical in list(self.alias_mappings.items())[:20]:  # 显示前20个
            guide += f"'{alias}' -> '{canonical}'\n"
        guide += "# ... 还有更多别名\n```\n\n"
        
        guide += """
### 5. 使用兼容性包装器
```python
from src.rewards.migration.compatibility_mapper import CompatibilityMapper

# 创建映射器
mapper = CompatibilityMapper()

# 包装新奖励函数以兼容旧API
CompatibleReward = mapper.create_compatibility_wrapper(NewRewardClass)
reward = CompatibleReward(mapper)

# 现在可以使用旧API
result = reward.compute_reward(old_context)
```

### 6. 迁移检查清单
- [ ] 更新方法名: get_reward_info -> get_info
- [ ] 更新参数传递: 单独参数 -> RewardContext对象  
- [ ] 更新别名引用: 检查reward_type是否需要映射
- [ ] 测试兼容性: 确保新函数通过所有旧测试
- [ ] 更新文档: 反映API变化
"""
        
        return guide
    
    def validate_compatibility(self, old_reward_class, new_reward_class) -> Dict[str, Any]:
        """验证兼容性"""
        validation_result = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'missing_methods': [],
            'parameter_mismatches': []
        }
        
        # 检查必需方法
        required_old_methods = ['calculate', 'get_info']
        for method in required_old_methods:
            if not hasattr(new_reward_class, method):
                validation_result['missing_methods'].append(method)
                validation_result['compatible'] = False
        
        # 检查兼容性方法
        compat_methods = ['compute_reward', 'calculate_reward', 'get_reward_info']
        for method in compat_methods:
            if hasattr(old_reward_class, method) and not hasattr(new_reward_class, method):
                validation_result['warnings'].append(f"缺少兼容性方法: {method}")
        
        return validation_result