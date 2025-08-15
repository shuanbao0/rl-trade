"""
Test CompatibilityMapper
"""

import unittest
import numpy as np

from src.rewards.core.reward_context import RewardContext
from src.rewards.migration.compatibility_mapper import CompatibilityMapper, ContextMapping, MethodMapping


class TestCompatibilityMapper(unittest.TestCase):
    """测试兼容性映射器"""
    
    def setUp(self):
        """设置测试环境"""
        self.mapper = CompatibilityMapper()
    
    def test_context_mapping_creation(self):
        """测试上下文映射创建"""
        mapping = ContextMapping('old_attr', 'new_attr')
        
        self.assertEqual(mapping.old_attribute, 'old_attr')
        self.assertEqual(mapping.new_attribute, 'new_attr')
        self.assertIsNone(mapping.converter)
        self.assertIsNone(mapping.default_value)
    
    def test_method_mapping_creation(self):
        """测试方法映射创建"""
        mapping = MethodMapping('old_method', 'new_method')
        
        self.assertEqual(mapping.old_method, 'old_method')
        self.assertEqual(mapping.new_method, 'new_method')
        self.assertIsNone(mapping.parameter_mapping)
        self.assertIsNone(mapping.return_converter)
    
    def test_map_context_from_dict(self):
        """测试从字典映射上下文"""
        old_context = {
            'portfolio_value': 10000.0,
            'action': 0.5,
            'current_price': 1.2345,
            'step': 100,
            'portfolio_val': 9999.0  # 旧属性名
        }
        
        new_context = self.mapper.map_context(old_context)
        
        self.assertIsInstance(new_context, RewardContext)
        self.assertEqual(new_context.portfolio_value, 10000.0)
        self.assertEqual(new_context.action, 0.5)
        self.assertEqual(new_context.current_price, 1.2345)
        self.assertEqual(new_context.step, 100)
    
    def test_map_context_from_object(self):
        """测试从对象映射上下文"""
        class OldContext:
            def __init__(self):
                self.portfolio_value = 11000.0
                self.action = -0.3
                self.price = 1.5432
                self.current_step = 200
        
        old_context = OldContext()
        new_context = self.mapper.map_context(old_context)
        
        self.assertIsInstance(new_context, RewardContext)
        self.assertEqual(new_context.portfolio_value, 11000.0)
        self.assertEqual(new_context.action, -0.3)
        # price应该映射到current_price
        self.assertEqual(new_context.current_price, 1.5432)
        # current_step应该映射到step
        self.assertEqual(new_context.step, 200)
    
    def test_map_context_existing_new_format(self):
        """测试映射已经是新格式的上下文"""
        existing_context = RewardContext(
            portfolio_value=12000.0,
            action=0.8,
            current_price=2.0,
            step=300
        )
        
        mapped_context = self.mapper.map_context(existing_context)
        
        # 应该返回相同的对象
        self.assertIs(mapped_context, existing_context)
    
    def test_map_context_with_defaults(self):
        """测试使用默认值映射上下文"""
        old_context = {
            'action': 0.2
            # 缺少其他必需字段
        }
        
        new_context = self.mapper.map_context(old_context)
        
        self.assertEqual(new_context.action, 0.2)
        # 应该使用默认值
        self.assertEqual(new_context.portfolio_value, 10000.0)
        self.assertEqual(new_context.current_price, 1.0)
        self.assertEqual(new_context.step, 0)
    
    def test_map_method_call(self):
        """测试映射方法调用"""
        # 测试已知的方法映射
        new_method, args, kwargs = self.mapper.map_method_call('get_reward_info', 'arg1', key='value')
        
        self.assertEqual(new_method, 'get_info')
        self.assertEqual(args, ('arg1',))
        self.assertEqual(kwargs, {'key': 'value'})
        
        # 测试未知方法（应该保持不变）
        new_method, args, kwargs = self.mapper.map_method_call('unknown_method', 'arg1', key='value')
        
        self.assertEqual(new_method, 'unknown_method')
        self.assertEqual(args, ('arg1',))
        self.assertEqual(kwargs, {'key': 'value'})
    
    def test_map_reward_type_direct(self):
        """测试直接别名映射"""
        # 测试直接映射
        self.assertEqual(self.mapper.map_reward_type('simple'), 'simple_return')
        self.assertEqual(self.mapper.map_reward_type('sharpe'), 'sharpe_ratio')
        self.assertEqual(self.mapper.map_reward_type('forex'), 'forex_optimized')
    
    def test_map_reward_type_fuzzy(self):
        """测试模糊匹配"""
        # 测试关键词匹配
        self.assertEqual(self.mapper.map_reward_type('forex_return'), 'forex_optimized')
        self.assertEqual(self.mapper.map_reward_type('risk_based'), 'risk_adjusted')
        self.assertEqual(self.mapper.map_reward_type('my_sharpe_ratio'), 'sharpe_ratio')
        
        # 测试未知类型
        self.assertEqual(self.mapper.map_reward_type('unknown_type'), 'unknown_type')
    
    def test_create_compatibility_wrapper(self):
        """测试创建兼容性包装器"""
        
        # 创建一个简单的新奖励类
        from src.rewards.core.base_reward import BaseReward
        
        class NewReward(BaseReward):
            def calculate(self, context: RewardContext) -> float:
                return context.portfolio_value * 0.001
            
            def get_info(self):
                return {'name': 'new_reward', 'type': 'test'}
        
        # 创建包装器
        CompatibleReward = self.mapper.create_compatibility_wrapper(NewReward)
        
        # 测试包装器
        reward = CompatibleReward(self.mapper)
        
        # 测试新API仍然工作
        context = RewardContext(portfolio_value=10000.0, action=0.0, current_price=1.0, step=0)
        result = reward.calculate(context)
        self.assertEqual(result, 10.0)  # 10000 * 0.001
        
        # 测试旧API兼容性
        old_context = {
            'portfolio_value': 20000.0,
            'action': 0.5,
            'current_price': 1.0,
            'step': 10
        }
        
        result = reward.compute_reward(old_context)
        self.assertEqual(result, 20.0)  # 20000 * 0.001
        
        result = reward.calculate_reward(old_context)
        self.assertEqual(result, 20.0)
        
        result = reward.get_reward(old_context)
        self.assertEqual(result, 20.0)
        
        # 测试信息方法
        info = reward.get_reward_info()
        self.assertEqual(info['name'], 'new_reward')
    
    def test_alias_mappings(self):
        """测试别名映射"""
        # 检查一些关键的别名映射
        self.assertIn('simple', self.mapper.alias_mappings)
        self.assertIn('return', self.mapper.alias_mappings)
        self.assertIn('sharpe', self.mapper.alias_mappings)
        self.assertIn('forex', self.mapper.alias_mappings)
        
        # 检查映射值
        self.assertEqual(self.mapper.alias_mappings['simple'], 'simple_return')
        self.assertEqual(self.mapper.alias_mappings['sharpe'], 'sharpe_ratio')
    
    def test_parameter_mappings(self):
        """测试参数映射"""
        # 检查参数映射
        self.assertIn('portfolio_val', self.mapper.parameter_mappings)
        self.assertIn('current_step', self.mapper.parameter_mappings)
        
        # 检查映射值
        self.assertEqual(self.mapper.parameter_mappings['portfolio_val'], 'portfolio_value')
        self.assertEqual(self.mapper.parameter_mappings['current_step'], 'step')
    
    def test_generate_migration_guide(self):
        """测试生成迁移指南"""
        guide = self.mapper.generate_migration_guide()
        
        self.assertIsInstance(guide, str)
        self.assertIn("奖励函数迁移指南", guide)
        self.assertIn("上下文对象", guide)
        self.assertIn("方法名称变化", guide)
        self.assertIn("参数名称变化", guide)
        self.assertIn("兼容性包装器", guide)
    
    def test_validate_compatibility(self):
        """测试验证兼容性"""
        
        # 创建模拟的旧奖励类
        class OldReward:
            def compute_reward(self, context):
                return 1.0
            
            def get_reward_info(self):
                return {}
        
        # 创建模拟的新奖励类 
        class NewReward:
            def calculate(self, context):
                return 1.0
            
            def get_info(self):
                return {}
        
        result = self.mapper.validate_compatibility(OldReward, NewReward)
        
        self.assertIsInstance(result, dict)
        self.assertIn('compatible', result)
        self.assertIn('warnings', result)
        self.assertIn('errors', result)
        
        # 新奖励类应该有必需的方法
        self.assertTrue(result['compatible'])
        
        # 应该有兼容性警告（缺少compute_reward等）
        self.assertGreater(len(result['warnings']), 0)


if __name__ == '__main__':
    unittest.main()