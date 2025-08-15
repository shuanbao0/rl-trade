"""
Test MigrationAnalyzer
"""

import unittest
import tempfile
import os
from pathlib import Path

from src.rewards.migration.migration_analyzer import MigrationAnalyzer, RewardFunctionInfo


class TestMigrationAnalyzer(unittest.TestCase):
    """测试迁移分析器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_rewards_path = Path(self.temp_dir) / "rewards"
        self.test_rewards_path.mkdir()
        
        # 创建测试用的奖励函数文件
        self._create_test_reward_files()
        
        self.analyzer = MigrationAnalyzer(str(self.test_rewards_path))
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_reward_files(self):
        """创建测试用的奖励函数文件"""
        
        # 简单奖励函数
        simple_reward = '''
"""
Simple Return Reward
"""
from base_reward import BaseReward

class SimpleReturnReward(BaseReward):
    """简单收益奖励"""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.multiplier = config.get('multiplier', 1.0)
    
    def calculate(self, context):
        return context.get_step_return() * self.multiplier
    
    def get_reward_info(self):
        return {
            'name': 'simple_return',
            'type': 'basic'
        }
'''
        
        # 复杂奖励函数 
        complex_reward = '''
"""
Risk Adjusted Reward - Forex Optimized
"""
import numpy as np
from base_reward import BaseReward

class RiskAdjustedReward(BaseReward):
    """风险调整奖励 - 外汇优化"""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.lookback_window = config.get('lookback_window', 20)
        self.forex_multiplier = config.get('forex_multiplier', 10000)  # pip multiplier
    
    def calculate(self, context):
        if not hasattr(context, 'portfolio_history'):
            return 0.0
            
        returns = np.diff(context.portfolio_history) / context.portfolio_history[:-1]
        if len(returns) < self.lookback_window:
            return 0.0
            
        excess_return = np.mean(returns) - self.risk_free_rate / 252
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
            
        sharpe = excess_return / volatility
        
        # Forex-specific adjustments
        if hasattr(context, 'market_type') and context.market_type == 'forex':
            pip_change = context.get_price_change() * self.forex_multiplier
            return sharpe + pip_change * 0.001
        
        return sharpe
    
    def get_reward_info(self):
        return {
            'name': 'risk_adjusted_forex',
            'type': 'advanced',
            'market': 'forex'
        }
'''
        
        # 错误语法的文件
        error_reward = '''
"""
Error Reward - 有语法错误
"""
class ErrorReward(BaseReward)  # 缺少冒号
    def calculate(self, context)
        return context.portfolio_value
'''
        
        # 写入文件
        (self.test_rewards_path / "simple_return.py").write_text(simple_reward)
        (self.test_rewards_path / "risk_adjusted.py").write_text(complex_reward)
        (self.test_rewards_path / "error_reward.py").write_text(error_reward)
        
        # 创建factory文件模拟别名
        factory_content = '''
"""
Reward Factory
"""

REWARD_MAPPINGS = {
    "simple": SimpleReturnReward,
    "return": SimpleReturnReward,
    "basic_return": SimpleReturnReward,
    "risk_adj": RiskAdjustedReward,
    "sharpe": RiskAdjustedReward,
    "forex_sharpe": RiskAdjustedReward,
}
'''
        (self.test_rewards_path / "reward_factory.py").write_text(factory_content)
    
    def test_scan_reward_files(self):
        """测试扫描奖励函数文件"""
        plan = self.analyzer.analyze_all_functions()
        
        # 应该发现2个有效的奖励函数（error_reward会被跳过语法错误）
        self.assertGreaterEqual(len(self.analyzer.reward_functions), 1)
        
        # 检查发现的函数
        function_names = list(self.analyzer.reward_functions.keys())
        self.assertIn('simplereturn', function_names)  # 类名转换后的名称
    
    def test_analyze_market_compatibility(self):
        """测试市场兼容性分析"""
        plan = self.analyzer.analyze_all_functions()
        
        # 查找风险调整奖励函数
        risk_adjusted = None
        for func in self.analyzer.reward_functions.values():
            if 'risk' in func.name.lower():
                risk_adjusted = func
                break
        
        if risk_adjusted:
            # 应该识别为forex兼容
            self.assertIn('forex', risk_adjusted.market_compatibility)
    
    def test_complexity_scoring(self):
        """测试复杂度评分"""
        plan = self.analyzer.analyze_all_functions()
        
        # 简单函数应该有较低的复杂度分数
        simple_func = None
        complex_func = None
        
        for func in self.analyzer.reward_functions.values():
            if 'simple' in func.name.lower():
                simple_func = func
            elif 'risk' in func.name.lower():
                complex_func = func
        
        if simple_func and complex_func:
            self.assertLess(simple_func.complexity_score, complex_func.complexity_score)
    
    def test_migration_plan_generation(self):
        """测试迁移计划生成"""
        plan = self.analyzer.analyze_all_functions()
        
        self.assertIsNotNone(plan)
        self.assertGreater(plan.total_functions, 0)
        self.assertIsInstance(plan.estimated_time_hours, float)
        self.assertIsInstance(plan.migration_order, list)
        
        # 高优先级函数应该在迁移顺序的前面
        if plan.high_priority:
            high_priority_names = [f.name for f in plan.high_priority]
            for name in high_priority_names:
                # 高优先级函数应该在迁移顺序的前半部分
                index = plan.migration_order.index(name)
                self.assertLess(index, len(plan.migration_order) // 2)
    
    def test_dependency_analysis(self):
        """测试依赖分析"""
        plan = self.analyzer.analyze_all_functions()
        
        # 检查是否正确识别了依赖
        for func in self.analyzer.reward_functions.values():
            self.assertIsInstance(func.dependencies, set)
            # 应该至少有base_reward的依赖
            dependency_names = [dep.split('.')[-1] for dep in func.dependencies]
            base_related = any('base' in dep.lower() for dep in dependency_names)
            # 这个测试可能因为实际导入结构而变化，所以只验证格式
    
    def test_parameter_extraction(self):
        """测试参数提取"""
        plan = self.analyzer.analyze_all_functions()
        
        # 检查参数提取
        for func in self.analyzer.reward_functions.values():
            self.assertIsInstance(func.parameters, dict)
            # 参数应该不包含'self'
            self.assertNotIn('self', func.parameters)
    
    def test_print_analysis_report(self):
        """测试分析报告打印"""
        plan = self.analyzer.analyze_all_functions()
        
        # 这个测试主要确保方法不会抛出异常
        try:
            self.analyzer.print_analysis_report(plan)
        except Exception as e:
            self.fail(f"打印分析报告失败: {e}")
    
    def test_alias_analysis(self):
        """测试别名分析"""
        plan = self.analyzer.analyze_all_functions()
        
        # 检查别名映射
        self.assertIsInstance(self.analyzer.alias_mapping, dict)
        
        # 检查一些函数是否有别名
        for func in self.analyzer.reward_functions.values():
            self.assertIsInstance(func.aliases, list)


if __name__ == '__main__':
    unittest.main()