"""
Test MigrationExecutor
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from src.rewards.migration.migration_executor import MigrationExecutor, MigrationResult


class TestMigrationExecutor(unittest.TestCase):
    """测试迁移执行器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.old_rewards_path = Path(self.temp_dir) / "old_rewards"
        self.new_rewards_path = Path(self.temp_dir) / "new_rewards"
        
        self.old_rewards_path.mkdir()
        self.new_rewards_path.mkdir()
        
        # 创建测试用的奖励函数文件
        self._create_test_files()
        
        self.executor = MigrationExecutor(
            str(self.old_rewards_path),
            str(self.new_rewards_path)
        )
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_files(self):
        """创建测试文件"""
        
        # 简单奖励函数
        simple_reward = '''"""
Simple Return Reward
"""
from src.environment.rewards.base_reward import BaseReward

class SimpleReturnReward(BaseReward):
    """简单收益奖励"""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.multiplier = config.get('multiplier', 1.0)
    
    def calculate(self, context):
        return context.portfolio_value * 0.01
    
    def get_reward_info(self):
        return {
            'name': 'simple_return',
            'type': 'basic'
        }
'''
        
        (self.old_rewards_path / "simple_return.py").write_text(simple_reward)
        
        # 创建__init__.py
        (self.old_rewards_path / "__init__.py").write_text("")
    
    def test_create_directory_structure(self):
        """测试创建目录结构"""
        self.executor._create_directory_structure()
        
        # 检查目录是否创建
        expected_dirs = [
            self.new_rewards_path / "functions",
            self.new_rewards_path / "functions" / "basic",
            self.new_rewards_path / "functions" / "advanced",
            self.new_rewards_path / "functions" / "forex",
        ]
        
        for directory in expected_dirs:
            self.assertTrue(directory.exists(), f"目录未创建: {directory}")
            self.assertTrue((directory / "__init__.py").exists(), f"__init__.py未创建: {directory}")
    
    def test_determine_target_location(self):
        """测试确定目标位置"""
        from src.rewards.migration.migration_analyzer import RewardFunctionInfo
        
        # 简单函数应该放在basic目录
        simple_info = RewardFunctionInfo(
            name="simple_return",
            file_path="test.py", 
            class_name="SimpleReturnReward",
            base_class="BaseReward",
            dependencies=set(),
            market_compatibility={'forex', 'stock', 'crypto'},
            granularity_compatibility={'1min', '5min', '1h', '1d', '1w'},
            complexity_score=2,
            migration_priority=8,
            aliases=[],
            parameters={},
            imports=set()
        )
        
        target_dir, filename = self.executor._determine_target_location(simple_info)
        
        self.assertTrue("basic" in str(target_dir))
        self.assertEqual(filename, "simple_return_reward.py")
        
        # 特定市场函数应该放在对应目录
        forex_info = RewardFunctionInfo(
            name="forex_optimized",
            file_path="test.py",
            class_name="ForexOptimizedReward", 
            base_class="BaseReward",
            dependencies=set(),
            market_compatibility={'forex'},  # 只支持forex
            granularity_compatibility={'1min', '5min'},
            complexity_score=6,
            migration_priority=5,
            aliases=[],
            parameters={},
            imports=set()
        )
        
        target_dir, filename = self.executor._determine_target_location(forex_info)
        
        self.assertTrue("forex" in str(target_dir))
        self.assertEqual(filename, "forex_optimized_reward.py")
    
    def test_code_conversion(self):
        """测试代码转换"""
        from src.rewards.migration.migration_analyzer import RewardFunctionInfo
        
        original_code = '''from src.environment.rewards.base_reward import BaseReward

class TestReward(BaseReward):
    def calculate(self, context):
        return context.portfolio_value
        
    def get_reward_info(self):
        return {'name': 'test'}
'''
        
        reward_info = RewardFunctionInfo(
            name="test",
            file_path="test.py",
            class_name="TestReward",
            base_class="BaseReward",
            dependencies=set(),
            market_compatibility={'forex'},
            granularity_compatibility={'1min'},
            complexity_score=1,
            migration_priority=5,
            aliases=[],
            parameters={},
            imports=set()
        )
        
        converted_code, warnings = self.executor._convert_code(original_code, reward_info)
        
        # 检查导入是否被替换
        self.assertIn("from src.rewards.core.base_reward import BaseReward", converted_code)
        self.assertNotIn("from src.environment.rewards.base_reward import BaseReward", converted_code)
        
        # 检查是否添加了新的导入
        self.assertIn("from src.rewards.core.reward_context import RewardContext", converted_code)
        
        # 检查是否有警告
        self.assertIsInstance(warnings, list)
    
    def test_migration_result_creation(self):
        """测试迁移结果创建"""
        result = MigrationResult(
            function_name="test_function",
            source_file="source.py",
            target_file="target.py",
            success=True
        )
        
        self.assertEqual(result.function_name, "test_function")
        self.assertTrue(result.success)
        self.assertEqual(result.warnings, [])  # 默认为空列表
        
        # 测试失败情况
        failed_result = MigrationResult(
            function_name="failed_function",
            source_file="source.py", 
            target_file="target.py",
            success=False,
            error_message="测试错误"
        )
        
        self.assertFalse(failed_result.success)
        self.assertEqual(failed_result.error_message, "测试错误")
    
    def test_add_new_imports(self):
        """测试添加新导入"""
        original_code = '''"""
Test module
"""
import os
from typing import Dict

class TestReward:
    pass
'''
        
        converted_code = self.executor._add_new_imports(original_code)
        
        # 检查是否添加了新的导入
        self.assertIn("from src.rewards.core.reward_context import RewardContext", converted_code)
        self.assertIn("from src.rewards.core.base_reward import BaseReward", converted_code)
        
        # 检查原有导入是否保留
        self.assertIn("import os", converted_code)
        self.assertIn("from typing import Dict", converted_code)
    
    def test_add_metadata(self):
        """测试添加元数据"""
        from src.rewards.migration.migration_analyzer import RewardFunctionInfo
        
        original_code = '''class TestReward(BaseReward):
    def calculate(self, context):
        return 0.0
'''
        
        reward_info = RewardFunctionInfo(
            name="test",
            file_path="/path/to/test.py",
            class_name="TestReward",
            base_class="BaseReward",
            dependencies=set(),
            market_compatibility={'forex'},
            granularity_compatibility={'1min'},
            complexity_score=3,
            migration_priority=7,
            aliases=['test_alias'],
            parameters={},
            imports=set()
        )
        
        code_with_metadata = self.executor._add_metadata(original_code, reward_info)
        
        # 检查是否添加了元数据注释
        self.assertIn("自动迁移的奖励函数: TestReward", code_with_metadata)
        self.assertIn("原始文件: /path/to/test.py", code_with_metadata)
        self.assertIn("市场兼容性: forex", code_with_metadata)
        self.assertIn("复杂度: 3/10", code_with_metadata)
    
    def test_create_migration_script(self):
        """测试创建迁移脚本"""
        script_path = Path(self.temp_dir) / "test_migrate.py"
        
        self.executor.create_migration_script(str(script_path))
        
        # 检查脚本是否创建
        self.assertTrue(script_path.exists())
        
        # 检查脚本内容
        script_content = script_path.read_text(encoding='utf-8')
        self.assertIn("MigrationExecutor", script_content)
        self.assertIn("def main():", script_content)
        self.assertIn("if __name__ == \"__main__\":", script_content)


if __name__ == '__main__':
    unittest.main()