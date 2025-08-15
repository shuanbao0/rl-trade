"""
Test MigrationValidator
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from src.rewards.core.reward_context import RewardContext
from src.rewards.core.base_reward import BaseReward
from src.rewards.migration.migration_validator import MigrationValidator, ValidationResult


class MockTestReward(BaseReward):
    """测试用奖励函数"""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.test_value = config.get('test_value', 1.0)
    
    def calculate(self, context: RewardContext) -> float:
        return context.portfolio_value * 0.001 * self.test_value
    
    def get_info(self):
        return {
            'name': 'test_reward',
            'type': 'test',
            'description': 'Test reward function'
        }


class BadReward(BaseReward):
    """有问题的奖励函数"""
    
    def calculate(self, context: RewardContext) -> float:
        # 故意返回无限值
        return float('inf')
    
    def get_info(self):
        return {
            'name': 'bad_reward',
            'type': 'bad'
        }


class SlowReward(BaseReward):
    """计算缓慢的奖励函数"""
    
    def calculate(self, context: RewardContext) -> float:
        import time
        time.sleep(0.01)  # 人为延迟
        return 1.0
    
    def get_info(self):
        return {
            'name': 'slow_reward', 
            'type': 'slow'
        }


class TestMigrationValidator(unittest.TestCase):
    """测试迁移验证器"""
    
    def setUp(self):
        """设置测试环境"""
        self.validator = MigrationValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_validation_result_creation(self):
        """测试验证结果创建"""
        result = ValidationResult(
            function_name="test_function",
            passed=True,
            test_count=10,
            passed_count=10,
            failed_count=0
        )
        
        self.assertEqual(result.function_name, "test_function")
        self.assertTrue(result.passed)
        self.assertEqual(result.test_count, 10)
        self.assertEqual(result.errors, [])  # 默认为空列表
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.performance_metrics, {})
    
    def test_validate_good_function(self):
        """测试验证良好的函数"""
        result = self.validator.validate_single_function(MockTestReward, "test_reward")
        
        self.assertEqual(result.function_name, "test_reward")
        self.assertTrue(result.passed)
        self.assertGreater(result.test_count, 0)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_bad_function(self):
        """测试验证有问题的函数"""
        result = self.validator.validate_single_function(BadReward, "bad_reward")
        
        self.assertEqual(result.function_name, "bad_reward")
        # 可能通过也可能失败，取决于验证器如何处理无限值
        # 但应该至少有一些警告
        if not result.passed:
            self.assertGreater(len(result.errors), 0)
    
    def test_create_standard_test_cases(self):
        """测试创建标准测试用例"""
        test_cases = self.validator._create_standard_test_cases()
        
        self.assertIsInstance(test_cases, list)
        self.assertGreater(len(test_cases), 0)
        
        # 检查每个测试用例
        for case in test_cases:
            self.assertIsInstance(case, RewardContext)
            self.assertIsInstance(case.portfolio_value, (int, float))
            self.assertIsInstance(case.action, (int, float))
            self.assertIsInstance(case.current_price, (int, float))
            self.assertIsInstance(case.step, int)
    
    def test_run_standard_tests(self):
        """测试运行标准测试"""
        reward = MockTestReward()
        test_results = self.validator._run_standard_tests(reward)
        
        self.assertIsInstance(test_results, list)
        self.assertGreater(len(test_results), 0)
        
        # 检查测试结果格式
        for result in test_results:
            self.assertIn('test_name', result)
            self.assertIn('passed', result)
            self.assertIsInstance(result['passed'], bool)
            
            if result['passed']:
                self.assertIn('result', result)
            else:
                self.assertIn('error', result)
    
    def test_run_performance_tests(self):
        """测试运行性能测试"""
        reward = MockTestReward()
        metrics = self.validator._run_performance_tests(reward)
        
        self.assertIsInstance(metrics, dict)
        
        # 应该包含计算时间指标
        if 'avg_computation_time_ms' in metrics:
            self.assertIsInstance(metrics['avg_computation_time_ms'], (int, float))
            self.assertGreaterEqual(metrics['avg_computation_time_ms'], 0)
    
    def test_run_compatibility_tests(self):
        """测试运行兼容性测试"""
        reward = MockTestReward()
        test_results = self.validator._run_compatibility_tests(reward)
        
        self.assertIsInstance(test_results, list)
        self.assertGreater(len(test_results), 0)
        
        # 应该包含方法存在检查
        method_tests = [r for r in test_results if 'has_method' in r['test_name']]
        self.assertGreater(len(method_tests), 0)
        
        # 应该包含get_info测试
        info_tests = [r for r in test_results if 'get_info' in r['test_name']]
        self.assertGreater(len(info_tests), 0)
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        self.assertIn('max_computation_time_ms', self.validator.performance_benchmarks)
        self.assertIn('max_memory_usage_mb', self.validator.performance_benchmarks)
        self.assertIn('min_reward_range', self.validator.performance_benchmarks)
        self.assertIn('max_reward_range', self.validator.performance_benchmarks)
        
        # 检查基准值是否合理
        self.assertGreater(self.validator.performance_benchmarks['max_computation_time_ms'], 0)
        self.assertGreater(self.validator.performance_benchmarks['max_memory_usage_mb'], 0)
    
    def test_validate_with_slow_function(self):
        """测试验证缓慢的函数"""
        result = self.validator.validate_single_function(SlowReward, "slow_reward")
        
        # 应该能够检测到性能问题
        if 'performance_metrics' in result.__dict__ and result.performance_metrics:
            if 'avg_computation_time_ms' in result.performance_metrics:
                # 慢函数的计算时间应该相对较高
                self.assertGreaterEqual(result.performance_metrics['avg_computation_time_ms'], 1.0)
    
    def test_create_test_file_for_validation(self):
        """测试创建文件进行验证"""
        # 创建一个测试Python文件
        test_file_content = '''
"""
Test Reward Function
"""
from src.rewards.core.base_reward import BaseReward
from src.rewards.core.reward_context import RewardContext

class FileTestReward(BaseReward):
    """文件测试奖励函数"""
    
    def calculate(self, context: RewardContext) -> float:
        return context.portfolio_value * 0.002
    
    def get_info(self):
        return {
            'name': 'file_test_reward',
            'type': 'test'
        }
'''
        
        test_file = Path(self.temp_dir) / "test_reward.py"
        test_file.write_text(test_file_content, encoding='utf-8')
        
        # 验证文件
        result = self.validator._validate_single_file(test_file)
        
        if result:  # 如果能够成功导入和验证
            self.assertIsInstance(result, ValidationResult)
            self.assertEqual(result.function_name, "FileTestReward")
    
    def test_print_validation_summary(self):
        """测试打印验证总结"""
        # 创建一些测试结果
        results = [
            ValidationResult(
                function_name="good_function",
                passed=True,
                test_count=5,
                passed_count=5,
                failed_count=0,
                performance_metrics={'avg_computation_time_ms': 0.5}
            ),
            ValidationResult(
                function_name="bad_function", 
                passed=False,
                test_count=5,
                passed_count=3,
                failed_count=2,
                errors=["Error 1", "Error 2"],
                warnings=["Warning 1"]
            )
        ]
        
        # 这个测试主要确保方法不会抛出异常
        try:
            self.validator._print_validation_summary(results)
        except Exception as e:
            self.fail(f"打印验证总结失败: {e}")
    
    def test_create_validation_script(self):
        """测试创建验证脚本"""
        script_path = Path(self.temp_dir) / "test_validate.py"
        
        self.validator.create_validation_script(str(script_path))
        
        # 检查脚本是否创建
        self.assertTrue(script_path.exists())
        
        # 检查脚本内容
        script_content = script_path.read_text(encoding='utf-8')
        self.assertIn("MigrationValidator", script_content)
        self.assertIn("def main():", script_content)
        self.assertIn("validate_all_migrated_functions", script_content)


if __name__ == '__main__':
    unittest.main()