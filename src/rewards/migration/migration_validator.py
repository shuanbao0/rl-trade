"""
迁移验证器 - 验证迁移的奖励函数是否正确工作
"""

import unittest
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..core.reward_context import RewardContext
from .compatibility_mapper import CompatibilityMapper


@dataclass
class ValidationResult:
    """验证结果"""
    function_name: str
    passed: bool
    test_count: int
    passed_count: int
    failed_count: int
    errors: List[str] = None
    warnings: List[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


class MigrationValidator:
    """迁移验证器 - 验证迁移后的奖励函数"""
    
    def __init__(self):
        self.compatibility_mapper = CompatibilityMapper()
        
        # 标准测试用例
        self.standard_test_cases = self._create_standard_test_cases()
        
        # 性能基准
        self.performance_benchmarks = {
            'max_computation_time_ms': 10.0,  # 最大计算时间
            'max_memory_usage_mb': 100.0,     # 最大内存使用
            'min_reward_range': -1000.0,      # 最小奖励范围
            'max_reward_range': 1000.0,       # 最大奖励范围
        }
    
    def validate_all_migrated_functions(self, functions_dir: str) -> List[ValidationResult]:
        """验证所有迁移的函数"""
        print("🔍 开始验证迁移的奖励函数...")
        
        functions_path = Path(functions_dir)
        results = []
        
        # 查找所有Python文件
        for py_file in functions_path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                # 尝试导入和验证
                result = self._validate_single_file(py_file)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"⚠️ 验证文件失败 {py_file}: {e}")
        
        # 打印总结
        self._print_validation_summary(results)
        
        return results
    
    def validate_single_function(self, reward_class, function_name: str = None) -> ValidationResult:
        """验证单个奖励函数"""
        if function_name is None:
            function_name = reward_class.__name__
            
        print(f"🧪 验证函数: {function_name}")
        
        errors = []
        warnings = []
        test_results = []
        
        try:
            # 创建实例
            reward_instance = reward_class()
            
            # 运行标准测试
            test_results = self._run_standard_tests(reward_instance)
            
            # 性能测试
            perf_metrics = self._run_performance_tests(reward_instance)
            
            # 兼容性测试
            compat_results = self._run_compatibility_tests(reward_instance)
            test_results.extend(compat_results)
            
            # 统计结果
            passed_count = sum(1 for result in test_results if result['passed'])
            failed_count = len(test_results) - passed_count
            
            # 收集错误和警告
            for result in test_results:
                if not result['passed']:
                    errors.append(f"{result['test_name']}: {result['error']}")
                if 'warning' in result:
                    warnings.append(f"{result['test_name']}: {result['warning']}")
            
            return ValidationResult(
                function_name=function_name,
                passed=failed_count == 0,
                test_count=len(test_results),
                passed_count=passed_count,
                failed_count=failed_count,
                errors=errors,
                warnings=warnings,
                performance_metrics=perf_metrics
            )
            
        except Exception as e:
            return ValidationResult(
                function_name=function_name,
                passed=False,
                test_count=0,
                passed_count=0,
                failed_count=1,
                errors=[f"初始化失败: {str(e)}"]
            )
    
    def _validate_single_file(self, py_file: Path) -> Optional[ValidationResult]:
        """验证单个Python文件"""
        # 动态导入文件中的类
        try:
            import importlib.util
            import sys
            
            spec = importlib.util.spec_from_file_location("temp_module", py_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["temp_module"] = module
            spec.loader.exec_module(module)
            
            # 查找奖励类
            reward_classes = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, 'calculate') and 
                    attr.__name__ != 'BaseReward'):
                    reward_classes.append(attr)
            
            # 验证找到的类
            if reward_classes:
                # 通常一个文件只有一个主要的奖励类
                main_class = reward_classes[0]
                return self.validate_single_function(main_class)
            
        except Exception as e:
            print(f"⚠️ 导入文件失败 {py_file}: {e}")
            
        return None
    
    def _create_standard_test_cases(self) -> List[RewardContext]:
        """创建标准测试用例"""
        test_cases = []
        
        # 基本测试用例
        test_cases.append(RewardContext(
            portfolio_value=10000.0,
            action=0.0,
            current_price=1.0,
            step=0
        ))
        
        # 盈利情况
        test_cases.append(RewardContext(
            portfolio_value=11000.0,
            action=0.5,
            current_price=1.1,
            step=100,
            portfolio_history=np.array([10000, 10500, 11000])
        ))
        
        # 亏损情况
        test_cases.append(RewardContext(
            portfolio_value=9000.0,
            action=-0.5,
            current_price=0.9,
            step=100,
            portfolio_history=np.array([10000, 9500, 9000])
        ))
        
        # 边界情况
        test_cases.append(RewardContext(
            portfolio_value=0.0,
            action=1.0,
            current_price=0.001,
            step=1000000
        ))
        
        # 外汇特定测试
        test_cases.append(RewardContext(
            portfolio_value=10500.0,
            action=0.3,
            current_price=1.2345,
            step=50,
            market_type='forex',
            granularity='1min',
            metadata={'pip_size': 0.0001, 'spread': 2, 'leverage': 100}
        ))
        
        return test_cases
    
    def _run_standard_tests(self, reward_instance) -> List[Dict[str, Any]]:
        """运行标准测试"""
        test_results = []
        
        for i, test_case in enumerate(self.standard_test_cases):
            try:
                # 测试基本计算
                result = reward_instance.calculate(test_case)
                
                test_result = {
                    'test_name': f'standard_test_{i+1}',
                    'passed': True,
                    'result': result
                }
                
                # 验证结果类型
                if not isinstance(result, (int, float)):
                    test_result['passed'] = False
                    test_result['error'] = f"返回类型错误: {type(result)}, 期望 float"
                
                # 验证结果范围
                elif not np.isfinite(result):
                    test_result['passed'] = False
                    test_result['error'] = f"返回值无效: {result}"
                
                # 验证结果范围
                elif not (self.performance_benchmarks['min_reward_range'] <= 
                         result <= self.performance_benchmarks['max_reward_range']):
                    test_result['warning'] = f"奖励值超出建议范围: {result}"
                
                test_results.append(test_result)
                
            except Exception as e:
                test_results.append({
                    'test_name': f'standard_test_{i+1}',
                    'passed': False,
                    'error': str(e)
                })
        
        return test_results
    
    def _run_performance_tests(self, reward_instance) -> Dict[str, float]:
        """运行性能测试"""
        import time
        import psutil
        import os
        
        metrics = {}
        
        try:
            # 准备测试数据
            test_context = self.standard_test_cases[1]  # 使用有历史数据的测试用例
            
            # 测试计算时间
            start_time = time.perf_counter()
            for _ in range(1000):  # 运行1000次
                reward_instance.calculate(test_context)
            end_time = time.perf_counter()
            
            avg_time_ms = (end_time - start_time) * 1000 / 1000
            metrics['avg_computation_time_ms'] = avg_time_ms
            metrics['computation_time_ok'] = avg_time_ms <= self.performance_benchmarks['max_computation_time_ms']
            
            # 测试内存使用
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建大量实例测试内存泄漏
            instances = [reward_instance.__class__() for _ in range(100)]
            for instance in instances:
                instance.calculate(test_context)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            metrics['memory_usage_mb'] = memory_usage
            metrics['memory_usage_ok'] = memory_usage <= self.performance_benchmarks['max_memory_usage_mb']
            
            del instances  # 清理
            
        except Exception as e:
            metrics['performance_test_error'] = str(e)
        
        return metrics
    
    def _run_compatibility_tests(self, reward_instance) -> List[Dict[str, Any]]:
        """运行兼容性测试"""
        test_results = []
        
        # 测试必需方法存在
        required_methods = ['calculate', 'get_info']
        for method_name in required_methods:
            test_result = {
                'test_name': f'has_method_{method_name}',
                'passed': hasattr(reward_instance, method_name),
            }
            if not test_result['passed']:
                test_result['error'] = f"缺少必需方法: {method_name}"
            test_results.append(test_result)
        
        # 测试get_info方法
        try:
            info = reward_instance.get_info()
            test_result = {
                'test_name': 'get_info_test',
                'passed': isinstance(info, dict),
            }
            if not test_result['passed']:
                test_result['error'] = f"get_info返回类型错误: {type(info)}"
            elif 'name' not in info:
                test_result['warning'] = "get_info结果缺少'name'字段"
            test_results.append(test_result)
        except Exception as e:
            test_results.append({
                'test_name': 'get_info_test',
                'passed': False,
                'error': f"get_info方法调用失败: {str(e)}"
            })
        
        # 测试向后兼容性（如果支持）
        if hasattr(reward_instance, 'compute_reward'):
            try:
                old_context = {
                    'portfolio_value': 10000.0,
                    'action': 0.5,
                    'current_price': 1.0,
                    'step': 10
                }
                result = reward_instance.compute_reward(old_context)
                test_results.append({
                    'test_name': 'backward_compatibility',
                    'passed': isinstance(result, (int, float)),
                })
            except Exception as e:
                test_results.append({
                    'test_name': 'backward_compatibility',
                    'passed': False,
                    'error': f"向后兼容性测试失败: {str(e)}"
                })
        
        return test_results
    
    def _print_validation_summary(self, results: List[ValidationResult]):
        """打印验证总结"""
        total_functions = len(results)
        passed_functions = sum(1 for r in results if r.passed)
        failed_functions = total_functions - passed_functions
        
        print("\n" + "="*60)
        print("📊 奖励函数验证报告")
        print("="*60)
        
        print(f"\n📈 总体情况:")
        print(f"  • 验证函数数量: {total_functions}")
        print(f"  • 通过验证: {passed_functions}")
        print(f"  • 验证失败: {failed_functions}")
        print(f"  • 成功率: {passed_functions/total_functions*100:.1f}%" if total_functions > 0 else "N/A")
        
        if failed_functions > 0:
            print(f"\n❌ 验证失败的函数:")
            for result in results:
                if not result.passed:
                    print(f"  • {result.function_name}: {result.failed_count} 个测试失败")
                    for error in result.errors[:3]:  # 只显示前3个错误
                        print(f"    - {error}")
        
        # 性能统计
        avg_compute_times = [r.performance_metrics.get('avg_computation_time_ms', 0) 
                           for r in results if r.performance_metrics]
        if avg_compute_times:
            print(f"\n⚡ 性能统计:")
            print(f"  • 平均计算时间: {np.mean(avg_compute_times):.2f} ms")
            print(f"  • 最大计算时间: {np.max(avg_compute_times):.2f} ms")
            print(f"  • 最小计算时间: {np.min(avg_compute_times):.2f} ms")
        
        # 警告汇总
        all_warnings = []
        for result in results:
            all_warnings.extend(result.warnings)
        
        if all_warnings:
            print(f"\n⚠️  常见警告 (显示前5个):")
            warning_counts = {}
            for warning in all_warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
            
            for warning, count in sorted(warning_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {warning} ({count}次)")
        
        print("\n" + "="*60)
    
    def create_validation_script(self, output_file: str = "validate_migrated_rewards.py"):
        """创建验证脚本"""
        script_content = f'''#!/usr/bin/env python3
"""
迁移奖励函数验证脚本
自动生成于: {self._get_timestamp()}
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.migration.migration_validator import MigrationValidator


def main():
    print("🔍 开始验证迁移的奖励函数...")
    
    # 创建验证器
    validator = MigrationValidator()
    
    # 验证所有迁移的函数
    results = validator.validate_all_migrated_functions("src/rewards/functions")
    
    # 分析结果
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    print(f"\\n📊 验证完成!")
    print(f"  ✅ 通过: {{passed}}/{{total}}")
    print(f"  ❌ 失败: {{failed}}/{{total}}")
    
    if failed > 0:
        print(f"\\n❌ 需要修复的函数:")
        for result in results:
            if not result.passed:
                print(f"  • {{result.function_name}}")
                for error in result.errors[:2]:
                    print(f"    - {{error}}")
        
        return False
    else:
        print("\\n🎉 所有函数验证通过!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        import os
        os.chmod(output_file, 0o755)
        print(f"✅ 验证脚本已创建: {output_file}")
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")