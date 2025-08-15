"""
迁移执行器 - 自动执行奖励函数迁移
"""

import os
import re
import ast
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .migration_analyzer import MigrationAnalyzer, RewardFunctionInfo
from ..core.reward_context import RewardContext


@dataclass
class MigrationResult:
    """迁移结果"""
    function_name: str
    source_file: str
    target_file: str
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class MigrationExecutor:
    """迁移执行器 - 自动迁移奖励函数到新架构"""
    
    def __init__(self, 
                 old_rewards_path: str = "src/environment/rewards",
                 new_rewards_path: str = "src/rewards"):
        self.old_rewards_path = Path(old_rewards_path)
        self.new_rewards_path = Path(new_rewards_path)
        self.analyzer = MigrationAnalyzer(old_rewards_path)
        
        # 迁移模板
        self.migration_templates = {
            'simple': self._get_simple_template(),
            'complex': self._get_complex_template(),
            'forex': self._get_forex_template()
        }
        
        # 代码替换规则
        self.replacement_rules = {
            # 旧的导入 -> 新的导入
            'from src.environment.rewards.base_reward import BaseReward': 
            'from src.rewards.core.base_reward import BaseReward',
            
            'from ..base_reward import BaseReward':
            'from src.rewards.core.base_reward import BaseReward',
            
            # 旧的上下文访问 -> 新的上下文访问
            'context.portfolio_value': 'context.portfolio_value',
            'context.action': 'context.action',
            'context.current_price': 'context.current_price',
            'context.step': 'context.step',
            
            # 方法名替换
            'get_reward_info': 'get_info',
        }
    
    def execute_migration(self, function_names: Optional[List[str]] = None) -> List[MigrationResult]:
        """执行迁移"""
        print("🚀 开始执行奖励函数迁移...")
        
        # 分析现有函数
        plan = self.analyzer.analyze_all_functions()
        
        # 确定要迁移的函数
        if function_names is None:
            function_names = plan.migration_order
        
        results = []
        
        # 创建目标目录结构
        self._create_directory_structure()
        
        for func_name in function_names:
            if func_name not in self.analyzer.reward_functions:
                print(f"⚠️ 函数 {func_name} 不存在，跳过")
                continue
                
            print(f"📦 迁移函数: {func_name}")
            result = self._migrate_single_function(func_name)
            results.append(result)
            
            if result.success:
                print(f"✅ 迁移成功: {func_name}")
            else:
                print(f"❌ 迁移失败: {func_name} - {result.error_message}")
                
        print(f"\n📊 迁移完成! 成功: {sum(1 for r in results if r.success)}/{len(results)}")
        return results
    
    def _create_directory_structure(self):
        """创建新的目录结构"""
        directories = [
            self.new_rewards_path / "functions",
            self.new_rewards_path / "functions" / "basic",
            self.new_rewards_path / "functions" / "advanced", 
            self.new_rewards_path / "functions" / "forex",
            self.new_rewards_path / "functions" / "stock",
            self.new_rewards_path / "functions" / "crypto",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
            # 创建__init__.py文件
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Auto-generated __init__.py"""\\n')
    
    def _migrate_single_function(self, function_name: str) -> MigrationResult:
        """迁移单个函数"""
        try:
            reward_info = self.analyzer.reward_functions[function_name]
            
            # 读取源文件
            with open(reward_info.file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 确定目标目录和文件名
            target_dir, target_filename = self._determine_target_location(reward_info)
            target_file = target_dir / target_filename
            
            # 转换代码
            converted_code, warnings = self._convert_code(source_code, reward_info)
            
            # 写入目标文件
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(converted_code)
            
            return MigrationResult(
                function_name=function_name,
                source_file=reward_info.file_path,
                target_file=str(target_file),
                success=True,
                warnings=warnings
            )
            
        except Exception as e:
            return MigrationResult(
                function_name=function_name,
                source_file=reward_info.file_path if 'reward_info' in locals() else "unknown",
                target_file="unknown",
                success=False,
                error_message=str(e)
            )
    
    def _determine_target_location(self, reward_info: RewardFunctionInfo) -> Tuple[Path, str]:
        """确定目标位置"""
        # 根据复杂度和市场类型确定目录
        if reward_info.complexity_score <= 3:
            target_dir = self.new_rewards_path / "functions" / "basic"
        elif len(reward_info.market_compatibility) == 1:
            # 特定市场的奖励函数
            market = list(reward_info.market_compatibility)[0]
            target_dir = self.new_rewards_path / "functions" / market
        else:
            target_dir = self.new_rewards_path / "functions" / "advanced"
            
        # 文件名使用函数名
        filename = f"{reward_info.name}_reward.py"
        
        return target_dir, filename
    
    def _convert_code(self, source_code: str, reward_info: RewardFunctionInfo) -> Tuple[str, List[str]]:
        """转换代码到新架构"""
        warnings = []
        converted_code = source_code
        
        # 应用替换规则
        for old_pattern, new_pattern in self.replacement_rules.items():
            if old_pattern in converted_code:
                converted_code = converted_code.replace(old_pattern, new_pattern)
                warnings.append(f"替换: {old_pattern} -> {new_pattern}")
        
        # 添加新的导入
        converted_code = self._add_new_imports(converted_code)
        
        # 添加元数据
        converted_code = self._add_metadata(converted_code, reward_info)
        
        # 验证语法
        try:
            ast.parse(converted_code)
        except SyntaxError as e:
            warnings.append(f"语法错误: {e}")
            # 尝试基本修复
            converted_code = self._fix_basic_syntax_issues(converted_code)
            
        return converted_code, warnings
    
    def _add_new_imports(self, code: str) -> str:
        """添加新的导入"""
        new_imports = [
            "from src.rewards.core.reward_context import RewardContext",
            "from src.rewards.core.base_reward import BaseReward",
            "from typing import Optional, Dict, Any"
        ]
        
        # 找到现有导入的位置
        lines = code.split('\\n')
        import_end_idx = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')) or line.strip().startswith('"""') or line.strip() == '':
                import_end_idx = i + 1
            else:
                break
        
        # 在导入区域末尾添加新导入
        for new_import in new_imports:
            if new_import not in code:
                lines.insert(import_end_idx, new_import)
                import_end_idx += 1
        
        return '\\n'.join(lines)
    
    def _add_metadata(self, code: str, reward_info: RewardFunctionInfo) -> str:
        """添加元数据"""
        metadata_comment = f'''
"""
自动迁移的奖励函数: {reward_info.class_name}
原始文件: {reward_info.file_path}
市场兼容性: {', '.join(reward_info.market_compatibility)}
时间粒度兼容性: {', '.join(reward_info.granularity_compatibility)}
复杂度: {reward_info.complexity_score}/10
别名: {', '.join(reward_info.aliases)}
"""
'''
        
        # 在文件开头添加元数据（在docstring之后）
        lines = code.split('\\n')
        
        # 找到类定义行
        class_line_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(f'class {reward_info.class_name}'):
                class_line_idx = i
                break
        
        if class_line_idx >= 0:
            lines.insert(class_line_idx, metadata_comment)
        
        return '\\n'.join(lines)
    
    def _fix_basic_syntax_issues(self, code: str) -> str:
        """修复基本语法问题"""
        # 这里可以添加一些常见的语法修复
        # 例如：缺少冒号、缩进问题等
        
        lines = code.split('\\n')
        fixed_lines = []
        
        for line in lines:
            # 修复常见的方法定义问题
            if 'def ' in line and not line.strip().endswith(':'):
                if '(' in line and ')' in line:
                    line = line.rstrip() + ':'
            
            fixed_lines.append(line)
        
        return '\\n'.join(fixed_lines)
    
    def _get_simple_template(self) -> str:
        """获取简单奖励函数模板"""
        return '''"""
{description}
"""

from src.rewards.core.base_reward import BaseReward
from src.rewards.core.reward_context import RewardContext
from typing import Optional, Dict, Any


class {class_name}(BaseReward):
    """
    {description}
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        # 初始化参数
        {init_params}
    
    def calculate(self, context: RewardContext) -> float:
        """计算奖励值"""
        # TODO: 实现奖励计算逻辑
        {calculation_logic}
        
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {{
            'name': self.name,
            'type': '{reward_type}',
            'description': '{description}',
            'market_compatibility': {market_compatibility},
            'granularity_compatibility': {granularity_compatibility},
            'parameters': {parameters}
        }}
'''
    
    def _get_complex_template(self) -> str:
        """获取复杂奖励函数模板"""
        return '''"""
{description}
"""

from src.rewards.core.base_reward import BaseReward, HistoryAwareRewardMixin
from src.rewards.core.reward_context import RewardContext
from typing import Optional, Dict, Any
import numpy as np


class {class_name}(BaseReward, HistoryAwareRewardMixin):
    """
    {description}
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        # 初始化参数
        {init_params}
        
        # 历史数据配置
        self.min_history_steps = config.get('min_history_steps', 10)
    
    def calculate(self, context: RewardContext) -> float:
        """计算奖励值"""
        # 检查历史数据充足性
        if not self.has_sufficient_history(context):
            return 0.0
            
        # TODO: 实现复杂奖励计算逻辑
        {calculation_logic}
        
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {{
            'name': self.name,
            'type': '{reward_type}',
            'description': '{description}',
            'market_compatibility': {market_compatibility},
            'granularity_compatibility': {granularity_compatibility},
            'parameters': {parameters},
            'requires_history': True,
            'min_history_steps': self.min_history_steps
        }}
'''
    
    def _get_forex_template(self) -> str:
        """获取外汇专用奖励函数模板"""
        return '''"""
{description} - 外汇优化版本
"""

from src.rewards.core.base_reward import BaseReward
from src.rewards.core.reward_context import RewardContext
from typing import Optional, Dict, Any


class {class_name}(BaseReward):
    """
    {description} - 专为外汇市场优化
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        # 外汇特定参数
        self.pip_value = config.get('pip_value', 0.0001)
        self.spread_penalty = config.get('spread_penalty', 0.01)
        self.leverage = config.get('leverage', 100)
        
        # 其他参数
        {init_params}
    
    def calculate(self, context: RewardContext) -> float:
        """计算外汇奖励值"""
        # 获取外汇特定数据
        pip_size = context.metadata.get('pip_size', self.pip_value)
        spread = context.metadata.get('spread', 0)
        
        # TODO: 实现外汇奖励计算逻辑
        {calculation_logic}
        
        # 应用点差惩罚
        spread_cost = spread * self.spread_penalty
        reward -= spread_cost
        
        return reward
        
    def get_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {{
            'name': self.name,
            'type': 'forex_{reward_type}',
            'description': '{description}',
            'market_compatibility': ['forex'],
            'granularity_compatibility': {granularity_compatibility},
            'parameters': {parameters},
            'forex_optimized': True,
            'pip_value': self.pip_value,
            'spread_penalty': self.spread_penalty,
            'leverage': self.leverage
        }}
'''
    
    def create_migration_script(self, output_file: str = "migrate_rewards.py"):
        """创建迁移脚本"""
        script_content = f'''#!/usr/bin/env python3
"""
奖励函数自动迁移脚本
自动生成于: {self._get_timestamp()}
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.migration.migration_executor import MigrationExecutor


def main():
    print("🚀 开始奖励函数迁移...")
    
    # 创建迁移执行器
    executor = MigrationExecutor()
    
    # 执行迁移
    results = executor.execute_migration()
    
    # 打印结果
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\\n📊 迁移结果:")
    print(f"  ✅ 成功: {{len(successful)}}")
    print(f"  ❌ 失败: {{len(failed)}}")
    
    if failed:
        print(f"\\n❌ 失败的函数:")
        for result in failed:
            print(f"  • {{result.function_name}}: {{result.error_message}}")
    
    if successful:
        print(f"\\n✅ 成功迁移的函数:")
        for result in successful:
            print(f"  • {{result.function_name}} -> {{result.target_file}}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"    ⚠️ {{warning}}")
    
    print("\\n🎉 迁移完成!")
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(output_file, 0o755)
        
        print(f"✅ 迁移脚本已创建: {output_file}")
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")