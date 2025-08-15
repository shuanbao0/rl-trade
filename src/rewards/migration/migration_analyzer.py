"""
迁移分析器 - 分析现有奖励函数结构和依赖关系
"""

import ast
import os
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..core.reward_context import RewardContext


@dataclass
class RewardFunctionInfo:
    """奖励函数信息"""
    name: str
    file_path: str
    class_name: str
    base_class: str
    dependencies: Set[str]
    market_compatibility: Set[str]  # 市场兼容性
    granularity_compatibility: Set[str]  # 时间粒度兼容性
    complexity_score: int  # 复杂度评分 1-10
    migration_priority: int  # 迁移优先级 1-10
    aliases: List[str]  # 别名列表
    parameters: Dict[str, str]  # 参数类型映射
    imports: Set[str]  # 导入依赖


@dataclass 
class MigrationPlan:
    """迁移计划"""
    total_functions: int
    high_priority: List[RewardFunctionInfo]
    medium_priority: List[RewardFunctionInfo]
    low_priority: List[RewardFunctionInfo]
    migration_order: List[str]  # 推荐迁移顺序
    estimated_time_hours: float
    potential_issues: List[str]


class MigrationAnalyzer:
    """迁移分析器 - 分析现有奖励函数并生成迁移计划"""
    
    def __init__(self, old_rewards_path: str = "src/environment/rewards"):
        self.old_rewards_path = Path(old_rewards_path)
        self.reward_functions: Dict[str, RewardFunctionInfo] = {}
        self.alias_mapping: Dict[str, str] = {}  # 别名到主名称的映射
        
        # 市场类型关键词映射
        self.market_keywords = {
            'forex': ['forex', 'fx', 'currency', 'pip', 'spread'],
            'stock': ['stock', 'equity', 'share', 'dividend'],
            'crypto': ['crypto', 'bitcoin', 'btc', 'eth', 'digital']
        }
        
        # 时间粒度关键词映射
        self.granularity_keywords = {
            '1min': ['minute', '1m', 'tick', 'high_freq'],
            '5min': ['5min', '5m', 'short_term'],
            '1h': ['hour', '1h', 'hourly'],
            '1d': ['day', 'daily', 'end_of_day'],
            '1w': ['week', 'weekly', 'swing']
        }
    
    def analyze_all_functions(self) -> MigrationPlan:
        """分析所有奖励函数并生成迁移计划"""
        print("🔍 开始分析现有奖励函数...")
        
        # 1. 扫描所有Python文件
        self._scan_reward_files()
        
        # 2. 分析别名映射
        self._analyze_aliases()
        
        # 3. 分析依赖关系
        self._analyze_dependencies()
        
        # 4. 评估复杂度和优先级
        self._evaluate_complexity_and_priority()
        
        # 5. 生成迁移计划
        plan = self._generate_migration_plan()
        
        print(f"✅ 分析完成！发现 {len(self.reward_functions)} 个奖励函数")
        return plan
    
    def _scan_reward_files(self):
        """扫描奖励函数文件"""
        if not self.old_rewards_path.exists():
            print(f"⚠️ 奖励函数路径不存在: {self.old_rewards_path}")
            return
            
        for py_file in self.old_rewards_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                self._analyze_file(py_file)
            except Exception as e:
                print(f"⚠️ 分析文件失败 {py_file}: {e}")
    
    def _analyze_file(self, file_path: Path):
        """分析单个Python文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"⚠️ 语法错误 {file_path}: {e}")
            return
            
        # 分析AST获取类定义
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._analyze_class(node, file_path, content)
    
    def _analyze_class(self, class_node: ast.ClassDef, file_path: Path, content: str):
        """分析类定义"""
        class_name = class_node.name
        
        # 跳过非奖励函数类
        if not self._is_reward_class(class_node):
            return
            
        # 获取基类
        base_class = self._get_base_class(class_node)
        
        # 分析依赖
        dependencies = self._extract_dependencies(content)
        
        # 分析市场和时间粒度兼容性
        market_compat = self._analyze_market_compatibility(content, class_name)
        granularity_compat = self._analyze_granularity_compatibility(content, class_name)
        
        # 提取参数
        parameters = self._extract_parameters(class_node)
        
        # 提取导入
        imports = self._extract_imports(content)
        
        # 创建奖励函数信息
        reward_info = RewardFunctionInfo(
            name=class_name.lower().replace('reward', ''),
            file_path=str(file_path),
            class_name=class_name,
            base_class=base_class,
            dependencies=dependencies,
            market_compatibility=market_compat,
            granularity_compatibility=granularity_compat,
            complexity_score=0,  # 稍后计算
            migration_priority=0,  # 稍后计算
            aliases=[],  # 稍后从factory分析
            parameters=parameters,
            imports=imports
        )
        
        self.reward_functions[reward_info.name] = reward_info
        print(f"📋 发现奖励函数: {class_name} -> {reward_info.name}")
    
    def _is_reward_class(self, class_node: ast.ClassDef) -> bool:
        """判断是否为奖励函数类"""
        # 检查类名
        if 'reward' not in class_node.name.lower():
            return False
            
        # 检查是否有calculate方法
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'calculate':
                return True
                
        return False
    
    def _get_base_class(self, class_node: ast.ClassDef) -> str:
        """获取基类名称"""
        if not class_node.bases:
            return "object"
            
        base = class_node.bases[0]
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
            
        return "unknown"
    
    def _extract_dependencies(self, content: str) -> Set[str]:
        """提取依赖关系"""
        dependencies = set()
        
        # 正则匹配导入
        import_patterns = [
            r'from\s+(\S+)\s+import',
            r'import\s+(\S+)',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.update(matches)
            
        return dependencies
    
    def _analyze_market_compatibility(self, content: str, class_name: str) -> Set[str]:
        """分析市场类型兼容性"""
        compatible_markets = set()
        content_lower = content.lower() + class_name.lower()
        
        for market, keywords in self.market_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                compatible_markets.add(market)
                
        # 如果没有特定市场关键词，认为是通用的
        if not compatible_markets:
            compatible_markets = {'forex', 'stock', 'crypto'}
            
        return compatible_markets
    
    def _analyze_granularity_compatibility(self, content: str, class_name: str) -> Set[str]:
        """分析时间粒度兼容性"""
        compatible_granularities = set()
        content_lower = content.lower() + class_name.lower()
        
        for granularity, keywords in self.granularity_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                compatible_granularities.add(granularity)
                
        # 如果没有特定粒度关键词，认为是通用的
        if not compatible_granularities:
            compatible_granularities = {'1min', '5min', '1h', '1d', '1w'}
            
        return compatible_granularities
    
    def _extract_parameters(self, class_node: ast.ClassDef) -> Dict[str, str]:
        """提取参数信息"""
        parameters = {}
        
        # 查找__init__方法
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for arg in node.args.args[1:]:  # 跳过self
                    param_type = "Any"
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            param_type = arg.annotation.id
                        elif isinstance(arg.annotation, ast.Constant):
                            param_type = str(arg.annotation.value)
                    parameters[arg.arg] = param_type
                break
                
        return parameters
    
    def _extract_imports(self, content: str) -> Set[str]:
        """提取导入信息"""
        imports = set()
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.add(line)
                
        return imports
    
    def _analyze_aliases(self):
        """分析别名映射（从reward_factory.py）"""
        factory_path = self.old_rewards_path / "reward_factory.py"
        if not factory_path.exists():
            print("⚠️ 未找到reward_factory.py，跳过别名分析")
            return
            
        try:
            with open(factory_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的正则匹配别名定义
            # 这里可能需要根据实际的factory实现调整
            alias_pattern = r'["\']([^"\']+)["\']\s*:\s*(\w+)'
            matches = re.findall(alias_pattern, content)
            
            for alias, class_name in matches:
                # 找到对应的奖励函数
                for reward_name, reward_info in self.reward_functions.items():
                    if reward_info.class_name == class_name:
                        reward_info.aliases.append(alias)
                        self.alias_mapping[alias] = reward_name
                        break
                        
        except Exception as e:
            print(f"⚠️ 分析别名失败: {e}")
    
    def _analyze_dependencies(self):
        """分析依赖关系"""
        # 这里可以添加更复杂的依赖分析
        # 例如分析哪些奖励函数依赖于其他奖励函数
        pass
    
    def _evaluate_complexity_and_priority(self):
        """评估复杂度和迁移优先级"""
        for reward_info in self.reward_functions.values():
            # 计算复杂度分数
            complexity = self._calculate_complexity(reward_info)
            reward_info.complexity_score = complexity
            
            # 计算优先级
            priority = self._calculate_priority(reward_info)
            reward_info.migration_priority = priority
    
    def _calculate_complexity(self, reward_info: RewardFunctionInfo) -> int:
        """计算复杂度分数 (1-10)"""
        score = 1
        
        # 依赖数量影响复杂度
        score += min(len(reward_info.dependencies), 3)
        
        # 参数数量影响复杂度
        score += min(len(reward_info.parameters), 3)
        
        # 特定市场或粒度的专用性增加复杂度
        if len(reward_info.market_compatibility) < 3:
            score += 1
        if len(reward_info.granularity_compatibility) < 5:
            score += 1
            
        # 别名数量（受欢迎程度）
        score += min(len(reward_info.aliases) // 5, 2)
        
        return min(score, 10)
    
    def _calculate_priority(self, reward_info: RewardFunctionInfo) -> int:
        """计算迁移优先级 (1-10, 10为最高优先级)"""
        priority = 5  # 基础优先级
        
        # 基础奖励函数优先级高
        basic_rewards = ['simple', 'return', 'profit', 'loss']
        if any(basic in reward_info.name for basic in basic_rewards):
            priority += 3
            
        # 别名多的（使用频繁）优先级高
        priority += min(len(reward_info.aliases) // 3, 2)
        
        # 复杂度低的优先迁移
        priority += (10 - reward_info.complexity_score) // 2
        
        # 通用性强的优先迁移
        if len(reward_info.market_compatibility) == 3:
            priority += 1
        if len(reward_info.granularity_compatibility) == 5:
            priority += 1
            
        return min(priority, 10)
    
    def _generate_migration_plan(self) -> MigrationPlan:
        """生成迁移计划"""
        functions = list(self.reward_functions.values())
        
        # 按优先级分组
        high_priority = [f for f in functions if f.migration_priority >= 8]
        medium_priority = [f for f in functions if 5 <= f.migration_priority < 8]
        low_priority = [f for f in functions if f.migration_priority < 5]
        
        # 生成推荐迁移顺序
        migration_order = []
        
        # 高优先级按复杂度排序（简单的先迁移）
        high_priority.sort(key=lambda x: x.complexity_score)
        migration_order.extend([f.name for f in high_priority])
        
        # 中等优先级
        medium_priority.sort(key=lambda x: x.complexity_score)
        migration_order.extend([f.name for f in medium_priority])
        
        # 低优先级
        low_priority.sort(key=lambda x: x.complexity_score)
        migration_order.extend([f.name for f in low_priority])
        
        # 估算时间
        estimated_time = self._estimate_migration_time(functions)
        
        # 识别潜在问题
        potential_issues = self._identify_potential_issues(functions)
        
        return MigrationPlan(
            total_functions=len(functions),
            high_priority=high_priority,
            medium_priority=medium_priority,
            low_priority=low_priority,
            migration_order=migration_order,
            estimated_time_hours=estimated_time,
            potential_issues=potential_issues
        )
    
    def _estimate_migration_time(self, functions: List[RewardFunctionInfo]) -> float:
        """估算迁移时间（小时）"""
        total_time = 0.0
        
        for func in functions:
            # 基础时间
            base_time = 0.5
            
            # 复杂度影响
            complexity_time = func.complexity_score * 0.2
            
            # 依赖数量影响
            dependency_time = len(func.dependencies) * 0.1
            
            # 参数数量影响
            param_time = len(func.parameters) * 0.1
            
            total_time += base_time + complexity_time + dependency_time + param_time
            
        return round(total_time, 1)
    
    def _identify_potential_issues(self, functions: List[RewardFunctionInfo]) -> List[str]:
        """识别潜在迁移问题"""
        issues = []
        
        # 检查循环依赖
        # TODO: 实现更复杂的依赖分析
        
        # 检查高复杂度函数
        high_complexity = [f for f in functions if f.complexity_score >= 8]
        if high_complexity:
            issues.append(f"发现 {len(high_complexity)} 个高复杂度函数，需要特别注意")
            
        # 检查未知基类
        unknown_base = [f for f in functions if f.base_class == "unknown"]
        if unknown_base:
            issues.append(f"发现 {len(unknown_base)} 个未知基类函数，需要手动检查")
            
        # 检查大量别名
        many_aliases = [f for f in functions if len(f.aliases) > 10]
        if many_aliases:
            issues.append(f"发现 {len(many_aliases)} 个函数有大量别名，需要仔细处理兼容性")
            
        return issues
    
    def print_analysis_report(self, plan: MigrationPlan):
        """打印分析报告"""
        print("\n" + "="*60)
        print("📊 奖励函数迁移分析报告")
        print("="*60)
        
        print(f"\n📈 总体情况:")
        print(f"  • 总函数数量: {plan.total_functions}")
        print(f"  • 高优先级: {len(plan.high_priority)}")
        print(f"  • 中优先级: {len(plan.medium_priority)}")
        print(f"  • 低优先级: {len(plan.low_priority)}")
        print(f"  • 预估迁移时间: {plan.estimated_time_hours} 小时")
        
        print(f"\n🎯 迁移优先级排序:")
        for i, name in enumerate(plan.migration_order[:10], 1):
            func = self.reward_functions[name]
            print(f"  {i:2d}. {func.class_name} (复杂度: {func.complexity_score}, 优先级: {func.migration_priority})")
        
        if len(plan.migration_order) > 10:
            print(f"     ... 还有 {len(plan.migration_order) - 10} 个函数")
        
        if plan.potential_issues:
            print(f"\n⚠️  潜在问题:")
            for issue in plan.potential_issues:
                print(f"  • {issue}")
        
        print(f"\n💡 推荐迁移策略:")
        print(f"  1. 先迁移高优先级的简单函数作为基础框架验证")
        print(f"  2. 逐步迁移中等优先级函数，积累经验")
        print(f"  3. 最后处理低优先级和高复杂度函数")
        print(f"  4. 每个函数迁移后立即进行测试验证")
        
        print("\n" + "="*60)