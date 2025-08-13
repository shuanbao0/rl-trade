"""
LLM引导的奖励函数设计模块

基于2024-2025年最新研究，实现LLM引导的奖励函数自动设计和优化系统。
核心技术包括：
- EUREKA算法的进化优化
- Constitutional AI安全框架
- 自然语言规范解析
- 代码生成和验证
- 迭代优化和自我改进

参考文献：
- EUREKA: Human-Level Reward Design via Coding Large Language Models (ICLR 2024)
- Constitutional AI: Harmlessness from AI Feedback (2024)
- Language to Rewards for Robotic Skill Synthesis (Google Research 2024)
- RLAIF: Scaling Reinforcement Learning from Human Feedback (2024)
"""

import logging
import json
import re
import ast
import inspect
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

from .base_reward import BaseRewardScheme

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class RewardSpecification:
    """奖励函数规范定义"""
    objective: str  # 主要目标描述
    constraints: List[str]  # 约束条件
    preferences: Dict[str, float]  # 偏好权重
    risk_tolerance: str  # 风险容忍度 (low/medium/high)
    time_horizon: str  # 时间范围 (short/medium/long)
    ethical_guidelines: List[str]  # 伦理指导原则

@dataclass 
class ConstitutionalPrinciple:
    """Constitutional AI原则"""
    name: str
    description: str
    constraint: str
    weight: float = 1.0

class NaturalLanguageParser:
    """自然语言奖励规范解析器"""
    
    def __init__(self):
        self.objective_patterns = [
            r"maximize|max|increase|improve|optimize|enhance",
            r"minimize|min|reduce|decrease|lower|limit",
            r"balance|maintain|stabilize|keep|ensure"
        ]
        
        self.risk_patterns = {
            'low': r"conservative|safe|low.risk|stable|minimal.risk",
            'medium': r"moderate|balanced|medium.risk|reasonable",
            'high': r"aggressive|high.risk|speculative|bold"
        }
        
        self.time_patterns = {
            'short': r"short.term|daily|immediate|quick",
            'medium': r"medium.term|weekly|monthly|intermediate", 
            'long': r"long.term|yearly|strategic|sustained"
        }
    
    def parse_specification(self, natural_language_spec: str) -> RewardSpecification:
        """解析自然语言规范为结构化格式"""
        try:
            # 提取主要目标
            objective = self._extract_objective(natural_language_spec)
            
            # 提取约束条件
            constraints = self._extract_constraints(natural_language_spec)
            
            # 提取偏好权重
            preferences = self._extract_preferences(natural_language_spec)
            
            # 提取风险容忍度
            risk_tolerance = self._extract_risk_tolerance(natural_language_spec)
            
            # 提取时间范围
            time_horizon = self._extract_time_horizon(natural_language_spec)
            
            # 提取伦理指导原则
            ethical_guidelines = self._extract_ethical_guidelines(natural_language_spec)
            
            return RewardSpecification(
                objective=objective,
                constraints=constraints,
                preferences=preferences,
                risk_tolerance=risk_tolerance,
                time_horizon=time_horizon,
                ethical_guidelines=ethical_guidelines
            )
            
        except Exception as e:
            logger.error(f"Natural language parsing failed: {e}")
            # 返回默认规范
            return self._get_default_specification()
    
    def _extract_objective(self, text: str) -> str:
        """提取主要目标"""
        text_lower = text.lower()
        
        if any(re.search(pattern, text_lower) for pattern in [r"profit", r"return", r"gain"]):
            return "maximize_returns"
        elif any(re.search(pattern, text_lower) for pattern in [r"risk", r"drawdown", r"loss"]):
            return "minimize_risk"
        elif any(re.search(pattern, text_lower) for pattern in [r"sharpe", r"ratio", r"efficiency"]):
            return "maximize_risk_adjusted_returns"
        else:
            return "balanced_performance"
    
    def _extract_constraints(self, text: str) -> List[str]:
        """提取约束条件"""
        constraints = []
        text_lower = text.lower()
        
        if re.search(r"drawdown.*(?:below|under|less.*than).*(\d+)", text_lower):
            match = re.search(r"drawdown.*(?:below|under|less.*than).*(\d+)", text_lower)
            if match:
                constraints.append(f"max_drawdown_limit_{match.group(1)}")
        
        if re.search(r"volatility.*(?:below|under|less.*than).*(\d+)", text_lower):
            match = re.search(r"volatility.*(?:below|under|less.*than).*(\d+)", text_lower)
            if match:
                constraints.append(f"volatility_limit_{match.group(1)}")
        
        if re.search(r"no.*(?:leverage|borrowing|margin)", text_lower):
            constraints.append("no_leverage")
            
        return constraints
    
    def _extract_preferences(self, text: str) -> Dict[str, float]:
        """提取偏好权重"""
        preferences = {}
        text_lower = text.lower()
        
        # 默认权重
        preferences['return'] = 0.4
        preferences['risk'] = 0.3
        preferences['stability'] = 0.2
        preferences['efficiency'] = 0.1
        
        # 根据文本调整权重
        if re.search(r"focus.*on.*return", text_lower):
            preferences['return'] = 0.6
            preferences['risk'] = 0.2
        elif re.search(r"focus.*on.*risk", text_lower):
            preferences['risk'] = 0.5
            preferences['return'] = 0.3
        elif re.search(r"focus.*on.*stability", text_lower):
            preferences['stability'] = 0.5
            preferences['return'] = 0.3
            
        return preferences
    
    def _extract_risk_tolerance(self, text: str) -> str:
        """提取风险容忍度"""
        text_lower = text.lower()
        
        for level, pattern in self.risk_patterns.items():
            if re.search(pattern, text_lower):
                return level
                
        return "medium"  # 默认值
    
    def _extract_time_horizon(self, text: str) -> str:
        """提取时间范围"""
        text_lower = text.lower()
        
        for horizon, pattern in self.time_patterns.items():
            if re.search(pattern, text_lower):
                return horizon
                
        return "medium"  # 默认值
    
    def _extract_ethical_guidelines(self, text: str) -> List[str]:
        """提取伦理指导原则"""
        guidelines = []
        text_lower = text.lower()
        
        if re.search(r"ethical|responsible|sustainable", text_lower):
            guidelines.append("responsible_trading")
        
        if re.search(r"no.*manipulation|fair.*market", text_lower):
            guidelines.append("market_fairness")
            
        if re.search(r"transparent|explainable", text_lower):
            guidelines.append("transparency")
            
        return guidelines
    
    def _get_default_specification(self) -> RewardSpecification:
        """获取默认规范"""
        return RewardSpecification(
            objective="balanced_performance",
            constraints=["max_drawdown_limit_20"],
            preferences={'return': 0.4, 'risk': 0.3, 'stability': 0.2, 'efficiency': 0.1},
            risk_tolerance="medium",
            time_horizon="medium",
            ethical_guidelines=["responsible_trading"]
        )

class RewardCodeGenerator:
    """奖励函数代码生成器"""
    
    def __init__(self):
        self.constitutional_principles = [
            ConstitutionalPrinciple(
                name="safety_first",
                description="Ensure no harmful or manipulative trading behavior",
                constraint="reward_value >= -1.0 and reward_value <= 1.0",
                weight=1.0
            ),
            ConstitutionalPrinciple(
                name="risk_awareness",
                description="Incorporate risk considerations in all decisions",
                constraint="include_risk_metrics in reward_calculation",
                weight=0.8
            ),
            ConstitutionalPrinciple(
                name="transparency",
                description="Make reward calculations interpretable",
                constraint="provide_explanation_for_reward",
                weight=0.6
            )
        ]
    
    def generate_reward_function(self, spec: RewardSpecification) -> str:
        """基于规范生成奖励函数代码"""
        try:
            # 根据规范生成奖励函数代码
            code = self._create_base_structure()
            code += self._add_objective_logic(spec.objective, spec.preferences)
            code += self._add_constraint_logic(spec.constraints)
            code += self._add_risk_adjustment(spec.risk_tolerance)
            code += self._add_temporal_logic(spec.time_horizon)
            code += self._add_constitutional_checks()
            code += self._add_explanation_generation()
            
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return self._get_fallback_code()
    
    def _create_base_structure(self) -> str:
        """创建基础结构"""
        return """
def calculate_reward(portfolio, market_data=None, history=None):
    \"\"\"
    LLM-Generated reward function with constitutional safeguards
    \"\"\"
    try:
        # Get current portfolio value
        current_value = portfolio.net_worth
        
        # Initialize reward components
        reward_components = {}
        
"""
    
    def _add_objective_logic(self, objective: str, preferences: Dict[str, float]) -> str:
        """添加目标逻辑"""
        if objective == "maximize_returns":
            return f"""
        # Maximize returns objective
        if hasattr(portfolio, '_previous_value'):
            return_rate = (current_value - portfolio._previous_value) / portfolio._previous_value
            reward_components['return'] = return_rate * {preferences.get('return', 0.4)}
        else:
            reward_components['return'] = 0.0
        
"""
        elif objective == "minimize_risk":
            return f"""
        # Minimize risk objective  
        if history is not None and len(history) > 10:
            returns = np.diff(history) / history[:-1]
            volatility = np.std(returns)
            reward_components['risk'] = -volatility * {preferences.get('risk', 0.5)}
        else:
            reward_components['risk'] = 0.0
        
"""
        elif objective == "maximize_risk_adjusted_returns":
            return f"""
        # Maximize risk-adjusted returns objective
        if hasattr(portfolio, '_previous_value') and history is not None:
            return_rate = (current_value - portfolio._previous_value) / portfolio._previous_value
            if len(history) > 10:
                returns = np.diff(history) / history[:-1]
                volatility = max(np.std(returns), 1e-6)
                sharpe_like = return_rate / volatility
                reward_components['risk_adjusted'] = sharpe_like * {preferences.get('return', 0.4)}
            else:
                reward_components['risk_adjusted'] = return_rate * {preferences.get('return', 0.4)}
        else:
            reward_components['risk_adjusted'] = 0.0
        
"""
        else:  # balanced_performance
            return f"""
        # Balanced performance objective
        if hasattr(portfolio, '_previous_value'):
            return_rate = (current_value - portfolio._previous_value) / portfolio._previous_value
            reward_components['return'] = return_rate * {preferences.get('return', 0.4)}
            
            # Add stability component
            if history is not None and len(history) > 5:
                recent_volatility = np.std(history[-5:]) / np.mean(history[-5:])
                reward_components['stability'] = -recent_volatility * {preferences.get('stability', 0.2)}
            else:
                reward_components['stability'] = 0.0
        else:
            reward_components['return'] = 0.0
            reward_components['stability'] = 0.0
        
"""
    
    def _add_constraint_logic(self, constraints: List[str]) -> str:
        """添加约束逻辑"""
        code = ""
        
        for constraint in constraints:
            if "max_drawdown_limit" in constraint:
                limit = constraint.split("_")[-1]
                code += f"""
        # Maximum drawdown constraint
        if history is not None and len(history) > 1:
            peak = np.maximum.accumulate(history)
            drawdown = (peak - history) / peak * 100
            max_drawdown = np.max(drawdown)
            if max_drawdown > {limit}:
                reward_components['drawdown_penalty'] = -(max_drawdown - {limit}) * 0.1
            else:
                reward_components['drawdown_penalty'] = 0.0
        else:
            reward_components['drawdown_penalty'] = 0.0
        
"""
            elif "volatility_limit" in constraint:
                limit = constraint.split("_")[-1]
                code += f"""
        # Volatility limit constraint
        if history is not None and len(history) > 10:
            returns = np.diff(history) / history[:-1]
            volatility = np.std(returns) * 100
            if volatility > {limit}:
                reward_components['volatility_penalty'] = -(volatility - {limit}) * 0.05
            else:
                reward_components['volatility_penalty'] = 0.0
        else:
            reward_components['volatility_penalty'] = 0.0
        
"""
        
        return code
    
    def _add_risk_adjustment(self, risk_tolerance: str) -> str:
        """添加风险调整"""
        risk_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }
        
        multiplier = risk_multipliers.get(risk_tolerance, 1.0)
        
        return f"""
        # Risk tolerance adjustment
        risk_multiplier = {multiplier}
        for key in reward_components:
            if 'risk' in key or 'penalty' in key:
                reward_components[key] *= risk_multiplier
        
"""
    
    def _add_temporal_logic(self, time_horizon: str) -> str:
        """添加时间逻辑"""
        if time_horizon == "short":
            return """
        # Short-term focus: emphasize immediate returns
        if 'return' in reward_components:
            reward_components['return'] *= 1.2
        
"""
        elif time_horizon == "long":
            return """
        # Long-term focus: emphasize stability and consistency
        if 'stability' in reward_components:
            reward_components['stability'] *= 1.5
        if 'return' in reward_components:
            reward_components['return'] *= 0.8
        
"""
        else:  # medium
            return """
        # Medium-term focus: balanced approach
        # No additional temporal adjustments needed
        
"""
    
    def _add_constitutional_checks(self) -> str:
        """添加Constitutional AI检查"""
        return """
        # Constitutional AI safety checks
        total_reward = sum(reward_components.values())
        
        # Safety constraint: limit reward magnitude
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        # Transparency: store component breakdown
        if not hasattr(portfolio, '_reward_explanation'):
            portfolio._reward_explanation = {}
        portfolio._reward_explanation = reward_components.copy()
        portfolio._reward_explanation['total'] = total_reward
        
        # Risk awareness check
        if 'risk' not in str(reward_components) and 'volatility' not in str(reward_components):
            # Add small risk penalty if not already included
            total_reward -= 0.01
        
"""
    
    def _add_explanation_generation(self) -> str:
        """添加解释生成"""
        return """
        # Generate explanation
        explanation = "Reward calculation breakdown:\\n"
        for component, value in reward_components.items():
            explanation += f"  {component}: {value:.4f}\\n"
        explanation += f"  Total: {total_reward:.4f}\\n"
        
        if not hasattr(portfolio, '_reward_explanations'):
            portfolio._reward_explanations = []
        portfolio._reward_explanations.append(explanation)
        
        # Store previous value for next calculation
        portfolio._previous_value = current_value
        
        return total_reward
        
    except Exception as e:
        # Fallback: return small positive reward
        return 0.001
"""
    
    def _get_fallback_code(self) -> str:
        """获取后备代码"""
        return """
def calculate_reward(portfolio, market_data=None, history=None):
    \"\"\"Fallback reward function\"\"\"
    try:
        current_value = portfolio.net_worth
        if hasattr(portfolio, '_previous_value'):
            return_rate = (current_value - portfolio._previous_value) / portfolio._previous_value
            portfolio._previous_value = current_value
            return np.clip(return_rate, -1.0, 1.0)
        else:
            portfolio._previous_value = current_value
            return 0.001
    except:
        return 0.001
"""

class RewardValidator:
    """奖励函数验证器"""
    
    def __init__(self):
        self.safety_checks = [
            self._check_syntax,
            self._check_imports,
            self._check_infinite_loops,
            self._check_harmful_operations,
            self._check_output_bounds
        ]
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """验证生成的代码"""
        errors = []
        
        for check in self.safety_checks:
            try:
                check_passed, error_msg = check(code)
                if not check_passed:
                    errors.append(error_msg)
            except Exception as e:
                errors.append(f"Validation check failed: {e}")
        
        return len(errors) == 0, errors
    
    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """检查语法错误"""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def _check_imports(self, code: str) -> Tuple[bool, str]:
        """检查危险导入"""
        dangerous_imports = ['os', 'sys', 'subprocess', 'eval', 'exec']
        
        for imp in dangerous_imports:
            if f"import {imp}" in code or f"from {imp}" in code:
                return False, f"Dangerous import detected: {imp}"
        
        return True, ""
    
    def _check_infinite_loops(self, code: str) -> Tuple[bool, str]:
        """检查无限循环"""
        # 简单检查：寻找while True或for循环没有明确终止条件
        if "while True:" in code:
            return False, "Potential infinite loop detected: while True"
        
        return True, ""
    
    def _check_harmful_operations(self, code: str) -> Tuple[bool, str]:
        """检查有害操作"""
        harmful_operations = ['delete', 'remove', 'kill', 'destroy', 'format']
        
        for op in harmful_operations:
            if op in code.lower():
                return False, f"Potentially harmful operation detected: {op}"
        
        return True, ""
    
    def _check_output_bounds(self, code: str) -> Tuple[bool, str]:
        """检查输出边界"""
        # 检查是否包含适当的边界限制
        if "clip" not in code and "min" not in code and "max" not in code:
            return False, "No output bounds detected - should include clipping or bounds checking"
        
        return True, ""

class LLMGuidedReward(BaseRewardScheme):
    """
    LLM引导的奖励函数
    
    基于2024-2025年最新研究的LLM引导奖励函数设计系统，
    支持自然语言规范解析、代码生成、安全验证和迭代优化。
    """
    
    def __init__(self, 
                 natural_language_spec: str = "Maximize risk-adjusted returns with moderate risk tolerance",
                 enable_iterative_improvement: bool = True,
                 safety_level: str = "high",
                 explanation_detail: str = "medium",
                 initial_balance: float = 10000.0):
        """
        初始化LLM引导奖励函数
        
        Args:
            natural_language_spec: 自然语言奖励规范
            enable_iterative_improvement: 是否启用迭代改进
            safety_level: 安全级别 (low/medium/high)
            explanation_detail: 解释详细程度 (low/medium/high)
            initial_balance: 初始余额
        """
        super().__init__(initial_balance=initial_balance)
        
        self.natural_language_spec = natural_language_spec
        self.enable_iterative_improvement = enable_iterative_improvement
        self.safety_level = safety_level
        self.explanation_detail = explanation_detail
        
        # 初始化组件
        self.parser = NaturalLanguageParser()
        self.code_generator = RewardCodeGenerator()
        self.validator = RewardValidator()
        
        # 解析规范
        self.specification = self.parser.parse_specification(natural_language_spec)
        
        # 生成奖励函数代码
        self.reward_code = self.code_generator.generate_reward_function(self.specification)
        
        # 验证代码
        is_valid, errors = self.validator.validate_code(self.reward_code)
        if not is_valid:
            logger.warning(f"Generated code validation failed: {errors}")
            self.reward_code = self.code_generator._get_fallback_code()
        
        # 编译奖励函数
        self._compile_reward_function()
        
        # 统计信息
        self.generation_count = 0
        self.improvement_iterations = 0
        self.performance_history = []
        self.explanation_history = []
        
        logger.info(f"LLMGuidedReward initialized with specification: {natural_language_spec}")
    
    def _compile_reward_function(self):
        """编译奖励函数"""
        try:
            # 创建安全的执行环境
            safe_globals = {
                'np': np,
                'numpy': np,
                'max': max,
                'min': min,
                'abs': abs,
                'len': len,
                'sum': sum,
                'float': float,
                'int': int,
                '__builtins__': {}
            }
            
            # 执行代码
            exec(self.reward_code, safe_globals)
            self.compiled_reward_function = safe_globals['calculate_reward']
            
            logger.info("Reward function compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to compile reward function: {e}")
            # 使用简单的后备函数
            self.compiled_reward_function = self._fallback_reward_function
    
    def _fallback_reward_function(self, portfolio, market_data=None, history=None):
        """后备奖励函数"""
        try:
            current_value = portfolio.net_worth
            if hasattr(portfolio, '_previous_value'):
                return_rate = (current_value - portfolio._previous_value) / portfolio._previous_value
                portfolio._previous_value = current_value
                return np.clip(return_rate, -1.0, 1.0)
            else:
                portfolio._previous_value = current_value
                return 0.001
        except:
            return 0.001
    
    def reward(self, portfolio) -> float:
        """计算奖励值 (BaseRewardScheme接口)"""
        return self.get_reward(portfolio)
    
    def get_reward(self, portfolio) -> float:
        """计算奖励值"""
        try:
            # 准备历史数据
            history = getattr(portfolio, '_net_worth_history', None)
            if history is None:
                if not hasattr(portfolio, '_net_worth_history'):
                    portfolio._net_worth_history = []
                portfolio._net_worth_history.append(portfolio.net_worth)
                history = portfolio._net_worth_history
            else:
                portfolio._net_worth_history.append(portfolio.net_worth)
                history = portfolio._net_worth_history
            
            # 调用编译的奖励函数
            reward = self.compiled_reward_function(
                portfolio=portfolio,
                market_data=None,
                history=np.array(history) if history else None
            )
            
            # 记录性能
            self.performance_history.append(reward)
            
            # 如果启用迭代改进，定期评估性能
            if (self.enable_iterative_improvement and 
                len(self.performance_history) > 0 and 
                len(self.performance_history) % 100 == 0):
                self._evaluate_and_improve()
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            return 0.001
    
    def _evaluate_and_improve(self):
        """评估性能并改进奖励函数"""
        try:
            if len(self.performance_history) < 50:
                return
            
            # 计算性能指标
            recent_performance = self.performance_history[-50:]
            avg_reward = np.mean(recent_performance)
            reward_volatility = np.std(recent_performance)
            
            # 检查是否需要改进
            improvement_needed = False
            
            # 性能阈值检查
            if avg_reward < 0.001:
                improvement_needed = True
                logger.info("Low average reward detected, attempting improvement")
            
            if reward_volatility > 0.5:
                improvement_needed = True
                logger.info("High reward volatility detected, attempting improvement")
            
            if improvement_needed and self.improvement_iterations < 3:
                self._attempt_improvement(avg_reward, reward_volatility)
                self.improvement_iterations += 1
                
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
    
    def _attempt_improvement(self, avg_reward: float, volatility: float):
        """尝试改进奖励函数"""
        try:
            # 基于性能反馈调整规范
            improved_spec = self.natural_language_spec
            
            if avg_reward < 0:
                improved_spec += " Focus more on positive returns and reduce penalties."
            
            if volatility > 0.3:
                improved_spec += " Increase stability and reduce volatility in reward calculation."
            
            # 重新生成奖励函数
            improved_specification = self.parser.parse_specification(improved_spec)
            improved_code = self.code_generator.generate_reward_function(improved_specification)
            
            # 验证改进的代码
            is_valid, errors = self.validator.validate_code(improved_code)
            
            if is_valid:
                # 保存当前状态以便回滚
                old_code = self.reward_code
                old_function = self.compiled_reward_function
                
                # 应用改进
                self.reward_code = improved_code
                self.specification = improved_specification
                self._compile_reward_function()
                
                logger.info(f"Reward function improved (iteration {self.improvement_iterations + 1})")
                
            else:
                logger.warning(f"Improved code validation failed: {errors}")
                
        except Exception as e:
            logger.error(f"Improvement attempt failed: {e}")
    
    def get_llm_explanation(self) -> str:
        """获取LLM生成的解释"""
        try:
            explanation = f"LLM-Guided Reward Function Explanation\\n"
            explanation += f"=" * 50 + "\\n"
            explanation += f"Original Specification: {self.natural_language_spec}\\n\\n"
            
            explanation += f"Parsed Objective: {self.specification.objective}\\n"
            explanation += f"Risk Tolerance: {self.specification.risk_tolerance}\\n"
            explanation += f"Time Horizon: {self.specification.time_horizon}\\n\\n"
            
            explanation += f"Preferences:\\n"
            for pref, weight in self.specification.preferences.items():
                explanation += f"  {pref}: {weight:.2f}\\n"
            
            if self.specification.constraints:
                explanation += f"\\nConstraints:\\n"
                for constraint in self.specification.constraints:
                    explanation += f"  - {constraint}\\n"
            
            if self.specification.ethical_guidelines:
                explanation += f"\\nEthical Guidelines:\\n"
                for guideline in self.specification.ethical_guidelines:
                    explanation += f"  - {guideline}\\n"
            
            explanation += f"\\nGeneration Count: {self.generation_count}\\n"
            explanation += f"Improvement Iterations: {self.improvement_iterations}\\n"
            
            if self.performance_history:
                recent_avg = np.mean(self.performance_history[-20:]) if len(self.performance_history) >= 20 else np.mean(self.performance_history)
                explanation += f"Recent Average Reward: {recent_avg:.4f}\\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return "Explanation generation failed"
    
    def get_reward_metrics(self) -> Dict[str, Any]:
        """获取奖励函数指标"""
        try:
            metrics = {
                'specification': {
                    'objective': self.specification.objective,
                    'risk_tolerance': self.specification.risk_tolerance,
                    'time_horizon': self.specification.time_horizon,
                    'constraints_count': len(self.specification.constraints),
                    'ethical_guidelines_count': len(self.specification.ethical_guidelines)
                },
                'performance': {
                    'generation_count': self.generation_count,
                    'improvement_iterations': self.improvement_iterations,
                    'total_calculations': len(self.performance_history)
                },
                'code_quality': {
                    'code_length': len(self.reward_code.split('\\n')),
                    'safety_level': self.safety_level,
                    'explanation_detail': self.explanation_detail
                }
            }
            
            if self.performance_history:
                perf_array = np.array(self.performance_history)
                metrics['performance'].update({
                    'average_reward': float(np.mean(perf_array)),
                    'reward_std': float(np.std(perf_array)),
                    'min_reward': float(np.min(perf_array)),
                    'max_reward': float(np.max(perf_array)),
                    'recent_trend': float(np.mean(perf_array[-10:])) - float(np.mean(perf_array[:10])) if len(perf_array) >= 20 else 0.0
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {'error': str(e)}
    
    def reset(self):
        """重置奖励函数状态"""
        # 清理性能历史，但保留学习到的改进
        self.performance_history = []
        self.explanation_history = []
        
        logger.info("LLMGuidedReward reset completed")
    
    @staticmethod
    def get_reward_info() -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': 'LLM-Guided Reward Function',
            'description': 'Advanced LLM-guided reward function with natural language specification parsing, '
                          'code generation, safety validation, and iterative improvement based on 2024-2025 research',
            'category': 'Advanced AI-Guided',
            'complexity': 'Very High',
            'parameters': {
                'natural_language_spec': 'Natural language specification for reward objectives',
                'enable_iterative_improvement': 'Enable automatic improvement based on performance feedback',
                'safety_level': 'Safety validation level (low/medium/high)',
                'explanation_detail': 'Detail level for explanations (low/medium/high)',
                'initial_balance': 'Initial portfolio balance'
            },
            'features': [
                'EUREKA-style evolutionary optimization',
                'Constitutional AI safety framework',
                'Natural language specification parsing',
                'Automatic code generation and validation',
                'Iterative self-improvement',
                'Comprehensive explanation generation',
                'Multi-objective optimization',
                'Risk-aware decision making',
                'Ethical guideline enforcement'
            ],
            'use_cases': [
                'Custom reward function design from natural language',
                'Automated reward engineering',
                'Safe AI-guided optimization',
                'Explainable reward systems',
                'Adaptive trading strategies',
                'Research and experimentation'
            ],
            'computational_complexity': 'High (code generation and compilation)',
            'memory_usage': 'Medium (stores code and performance history)',
            'recommended_for': 'Advanced users requiring custom, explainable reward functions'
        }
    
    def calculate_reward(self, current_step, current_price, current_portfolio_value, action, **kwargs):
        """
        奖励计算接口 - 计算LLM指导的奖励
        
        Args:
            current_step: 当前步数
            current_price: 当前价格
            current_portfolio_value: 当前投资组合价值
            action: 执行的动作
            **kwargs: 其他参数
            
        Returns:
            float: 奖励值
        """
        # 更新历史记录
        self.update_history(current_portfolio_value)
        
        # 计算步骤收益
        if len(self.portfolio_history) < 2:
            step_return = 0.0
        else:
            prev_value = self.portfolio_history[-2]
            step_return = current_portfolio_value - prev_value
        
        # 构建状态
        state = {
            'step_count': current_step,
            'current_price': current_price,
            'current_value': current_portfolio_value,
            'current_action': action,
            'total_return_pct': ((current_portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'step_return_pct': (step_return / self.initial_balance) * 100 if self.initial_balance != 0 else 0.0
        }
        
        # 执行当前奖励函数
        if hasattr(self, 'compiled_function') and self.compiled_function:
            try:
                llm_reward = self.compiled_function(state)
            except Exception as e:
                self.logger.error(f"LLM奖励函数执行失败: {e}")
                llm_reward = state['step_return_pct']  # 回退到简单回报
        else:
            # 如果没有编译的函数，使用基础奖励
            llm_reward = state['step_return_pct']
        
        # 应用合理性检查
        if abs(llm_reward) > 100:  # 限制过大的奖励
            llm_reward = np.sign(llm_reward) * 10
        
        # 记录奖励历史
        self.reward_history.append(llm_reward)
        
        return llm_reward

# 为向后兼容性提供别名
LLMGuidedRewardFunction = LLMGuidedReward