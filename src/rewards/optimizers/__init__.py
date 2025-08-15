"""
奖励优化器模块 - Reward Optimizers Module

提供奖励函数组合和优化功能。

主要组件：
- RewardOptimizer: 奖励组合优化器基类
- GridSearchOptimizer: 网格搜索优化器
- GeneticOptimizer: 遗传算法优化器
- RewardCombination: 奖励组合类
- OptimizationResult: 优化结果类

使用示例：
    ```python
    from src.rewards.optimizers import optimize_reward_combination
    
    # 优化奖励组合
    result = optimize_reward_combination(
        reward_functions=reward_funcs,
        historical_data=data,
        method="grid_search",
        objective="sharpe_ratio"
    )
    
    # 创建优化的组合
    combination = create_optimal_reward_combination(
        reward_functions=reward_funcs,
        weights=result.optimal_weights
    )
    ```
"""

from .reward_optimizer import (
    # 核心类
    BaseRewardOptimizer,
    GridSearchOptimizer,
    GeneticOptimizer,
    RewardCombination,
    RewardOptimizerFactory,
    
    # 枚举
    OptimizationMethod,
    OptimizationObjective,
    
    # 数据类
    OptimizationConstraints,
    OptimizationResult,
    
    # 便利函数
    optimize_reward_combination,
    create_optimal_reward_combination
)


# 导出主要接口
__all__ = [
    # 核心类
    "BaseRewardOptimizer",
    "GridSearchOptimizer", 
    "GeneticOptimizer",
    "RewardCombination",
    "RewardOptimizerFactory",
    
    # 枚举
    "OptimizationMethod",
    "OptimizationObjective",
    
    # 数据类
    "OptimizationConstraints",
    "OptimizationResult",
    
    # 便利函数
    "optimize_reward_combination",
    "create_optimal_reward_combination"
]


def get_available_methods():
    """获取所有可用的优化方法"""
    return [method.value for method in OptimizationMethod]


def get_available_objectives():
    """获取所有可用的优化目标"""
    return [obj.value for obj in OptimizationObjective]


def get_method_info(method: str) -> dict:
    """获取优化方法的详细信息"""
    
    method_info = {
        "grid_search": {
            "name": "Grid Search Optimizer",
            "description": "Exhaustive search over parameter grid",
            "pros": [
                "Guaranteed to find global optimum in discrete space",
                "Simple and reliable",
                "Good for small parameter spaces"
            ],
            "cons": [
                "Exponential complexity with parameters",
                "Not suitable for continuous optimization",
                "Can be very slow for large spaces"
            ],
            "best_for": "Small number of rewards (2-4), quick optimization",
            "parameters": ["grid_resolution"]
        },
        
        "genetic": {
            "name": "Genetic Algorithm Optimizer", 
            "description": "Evolutionary optimization algorithm",
            "pros": [
                "Good for complex, non-linear problems",
                "Can escape local optima",
                "Handles constraints well",
                "Scalable to many parameters"
            ],
            "cons": [
                "No guarantee of global optimum",
                "Requires tuning of GA parameters",
                "Can be slow to converge"
            ],
            "best_for": "Large number of rewards (5+), complex optimization landscapes",
            "parameters": ["population_size", "max_generations", "mutation_rate", "crossover_rate"]
        },
        
        "random_search": {
            "name": "Random Search Optimizer",
            "description": "Random sampling of parameter space",
            "pros": [
                "Simple implementation",
                "Good baseline method",
                "Parallelizable"
            ],
            "cons": [
                "No guarantee of good solution",
                "Inefficient for structured problems"
            ],
            "best_for": "Quick baseline, exploration of parameter space",
            "parameters": ["n_iterations"]
        },
        
        "bayesian": {
            "name": "Bayesian Optimization",
            "description": "Model-based optimization using Gaussian processes",
            "pros": [
                "Very efficient for expensive evaluations",
                "Balances exploration vs exploitation",
                "Good for continuous parameters"
            ],
            "cons": [
                "Complex implementation",
                "Requires careful tuning",
                "Not implemented yet"
            ],
            "best_for": "Expensive objective functions, continuous optimization",
            "parameters": ["acquisition_function", "n_initial_points"]
        }
    }
    
    return method_info.get(method, {"error": "Unknown optimization method"})


def get_objective_info(objective: str) -> dict:
    """获取优化目标的详细信息"""
    
    objective_info = {
        "sharpe_ratio": {
            "name": "Sharpe Ratio",
            "description": "Risk-adjusted return metric",
            "formula": "(Mean Return - Risk Free Rate) / Standard Deviation",
            "range": "(-∞, +∞), higher is better",
            "pros": ["Risk-adjusted", "Widely used", "Easy to interpret"],
            "cons": ["Assumes normal returns", "Sensitive to outliers"],
            "best_for": "Balanced risk-return optimization"
        },
        
        "total_return": {
            "name": "Total Return",
            "description": "Overall investment return",
            "formula": "(Final Value - Initial Value) / Initial Value",
            "range": "(-1, +∞), higher is better",
            "pros": ["Simple", "Direct measure of profitability"],
            "cons": ["Ignores risk", "Time-dependent"],
            "best_for": "Pure profit maximization"
        },
        
        "max_drawdown": {
            "name": "Maximum Drawdown",
            "description": "Largest peak-to-trough decline",
            "formula": "Max((Peak - Trough) / Peak)",
            "range": "[0, 1], lower is better",
            "pros": ["Risk measure", "Easy to understand"],
            "cons": ["Backward-looking", "Doesn't consider frequency"],
            "best_for": "Risk minimization, capital preservation"
        },
        
        "calmar_ratio": {
            "name": "Calmar Ratio",
            "description": "Return vs maximum drawdown",
            "formula": "Annual Return / Maximum Drawdown",
            "range": "(-∞, +∞), higher is better",
            "pros": ["Risk-adjusted", "Focuses on downside risk"],
            "cons": ["Sensitive to time period", "Can be unstable"],
            "best_for": "Downside risk-adjusted optimization"
        },
        
        "multi_objective": {
            "name": "Multi-Objective",
            "description": "Combination of multiple metrics",
            "formula": "Weighted sum of normalized objectives",
            "range": "Depends on components",
            "pros": ["Comprehensive", "Balances multiple goals"],
            "cons": ["Complex", "Requires weight tuning"],
            "best_for": "Comprehensive optimization with multiple goals"
        }
    }
    
    return objective_info.get(objective, {"error": "Unknown optimization objective"})


# 模块级配置
VERSION = "1.0.0"
AUTHOR = "Reward Optimizer Team"
DESCRIPTION = "Advanced reward combination and optimization tools"