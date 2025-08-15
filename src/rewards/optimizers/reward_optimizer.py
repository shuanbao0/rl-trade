"""
奖励组合优化器 - Reward Combination Optimizer

实现奖励函数的组合、权重优化和动态调整功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import ParameterGrid
import logging
from datetime import datetime, timedelta
import warnings

from ..core.reward_context import RewardContext
from ..core.base_reward import BaseReward
from ..enums.market_types import MarketType
from ..enums.time_granularities import TimeGranularity
from ..enums.risk_profiles import RiskProfile


class OptimizationMethod(Enum):
    """优化方法"""
    GRID_SEARCH = "grid_search"  # 网格搜索
    RANDOM_SEARCH = "random_search"  # 随机搜索
    BAYESIAN = "bayesian"  # 贝叶斯优化
    GENETIC = "genetic"  # 遗传算法
    GRADIENT_BASED = "gradient_based"  # 梯度优化
    ENSEMBLE = "ensemble"  # 集成方法


class OptimizationObjective(Enum):
    """优化目标"""
    SHARPE_RATIO = "sharpe_ratio"  # 夏普比率
    TOTAL_RETURN = "total_return"  # 总收益率
    MAX_DRAWDOWN = "max_drawdown"  # 最大回撤
    CALMAR_RATIO = "calmar_ratio"  # 卡尔马比率
    SORTINO_RATIO = "sortino_ratio"  # 索蒂诺比率
    INFORMATION_RATIO = "information_ratio"  # 信息比率
    MULTI_OBJECTIVE = "multi_objective"  # 多目标优化


@dataclass
class OptimizationConstraints:
    """优化约束条件"""
    max_weight: float = 1.0  # 单个奖励最大权重
    min_weight: float = 0.0  # 单个奖励最小权重
    weight_sum: float = 1.0  # 权重总和
    max_complexity: Optional[int] = None  # 最大复杂度
    max_correlation: Optional[float] = None  # 最大相关性
    diversity_requirement: Optional[float] = None  # 多样性要求
    performance_threshold: Optional[float] = None  # 性能阈值
    stability_requirement: Optional[float] = None  # 稳定性要求


@dataclass
class OptimizationResult:
    """优化结果"""
    optimal_weights: Dict[str, float]
    optimal_parameters: Dict[str, Dict[str, Any]]
    objective_value: float
    objective_history: List[float]
    convergence_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    robustness_analysis: Dict[str, Any]
    execution_time: float
    validation_results: Dict[str, Any]


@dataclass
class RewardCombination:
    """奖励组合"""
    reward_functions: Dict[str, BaseReward]
    weights: Dict[str, float]
    parameters: Dict[str, Dict[str, Any]]
    combination_method: str = "weighted_sum"  # 组合方法
    normalization: str = "standard"  # 标准化方法
    meta_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_combined_reward(self, context: RewardContext) -> float:
        """计算组合奖励"""
        if self.combination_method == "weighted_sum":
            return self._weighted_sum(context)
        elif self.combination_method == "weighted_geometric_mean":
            return self._weighted_geometric_mean(context)
        elif self.combination_method == "ensemble":
            return self._ensemble_combination(context)
        elif self.combination_method == "adaptive":
            return self._adaptive_combination(context)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def _weighted_sum(self, context: RewardContext) -> float:
        """加权求和"""
        total_reward = 0.0
        total_weight = 0.0
        
        for name, reward_func in self.reward_functions.items():
            if name in self.weights and self.weights[name] > 0:
                try:
                    # 更新奖励函数参数
                    if name in self.parameters:
                        for param_name, param_value in self.parameters[name].items():
                            if hasattr(reward_func, param_name):
                                setattr(reward_func, param_name, param_value)
                    
                    # 计算奖励
                    reward_value = reward_func.calculate(context)
                    weight = self.weights[name]
                    
                    # 标准化处理
                    if self.normalization == "standard":
                        reward_value = self._standardize_reward(name, reward_value)
                    elif self.normalization == "minmax":
                        reward_value = self._minmax_normalize_reward(name, reward_value)
                    
                    total_reward += weight * reward_value
                    total_weight += weight
                    
                except Exception as e:
                    logging.warning(f"Error calculating reward {name}: {e}")
                    continue
        
        return total_reward / total_weight if total_weight > 0 else 0.0
    
    def _weighted_geometric_mean(self, context: RewardContext) -> float:
        """加权几何平均"""
        product = 1.0
        total_weight = 0.0
        
        for name, reward_func in self.reward_functions.items():
            if name in self.weights and self.weights[name] > 0:
                try:
                    reward_value = reward_func.calculate(context)
                    weight = self.weights[name]
                    
                    # 确保值为正（几何平均需要）
                    reward_value = max(0.01, reward_value + 1.0)  # 偏移确保正值
                    
                    product *= reward_value ** weight
                    total_weight += weight
                    
                except Exception as e:
                    logging.warning(f"Error calculating reward {name}: {e}")
                    continue
        
        return product ** (1.0 / total_weight) - 1.0 if total_weight > 0 else 0.0
    
    def _ensemble_combination(self, context: RewardContext) -> float:
        """集成组合方法"""
        # 获取所有奖励值
        rewards = []
        weights = []
        
        for name, reward_func in self.reward_functions.items():
            if name in self.weights and self.weights[name] > 0:
                try:
                    reward_value = reward_func.calculate(context)
                    rewards.append(reward_value)
                    weights.append(self.weights[name])
                except:
                    continue
        
        if not rewards:
            return 0.0
        
        rewards = np.array(rewards)
        weights = np.array(weights)
        
        # 多种组合方式的集成
        weighted_mean = np.average(rewards, weights=weights)
        median_reward = np.median(rewards)
        max_reward = np.max(rewards)
        
        # 根据元参数决定最终组合
        ensemble_weights = self.meta_parameters.get('ensemble_weights', [0.7, 0.2, 0.1])
        
        return (ensemble_weights[0] * weighted_mean + 
                ensemble_weights[1] * median_reward + 
                ensemble_weights[2] * max_reward)
    
    def _adaptive_combination(self, context: RewardContext) -> float:
        """自适应组合方法"""
        # 根据市场状态和性能历史动态调整权重
        base_reward = self._weighted_sum(context)
        
        # 获取自适应参数
        adaptation_rate = self.meta_parameters.get('adaptation_rate', 0.1)
        volatility_factor = self.meta_parameters.get('volatility_factor', 1.0)
        
        # 简化的自适应调整
        market_volatility = getattr(context, 'market_volatility', 0.02)
        adjustment = 1.0 + adaptation_rate * (market_volatility - 0.02) * volatility_factor
        
        return base_reward * adjustment
    
    def _standardize_reward(self, reward_name: str, reward_value: float) -> float:
        """标准化奖励值"""
        # 简化标准化（实际应该基于历史统计）
        return np.tanh(reward_value)  # 限制在[-1, 1]范围
    
    def _minmax_normalize_reward(self, reward_name: str, reward_value: float) -> float:
        """MinMax标准化"""
        # 简化实现（实际应该基于历史最值）
        return max(0, min(1, (reward_value + 1) / 2))


class BaseRewardOptimizer(ABC):
    """奖励优化器基类"""
    
    def __init__(
        self,
        method: OptimizationMethod,
        objective: OptimizationObjective,
        constraints: OptimizationConstraints
    ):
        self.method = method
        self.objective = objective
        self.constraints = constraints
        # 使用统一日志系统
        from ...utils.logger import get_logger
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # 优化历史
        self.optimization_history: List[OptimizationResult] = []
        
        # 性能缓存
        self.performance_cache: Dict[str, Dict[str, float]] = {}
    
    @abstractmethod
    def optimize(
        self,
        reward_functions: Dict[str, BaseReward],
        historical_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> OptimizationResult:
        """执行优化"""
        pass
    
    @abstractmethod
    def evaluate_combination(
        self,
        combination: RewardCombination,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """评估奖励组合"""
        pass
    
    def validate_constraints(self, weights: Dict[str, float]) -> bool:
        """验证约束条件"""
        # 检查权重范围
        for weight in weights.values():
            if weight < self.constraints.min_weight or weight > self.constraints.max_weight:
                return False
        
        # 检查权重总和
        weight_sum = sum(weights.values())
        if abs(weight_sum - self.constraints.weight_sum) > 1e-6:
            return False
        
        return True
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """标准化权重"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {k: 1.0 / len(weights) for k in weights.keys()}
        
        normalized = {k: v / total_weight for k, v in weights.items()}
        
        # 应用约束
        for k in normalized.keys():
            normalized[k] = max(self.constraints.min_weight, 
                              min(self.constraints.max_weight, normalized[k]))
        
        # 重新标准化
        total_weight = sum(normalized.values())
        if total_weight > 0:
            normalized = {k: v / total_weight for k, v in normalized.items()}
        
        return normalized


class GridSearchOptimizer(BaseRewardOptimizer):
    """网格搜索优化器"""
    
    def __init__(
        self,
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
        grid_resolution: int = 10
    ):
        super().__init__(OptimizationMethod.GRID_SEARCH, objective, constraints)
        self.grid_resolution = grid_resolution
    
    def optimize(
        self,
        reward_functions: Dict[str, BaseReward],
        historical_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> OptimizationResult:
        """执行网格搜索优化"""
        
        start_time = datetime.now()
        
        # 生成权重网格
        weight_grid = self._generate_weight_grid(list(reward_functions.keys()))
        
        best_combination = None
        best_objective = -float('inf')
        objective_history = []
        
        self.logger.info(f"Starting grid search with {len(weight_grid)} combinations")
        
        for i, weights in enumerate(weight_grid):
            if not self.validate_constraints(weights):
                continue
            
            try:
                # 创建奖励组合
                combination = RewardCombination(
                    reward_functions=reward_functions,
                    weights=weights,
                    parameters={}
                )
                
                # 评估组合
                metrics = self.evaluate_combination(combination, historical_data)
                
                objective_value = self._calculate_objective_value(metrics)
                objective_history.append(objective_value)
                
                if objective_value > best_objective:
                    best_objective = objective_value
                    best_combination = combination
                
                if i % max(1, len(weight_grid) // 10) == 0:
                    self.logger.info(f"Progress: {i}/{len(weight_grid)}, Best: {best_objective:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Error evaluating combination {i}: {e}")
                continue
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 验证结果
        validation_results = {}
        if validation_data is not None and best_combination is not None:
            validation_results = self.evaluate_combination(best_combination, validation_data)
        
        return OptimizationResult(
            optimal_weights=best_combination.weights if best_combination else {},
            optimal_parameters=best_combination.parameters if best_combination else {},
            objective_value=best_objective,
            objective_history=objective_history,
            convergence_info={"method": "grid_search", "iterations": len(weight_grid)},
            performance_metrics=self.evaluate_combination(best_combination, historical_data) if best_combination else {},
            statistical_significance={},
            robustness_analysis={},
            execution_time=execution_time,
            validation_results=validation_results
        )
    
    def _generate_weight_grid(self, reward_names: List[str]) -> List[Dict[str, float]]:
        """生成权重网格"""
        n_rewards = len(reward_names)
        
        if n_rewards == 1:
            return [{reward_names[0]: 1.0}]
        
        # 简化网格生成 - 使用Dirichlet分布采样
        grid = []
        
        # 等权重基线
        equal_weight = 1.0 / n_rewards
        grid.append({name: equal_weight for name in reward_names})
        
        # 单一权重（专家系统）
        for name in reward_names:
            weights = {n: 0.0 for n in reward_names}
            weights[name] = 1.0
            grid.append(weights)
        
        # 随机权重组合
        np.random.seed(42)  # 确保可重复性
        for _ in range(self.grid_resolution - len(reward_names) - 1):
            # 使用Dirichlet分布生成权重
            alphas = np.ones(n_rewards)
            random_weights = np.random.dirichlet(alphas)
            
            weight_dict = {name: float(w) for name, w in zip(reward_names, random_weights)}
            grid.append(weight_dict)
        
        return grid
    
    def evaluate_combination(
        self,
        combination: RewardCombination,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """评估奖励组合"""
        
        if data.empty:
            return {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 1.0}
        
        # 模拟交易过程
        portfolio_values = [10000.0]  # 初始资金
        rewards = []
        
        for idx in range(len(data)):
            # 创建奖励上下文（简化）
            context = RewardContext(
                portfolio_value=portfolio_values[-1],
                action=0.0,  # 简化
                current_price=100.0,  # 简化
                step=idx
            )
            
            # 计算组合奖励
            try:
                reward = combination.calculate_combined_reward(context)
                rewards.append(reward)
                
                # 简化的组合价值更新
                new_value = portfolio_values[-1] * (1 + reward * 0.01)  # 假设1%的影响
                portfolio_values.append(new_value)
                
            except Exception as e:
                self.logger.warning(f"Error calculating reward at step {idx}: {e}")
                rewards.append(0.0)
                portfolio_values.append(portfolio_values[-1])
        
        # 计算性能指标
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if len(returns) == 0:
            return {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 1.0}
        
        # 夏普比率
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 总收益率
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # 最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # 波动率
        volatility = np.std(returns) * np.sqrt(252)
        
        # Calmar比率
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "calmar_ratio": calmar_ratio
        }
    
    def _calculate_objective_value(self, metrics: Dict[str, float]) -> float:
        """计算目标函数值"""
        if self.objective == OptimizationObjective.SHARPE_RATIO:
            return metrics.get("sharpe_ratio", 0.0)
        elif self.objective == OptimizationObjective.TOTAL_RETURN:
            return metrics.get("total_return", 0.0)
        elif self.objective == OptimizationObjective.MAX_DRAWDOWN:
            return -metrics.get("max_drawdown", 1.0)  # 负号因为要最小化回撤
        elif self.objective == OptimizationObjective.CALMAR_RATIO:
            return metrics.get("calmar_ratio", 0.0)
        elif self.objective == OptimizationObjective.MULTI_OBJECTIVE:
            # 多目标组合
            sharpe = metrics.get("sharpe_ratio", 0.0)
            return_val = metrics.get("total_return", 0.0)
            drawdown = metrics.get("max_drawdown", 1.0)
            
            # 加权组合
            return 0.5 * sharpe + 0.3 * return_val - 0.2 * drawdown
        else:
            return metrics.get("sharpe_ratio", 0.0)


class GeneticOptimizer(BaseRewardOptimizer):
    """遗传算法优化器"""
    
    def __init__(
        self,
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        super().__init__(OptimizationMethod.GENETIC, objective, constraints)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(
        self,
        reward_functions: Dict[str, BaseReward],
        historical_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> OptimizationResult:
        """执行遗传算法优化"""
        
        start_time = datetime.now()
        reward_names = list(reward_functions.keys())
        n_rewards = len(reward_names)
        
        # 使用scipy的differential_evolution
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_rewards
        
        def objective_function(weights):
            # 标准化权重
            weights = weights / np.sum(weights) * self.constraints.weight_sum
            weight_dict = {name: weight for name, weight in zip(reward_names, weights)}
            
            try:
                combination = RewardCombination(
                    reward_functions=reward_functions,
                    weights=weight_dict,
                    parameters={}
                )
                
                metrics = self.evaluate_combination(combination, historical_data)
                objective_value = self._calculate_objective_value(metrics)
                
                return -objective_value  # minimize (负号)
                
            except Exception as e:
                self.logger.warning(f"Error in objective function: {e}")
                return float('inf')
        
        # 执行优化
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.max_generations,
            popsize=self.population_size // n_rewards,  # scipy的popsize是相对的
            mutation=(0.5, 1.5),
            recombination=self.crossover_rate,
            seed=42
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 处理结果
        optimal_weights_array = result.x / np.sum(result.x) * self.constraints.weight_sum
        optimal_weights = {name: weight for name, weight in zip(reward_names, optimal_weights_array)}
        
        best_combination = RewardCombination(
            reward_functions=reward_functions,
            weights=optimal_weights,
            parameters={}
        )
        
        # 验证结果
        validation_results = {}
        if validation_data is not None:
            validation_results = self.evaluate_combination(best_combination, validation_data)
        
        return OptimizationResult(
            optimal_weights=optimal_weights,
            optimal_parameters={},
            objective_value=-result.fun,  # 恢复符号
            objective_history=[],
            convergence_info={
                "method": "genetic",
                "iterations": result.nit,
                "success": result.success,
                "message": result.message
            },
            performance_metrics=self.evaluate_combination(best_combination, historical_data),
            statistical_significance={},
            robustness_analysis={},
            execution_time=execution_time,
            validation_results=validation_results
        )
    
    def evaluate_combination(
        self,
        combination: RewardCombination,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """评估奖励组合（复用GridSearchOptimizer的实现）"""
        optimizer = GridSearchOptimizer(self.objective, self.constraints)
        return optimizer.evaluate_combination(combination, data)
    
    def _calculate_objective_value(self, metrics: Dict[str, float]) -> float:
        """计算目标函数值（复用GridSearchOptimizer的实现）"""
        optimizer = GridSearchOptimizer(self.objective, self.constraints)
        return optimizer._calculate_objective_value(metrics)


class RewardOptimizerFactory:
    """奖励优化器工厂"""
    
    @staticmethod
    def create_optimizer(
        method: OptimizationMethod,
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
        **kwargs
    ) -> BaseRewardOptimizer:
        """创建优化器"""
        
        if method == OptimizationMethod.GRID_SEARCH:
            return GridSearchOptimizer(
                objective=objective,
                constraints=constraints,
                grid_resolution=kwargs.get('grid_resolution', 10)
            )
        
        elif method == OptimizationMethod.GENETIC:
            return GeneticOptimizer(
                objective=objective,
                constraints=constraints,
                population_size=kwargs.get('population_size', 50),
                max_generations=kwargs.get('max_generations', 100),
                mutation_rate=kwargs.get('mutation_rate', 0.1),
                crossover_rate=kwargs.get('crossover_rate', 0.8)
            )
        
        else:
            raise ValueError(f"Unsupported optimization method: {method}")


# 便利函数
def optimize_reward_combination(
    reward_functions: Dict[str, BaseReward],
    historical_data: pd.DataFrame,
    method: str = "grid_search",
    objective: str = "sharpe_ratio",
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    validation_data: Optional[pd.DataFrame] = None,
    **kwargs
) -> OptimizationResult:
    """便利函数：优化奖励组合"""
    
    # 转换枚举
    method_enum = OptimizationMethod(method)
    objective_enum = OptimizationObjective(objective)
    
    # 创建约束
    constraints = OptimizationConstraints(
        max_weight=max_weight,
        min_weight=min_weight,
        weight_sum=1.0
    )
    
    # 创建优化器
    optimizer = RewardOptimizerFactory.create_optimizer(
        method=method_enum,
        objective=objective_enum,
        constraints=constraints,
        **kwargs
    )
    
    # 执行优化
    return optimizer.optimize(
        reward_functions=reward_functions,
        historical_data=historical_data,
        validation_data=validation_data,
        **kwargs
    )


def create_optimal_reward_combination(
    reward_functions: Dict[str, BaseReward],
    weights: Dict[str, float],
    parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    combination_method: str = "weighted_sum",
    normalization: str = "standard"
) -> RewardCombination:
    """便利函数：创建优化的奖励组合"""
    
    return RewardCombination(
        reward_functions=reward_functions,
        weights=weights,
        parameters=parameters or {},
        combination_method=combination_method,
        normalization=normalization
    )