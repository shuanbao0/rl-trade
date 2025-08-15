"""
奖励组合优化器测试 - Reward Optimizer Tests

测试奖励组合优化器的各种功能。
"""

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.rewards.optimizers.reward_optimizer import (
    OptimizationMethod, OptimizationObjective, OptimizationConstraints,
    OptimizationResult, RewardCombination, BaseRewardOptimizer,
    GridSearchOptimizer, GeneticOptimizer, RewardOptimizerFactory,
    optimize_reward_combination, create_optimal_reward_combination
)
from src.rewards.core.reward_context import RewardContext
from src.rewards.core.base_reward import BaseReward


class MockReward(BaseReward):
    """测试用奖励函数"""
    
    def __init__(self, name: str, base_value: float = 0.0, volatility: float = 0.1):
        self.name = name
        self.base_value = base_value
        self.volatility = volatility
        self.history = []
    
    def calculate(self, context: RewardContext) -> float:
        # 简单的模拟奖励计算
        reward = self.base_value + np.random.normal(0, self.volatility)
        self.history.append(reward)
        return reward
    
    def reset(self):
        self.history = []
    
    def get_info(self):
        return {"name": self.name, "type": "mock_reward"}


class TestOptimizationConstraints(unittest.TestCase):
    """测试优化约束"""
    
    def test_constraints_creation(self):
        """测试约束创建"""
        constraints = OptimizationConstraints(
            max_weight=0.8,
            min_weight=0.1,
            weight_sum=1.0,
            max_complexity=5
        )
        
        self.assertEqual(constraints.max_weight, 0.8)
        self.assertEqual(constraints.min_weight, 0.1)
        self.assertEqual(constraints.weight_sum, 1.0)
        self.assertEqual(constraints.max_complexity, 5)
    
    def test_constraints_defaults(self):
        """测试约束默认值"""
        constraints = OptimizationConstraints()
        
        self.assertEqual(constraints.max_weight, 1.0)
        self.assertEqual(constraints.min_weight, 0.0)
        self.assertEqual(constraints.weight_sum, 1.0)
        self.assertIsNone(constraints.max_complexity)


class TestRewardCombination(unittest.TestCase):
    """测试奖励组合"""
    
    def setUp(self):
        """设置测试"""
        self.reward1 = MockReward("reward1", 0.1, 0.05)
        self.reward2 = MockReward("reward2", 0.05, 0.03)
        self.reward3 = MockReward("reward3", -0.02, 0.08)
        
        self.reward_functions = {
            "reward1": self.reward1,
            "reward2": self.reward2,
            "reward3": self.reward3
        }
        
        self.weights = {
            "reward1": 0.5,
            "reward2": 0.3,
            "reward3": 0.2
        }
    
    def test_combination_creation(self):
        """测试组合创建"""
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights=self.weights,
            parameters={},
            combination_method="weighted_sum"
        )
        
        self.assertEqual(len(combination.reward_functions), 3)
        self.assertEqual(combination.weights["reward1"], 0.5)
        self.assertEqual(combination.combination_method, "weighted_sum")
    
    def test_weighted_sum_calculation(self):
        """测试加权求和计算"""
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights=self.weights,
            parameters={},
            combination_method="weighted_sum"
        )
        
        context = RewardContext(
            portfolio_value=10000.0,
            action=0.1,
            current_price=100.0,
            step=1
        )
        
        # 设置固定随机种子以确保可重复性
        np.random.seed(42)
        reward = combination.calculate_combined_reward(context)
        
        self.assertIsInstance(reward, float)
        # 由于使用随机数，只检查合理范围
        self.assertTrue(-1.0 <= reward <= 1.0)
    
    def test_weighted_geometric_mean_calculation(self):
        """测试加权几何平均计算"""
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights=self.weights,
            parameters={},
            combination_method="weighted_geometric_mean"
        )
        
        context = RewardContext(
            portfolio_value=10000.0,
            action=0.1,
            current_price=100.0,
            step=1
        )
        
        np.random.seed(42)
        reward = combination.calculate_combined_reward(context)
        
        self.assertIsInstance(reward, float)
    
    def test_ensemble_combination_calculation(self):
        """测试集成组合计算"""
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights=self.weights,
            parameters={},
            combination_method="ensemble",
            meta_parameters={"ensemble_weights": [0.6, 0.3, 0.1]}
        )
        
        context = RewardContext(
            portfolio_value=10000.0,
            action=0.1,
            current_price=100.0,
            step=1
        )
        
        np.random.seed(42)
        reward = combination.calculate_combined_reward(context)
        
        self.assertIsInstance(reward, float)
    
    def test_adaptive_combination_calculation(self):
        """测试自适应组合计算"""
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights=self.weights,
            parameters={},
            combination_method="adaptive",
            meta_parameters={"adaptation_rate": 0.1, "volatility_factor": 1.5}
        )
        
        context = RewardContext(
            portfolio_value=10000.0,
            action=0.1,
            current_price=100.0,
            step=1
        )
        
        np.random.seed(42)
        reward = combination.calculate_combined_reward(context)
        
        self.assertIsInstance(reward, float)
    
    def test_invalid_combination_method(self):
        """测试无效组合方法"""
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights=self.weights,
            parameters={},
            combination_method="invalid_method"
        )
        
        context = RewardContext(
            portfolio_value=10000.0,
            action=0.1,
            current_price=100.0,
            step=1
        )
        
        with self.assertRaises(ValueError):
            combination.calculate_combined_reward(context)


class TestGridSearchOptimizer(unittest.TestCase):
    """测试网格搜索优化器"""
    
    def setUp(self):
        """设置测试"""
        self.reward_functions = {
            "reward1": MockReward("reward1", 0.1, 0.05),
            "reward2": MockReward("reward2", 0.05, 0.03),
            "reward3": MockReward("reward3", -0.02, 0.08)
        }
        
        # 创建测试数据
        np.random.seed(42)
        self.historical_data = pd.DataFrame({
            'price': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        self.constraints = OptimizationConstraints(
            max_weight=1.0,
            min_weight=0.0,
            weight_sum=1.0
        )
        
        self.optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.SHARPE_RATIO,
            constraints=self.constraints,
            grid_resolution=5
        )
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(self.optimizer.method, OptimizationMethod.GRID_SEARCH)
        self.assertEqual(self.optimizer.objective, OptimizationObjective.SHARPE_RATIO)
        self.assertEqual(self.optimizer.grid_resolution, 5)
    
    def test_weight_grid_generation(self):
        """测试权重网格生成"""
        reward_names = list(self.reward_functions.keys())
        grid = self.optimizer._generate_weight_grid(reward_names)
        
        self.assertIsInstance(grid, list)
        self.assertTrue(len(grid) > 0)
        
        # 检查权重网格的有效性
        for weights in grid:
            self.assertIsInstance(weights, dict)
            # 检查权重总和接近1.0
            weight_sum = sum(weights.values())
            self.assertAlmostEqual(weight_sum, 1.0, places=6)
    
    def test_constraints_validation(self):
        """测试约束验证"""
        valid_weights = {"reward1": 0.5, "reward2": 0.3, "reward3": 0.2}
        self.assertTrue(self.optimizer.validate_constraints(valid_weights))
        
        invalid_weights = {"reward1": 1.5, "reward2": 0.3, "reward3": 0.2}  # 超过最大权重
        self.assertFalse(self.optimizer.validate_constraints(invalid_weights))
        
        invalid_sum = {"reward1": 0.5, "reward2": 0.3, "reward3": 0.1}  # 总和不为1
        self.assertFalse(self.optimizer.validate_constraints(invalid_sum))
    
    def test_weights_normalization(self):
        """测试权重标准化"""
        weights = {"reward1": 2.0, "reward2": 3.0, "reward3": 1.0}
        normalized = self.optimizer.normalize_weights(weights)
        
        # 检查标准化后权重总和为1
        weight_sum = sum(normalized.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=6)
        
        # 检查比例保持
        self.assertAlmostEqual(normalized["reward2"] / normalized["reward1"], 3.0 / 2.0, places=6)
    
    def test_combination_evaluation(self):
        """测试组合评估"""
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights={"reward1": 0.5, "reward2": 0.3, "reward3": 0.2},
            parameters={}
        )
        
        metrics = self.optimizer.evaluate_combination(combination, self.historical_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("total_return", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("volatility", metrics)
        
        # 检查指标的合理范围（放宽范围以适应随机性）
        self.assertTrue(-50 <= metrics["sharpe_ratio"] <= 50)
        self.assertTrue(-5 <= metrics["total_return"] <= 5)
        self.assertTrue(0 <= metrics["max_drawdown"] <= 1)
    
    def test_optimization_execution(self):
        """测试优化执行"""
        np.random.seed(42)  # 确保可重复性
        
        result = self.optimizer.optimize(
            reward_functions=self.reward_functions,
            historical_data=self.historical_data
        )
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsInstance(result.optimal_weights, dict)
        self.assertIsInstance(result.objective_value, float)
        self.assertIsInstance(result.execution_time, float)
        
        # 检查权重有效性
        weight_sum = sum(result.optimal_weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=6)
        
        # 检查所有奖励都有权重
        for reward_name in self.reward_functions.keys():
            self.assertIn(reward_name, result.optimal_weights)
    
    def test_objective_value_calculation(self):
        """测试目标函数值计算"""
        metrics = {
            "sharpe_ratio": 1.5,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "calmar_ratio": 2.0
        }
        
        # 测试不同目标
        optimizer_sharpe = GridSearchOptimizer(
            OptimizationObjective.SHARPE_RATIO, self.constraints
        )
        self.assertEqual(optimizer_sharpe._calculate_objective_value(metrics), 1.5)
        
        optimizer_return = GridSearchOptimizer(
            OptimizationObjective.TOTAL_RETURN, self.constraints
        )
        self.assertEqual(optimizer_return._calculate_objective_value(metrics), 0.2)
        
        optimizer_drawdown = GridSearchOptimizer(
            OptimizationObjective.MAX_DRAWDOWN, self.constraints
        )
        self.assertEqual(optimizer_drawdown._calculate_objective_value(metrics), -0.1)
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        
        combination = RewardCombination(
            reward_functions=self.reward_functions,
            weights={"reward1": 0.5, "reward2": 0.3, "reward3": 0.2},
            parameters={}
        )
        
        metrics = self.optimizer.evaluate_combination(combination, empty_data)
        
        # 应该返回默认值
        self.assertEqual(metrics["sharpe_ratio"], 0.0)
        self.assertEqual(metrics["total_return"], 0.0)
        self.assertEqual(metrics["max_drawdown"], 1.0)


class TestGeneticOptimizer(unittest.TestCase):
    """测试遗传算法优化器"""
    
    def setUp(self):
        """设置测试"""
        self.reward_functions = {
            "reward1": MockReward("reward1", 0.1, 0.05),
            "reward2": MockReward("reward2", 0.05, 0.03)
        }
        
        np.random.seed(42)
        self.historical_data = pd.DataFrame({
            'price': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        self.constraints = OptimizationConstraints()
        
        self.optimizer = GeneticOptimizer(
            objective=OptimizationObjective.SHARPE_RATIO,
            constraints=self.constraints,
            population_size=10,
            max_generations=5  # 减少测试时间
        )
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(self.optimizer.method, OptimizationMethod.GENETIC)
        self.assertEqual(self.optimizer.population_size, 10)
        self.assertEqual(self.optimizer.max_generations, 5)
    
    def test_optimization_execution(self):
        """测试优化执行"""
        np.random.seed(42)
        
        result = self.optimizer.optimize(
            reward_functions=self.reward_functions,
            historical_data=self.historical_data
        )
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsInstance(result.optimal_weights, dict)
        self.assertIsInstance(result.objective_value, float)
        
        # 检查权重有效性
        weight_sum = sum(result.optimal_weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=6)
        
        # 检查收敛信息
        self.assertIn("method", result.convergence_info)
        self.assertEqual(result.convergence_info["method"], "genetic")


class TestRewardOptimizerFactory(unittest.TestCase):
    """测试奖励优化器工厂"""
    
    def setUp(self):
        """设置测试"""
        self.constraints = OptimizationConstraints()
    
    def test_create_grid_search_optimizer(self):
        """测试创建网格搜索优化器"""
        optimizer = RewardOptimizerFactory.create_optimizer(
            method=OptimizationMethod.GRID_SEARCH,
            objective=OptimizationObjective.SHARPE_RATIO,
            constraints=self.constraints,
            grid_resolution=15
        )
        
        self.assertIsInstance(optimizer, GridSearchOptimizer)
        self.assertEqual(optimizer.grid_resolution, 15)
    
    def test_create_genetic_optimizer(self):
        """测试创建遗传算法优化器"""
        optimizer = RewardOptimizerFactory.create_optimizer(
            method=OptimizationMethod.GENETIC,
            objective=OptimizationObjective.TOTAL_RETURN,
            constraints=self.constraints,
            population_size=30,
            max_generations=50
        )
        
        self.assertIsInstance(optimizer, GeneticOptimizer)
        self.assertEqual(optimizer.population_size, 30)
        self.assertEqual(optimizer.max_generations, 50)
    
    def test_unsupported_method(self):
        """测试不支持的方法"""
        with self.assertRaises(ValueError):
            RewardOptimizerFactory.create_optimizer(
                method=OptimizationMethod.BAYESIAN,  # 未实现
                objective=OptimizationObjective.SHARPE_RATIO,
                constraints=self.constraints
            )


class TestConvenienceFunctions(unittest.TestCase):
    """测试便利函数"""
    
    def setUp(self):
        """设置测试"""
        self.reward_functions = {
            "reward1": MockReward("reward1", 0.1, 0.05),
            "reward2": MockReward("reward2", 0.05, 0.03)
        }
        
        np.random.seed(42)
        self.historical_data = pd.DataFrame({
            'price': np.random.randn(30).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 30)
        })
    
    def test_optimize_reward_combination_function(self):
        """测试优化奖励组合便利函数"""
        result = optimize_reward_combination(
            reward_functions=self.reward_functions,
            historical_data=self.historical_data,
            method="grid_search",
            objective="sharpe_ratio",
            max_weight=0.8,
            grid_resolution=5
        )
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsInstance(result.optimal_weights, dict)
    
    def test_create_optimal_reward_combination_function(self):
        """测试创建优化奖励组合便利函数"""
        weights = {"reward1": 0.6, "reward2": 0.4}
        
        combination = create_optimal_reward_combination(
            reward_functions=self.reward_functions,
            weights=weights,
            combination_method="weighted_sum"
        )
        
        self.assertIsInstance(combination, RewardCombination)
        self.assertEqual(combination.weights, weights)
        self.assertEqual(combination.combination_method, "weighted_sum")


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_single_reward_optimization(self):
        """测试单一奖励优化"""
        single_reward = {"reward1": MockReward("reward1", 0.1, 0.05)}
        
        np.random.seed(42)
        data = pd.DataFrame({
            'price': np.random.randn(20).cumsum() + 100
        })
        
        result = optimize_reward_combination(
            reward_functions=single_reward,
            historical_data=data,
            method="grid_search",
            grid_resolution=3
        )
        
        self.assertEqual(result.optimal_weights["reward1"], 1.0)
    
    def test_zero_volatility_rewards(self):
        """测试零波动奖励"""
        zero_vol_rewards = {
            "reward1": MockReward("reward1", 0.1, 0.0),
            "reward2": MockReward("reward2", 0.05, 0.0)
        }
        
        np.random.seed(42)
        data = pd.DataFrame({
            'price': [100] * 20  # 无变化的价格
        })
        
        result = optimize_reward_combination(
            reward_functions=zero_vol_rewards,
            historical_data=data,
            method="grid_search",
            grid_resolution=3
        )
        
        self.assertIsInstance(result, OptimizationResult)
    
    def test_negative_reward_handling(self):
        """测试负奖励处理"""
        negative_rewards = {
            "reward1": MockReward("reward1", -0.2, 0.1),
            "reward2": MockReward("reward2", -0.1, 0.05)
        }
        
        np.random.seed(42)
        data = pd.DataFrame({
            'price': np.random.randn(15).cumsum() + 100
        })
        
        result = optimize_reward_combination(
            reward_functions=negative_rewards,
            historical_data=data,
            method="grid_search",
            grid_resolution=3
        )
        
        self.assertIsInstance(result, OptimizationResult)
    
    def test_extreme_constraints(self):
        """测试极端约束"""
        rewards = {
            "reward1": MockReward("reward1", 0.1, 0.05),
            "reward2": MockReward("reward2", 0.05, 0.03)
        }
        
        # 极端约束：最小权重过高（创建无法满足的约束）
        constraints = OptimizationConstraints(min_weight=0.8, max_weight=1.0)
        optimizer = GridSearchOptimizer(
            OptimizationObjective.SHARPE_RATIO, constraints
        )
        
        np.random.seed(42)
        data = pd.DataFrame({
            'price': np.random.randn(10).cumsum() + 100
        })
        
        # 极端约束下权重验证应该失败
        invalid_weights = {"reward1": 0.5, "reward2": 0.5}  # 两个都小于最小权重0.8
        self.assertFalse(optimizer.validate_constraints(invalid_weights))


if __name__ == '__main__':
    unittest.main()