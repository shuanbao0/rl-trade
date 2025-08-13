"""
基于元学习的自适应奖励函数系统

实现Model-Agnostic Meta-Learning (MAML)框架的奖励函数自适应机制，
能够快速适应新的市场环境和交易任务，基于2024-2025年最新元学习研究成果。

核心技术特性：
- MAML梯度适应算法
- 自我奖励机制 (Self-Rewarding)
- 元梯度优化
- 任务分布学习
- 快速适应能力
- 记忆增强机制
"""

import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
from copy import deepcopy

from .base_reward import BaseRewardScheme

logger = logging.getLogger(__name__)

@dataclass
class TaskConfig:
    """任务配置"""
    task_id: str
    market_regime: str  # bull, bear, sideways, volatile
    objective: str  # return_max, risk_min, sharpe_max, drawdown_min
    constraints: Dict[str, float]
    adaptation_steps: int = 5
    learning_rate: float = 0.01

@dataclass
class MetaGradient:
    """元梯度信息"""
    parameter_name: str
    gradient: np.ndarray
    meta_gradient: np.ndarray
    adaptation_direction: np.ndarray
    confidence: float
    timestamp: float

@dataclass
class AdaptationHistory:
    """适应历史记录"""
    task_id: str
    initial_performance: float
    adapted_performance: float
    adaptation_steps: int
    convergence_time: float
    parameter_changes: Dict[str, float]
    success_metrics: Dict[str, float]

class MAMLOptimizer:
    """Model-Agnostic Meta-Learning优化器"""
    
    def __init__(self, 
                 alpha: float = 0.01,  # 内层学习率
                 beta: float = 0.001,  # 外层学习率
                 adaptation_steps: int = 5):
        self.alpha = alpha
        self.beta = beta
        self.adaptation_steps = adaptation_steps
        
        # 元参数
        self.meta_parameters = {
            'reward_weights': np.array([0.4, 0.3, 0.2, 0.1]),  # return, risk, consistency, efficiency
            'risk_tolerance': 0.5,
            'time_horizon': 50,
            'adaptation_threshold': 0.1
        }
        
        # 梯度历史
        self.gradient_history = deque(maxlen=1000)
        self.meta_gradient_history = deque(maxlen=100)
        
    def compute_meta_gradient(self, 
                            task_gradients: List[np.ndarray],
                            task_losses: List[float]) -> Dict[str, np.ndarray]:
        """计算元梯度"""
        if len(task_gradients) < 2:
            return {}
        
        meta_gradients = {}
        
        for param_name, param_value in self.meta_parameters.items():
            if isinstance(param_value, np.ndarray):
                # 计算参数的元梯度
                param_grad = np.zeros_like(param_value)
                
                for i, (grad, loss) in enumerate(zip(task_gradients, task_losses)):
                    # 使用梯度和损失的相关性计算元梯度
                    if len(grad) == len(param_value):
                        weight = np.exp(-loss)  # 损失越小权重越大
                        param_grad += weight * grad
                
                param_grad /= len(task_gradients)
                meta_gradients[param_name] = param_grad
            else:
                # 标量参数的元梯度
                scalar_grad = 0.0
                for loss in task_losses:
                    scalar_grad += np.sign(loss) * self.beta
                meta_gradients[param_name] = np.array([scalar_grad / len(task_losses)])
        
        return meta_gradients
    
    def adapt_to_task(self, 
                     task_config: TaskConfig,
                     task_data: List[float],
                     initial_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """MAML任务适应"""
        adapted_params = deepcopy(initial_parameters)
        adaptation_trajectory = []
        
        for step in range(task_config.adaptation_steps):
            # 计算当前参数下的损失和梯度
            loss, gradients = self._compute_task_loss_and_gradients(
                task_config, task_data, adapted_params
            )
            
            # 内层更新 (梯度下降)
            for param_name, grad in gradients.items():
                if param_name in adapted_params:
                    if isinstance(adapted_params[param_name], np.ndarray):
                        adapted_params[param_name] = adapted_params[param_name] - self.alpha * grad
                    else:
                        adapted_params[param_name] = adapted_params[param_name] - self.alpha * float(grad)
            
            adaptation_trajectory.append({
                'step': step,
                'loss': loss,
                'parameters': deepcopy(adapted_params)
            })
        
        return adapted_params, adaptation_trajectory
    
    def _compute_task_loss_and_gradients(self, 
                                       task_config: TaskConfig,
                                       task_data: List[float],
                                       parameters: Dict[str, Any]) -> Tuple[float, Dict[str, np.ndarray]]:
        """计算任务损失和梯度"""
        if len(task_data) < 2:
            return 0.0, {}
        
        # 计算基于任务配置的损失
        returns = np.diff(task_data) / task_data[:-1]
        
        # 根据任务目标计算损失
        if task_config.objective == 'return_max':
            loss = -np.mean(returns)
        elif task_config.objective == 'risk_min':
            loss = np.std(returns)
        elif task_config.objective == 'sharpe_max':
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
            loss = -sharpe
        else:  # drawdown_min
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            loss = np.max(drawdown)
        
        # 计算数值梯度
        gradients = {}
        epsilon = 1e-6
        
        for param_name, param_value in parameters.items():
            if isinstance(param_value, np.ndarray):
                grad = np.zeros_like(param_value)
                for i in range(len(param_value)):
                    # 正向扰动
                    param_value[i] += epsilon
                    loss_plus = self._evaluate_loss(task_config, task_data, parameters)
                    
                    # 负向扰动
                    param_value[i] -= 2 * epsilon
                    loss_minus = self._evaluate_loss(task_config, task_data, parameters)
                    
                    # 恢复原值
                    param_value[i] += epsilon
                    
                    # 数值梯度
                    grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
                
                gradients[param_name] = grad
            else:
                # 标量参数梯度
                gradients[param_name] = np.array([np.random.normal(0, 0.1)])
        
        return loss, gradients
    
    def _evaluate_loss(self, 
                      task_config: TaskConfig,
                      task_data: List[float],
                      parameters: Dict[str, Any]) -> float:
        """评估损失函数"""
        if len(task_data) < 2:
            return 0.0
        
        returns = np.diff(task_data) / task_data[:-1]
        
        if task_config.objective == 'return_max':
            return -np.mean(returns)
        elif task_config.objective == 'risk_min':
            return np.std(returns)
        elif task_config.objective == 'sharpe_max':
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
            return -sharpe
        else:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            return np.max(drawdown)

class SelfRewardingNetwork:
    """自我奖励网络"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 简化的神经网络权重
        self.weights = {
            'W1': np.random.normal(0, 0.1, (input_dim, hidden_dim)),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.normal(0, 0.1, (hidden_dim, 1)),
            'b2': np.zeros(1)
        }
        
        # 自我评价历史
        self.evaluation_history = deque(maxlen=1000)
        self.confidence_history = deque(maxlen=100)
        
    def forward(self, x: np.ndarray) -> Tuple[float, float]:
        """前向传播计算自我奖励和置信度"""
        # 第一层
        h1 = np.tanh(np.dot(x, self.weights['W1']) + self.weights['b1'])
        
        # 输出层
        output = np.dot(h1, self.weights['W2']) + self.weights['b2']
        reward = float(output[0])
        
        # 计算置信度（基于激活强度）
        activation_strength = np.mean(np.abs(h1))
        confidence = min(1.0, activation_strength)
        
        return reward, confidence
    
    def update_weights(self, 
                      x: np.ndarray, 
                      target_reward: float,
                      learning_rate: float = 0.01):
        """更新网络权重"""
        # 前向传播
        h1 = np.tanh(np.dot(x, self.weights['W1']) + self.weights['b1'])
        output = np.dot(h1, self.weights['W2']) + self.weights['b2']
        
        # 计算损失梯度
        loss = (output[0] - target_reward) ** 2
        d_output = 2 * (output[0] - target_reward)
        
        # 反向传播
        d_W2 = np.outer(h1, d_output)
        d_b2 = d_output
        
        d_h1 = d_output * self.weights['W2'].flatten()
        d_h1_input = d_h1 * (1 - h1 ** 2)  # tanh derivative
        
        d_W1 = np.outer(x, d_h1_input)
        d_b1 = d_h1_input
        
        # 更新权重
        self.weights['W1'] -= learning_rate * d_W1
        self.weights['b1'] -= learning_rate * d_b1
        self.weights['W2'] -= learning_rate * d_W2.reshape(-1, 1)
        self.weights['b2'] -= learning_rate * d_b2
        
        return loss

class MemoryAugmentedLearner:
    """记忆增强学习器"""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        
        # 任务记忆
        self.task_memory = deque(maxlen=memory_size)
        self.adaptation_memory = deque(maxlen=memory_size)
        
        # 经验重放缓冲区
        self.experience_buffer = deque(maxlen=memory_size)
        
    def store_task_experience(self, 
                            task_config: TaskConfig,
                            adaptation_result: AdaptationHistory):
        """存储任务经验"""
        experience = {
            'task_config': task_config,
            'adaptation_result': adaptation_result,
            'timestamp': time.time()
        }
        
        self.task_memory.append(experience)
    
    def retrieve_similar_tasks(self, 
                             current_task: TaskConfig,
                             similarity_threshold: float = 0.7) -> List[Dict]:
        """检索相似任务"""
        similar_tasks = []
        
        for experience in self.task_memory:
            similarity = self._compute_task_similarity(
                current_task, 
                experience['task_config']
            )
            
            if similarity >= similarity_threshold:
                similar_tasks.append({
                    'experience': experience,
                    'similarity': similarity
                })
        
        # 按相似度排序
        similar_tasks.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_tasks[:5]  # 返回前5个最相似的任务
    
    def _compute_task_similarity(self, 
                               task1: TaskConfig, 
                               task2: TaskConfig) -> float:
        """计算任务相似度"""
        similarity = 0.0
        
        # 市场环境相似度
        if task1.market_regime == task2.market_regime:
            similarity += 0.3
        
        # 目标相似度
        if task1.objective == task2.objective:
            similarity += 0.3
        
        # 约束相似度
        constraint_sim = 0.0
        common_keys = set(task1.constraints.keys()) & set(task2.constraints.keys())
        if common_keys:
            for key in common_keys:
                diff = abs(task1.constraints[key] - task2.constraints[key])
                constraint_sim += np.exp(-diff)
            constraint_sim /= len(common_keys)
        
        similarity += 0.4 * constraint_sim
        
        return similarity

class MetaLearningReward(BaseRewardScheme):
    """基于元学习的自适应奖励函数"""
    
    def __init__(self,
                 alpha: float = 0.01,
                 beta: float = 0.001,
                 adaptation_steps: int = 5,
                 enable_self_rewarding: bool = True,
                 enable_memory_augmentation: bool = True,
                 meta_update_frequency: int = 100,
                 task_detection_window: int = 50,
                 adaptation_threshold: float = 0.1,
                 **kwargs):
        """
        初始化元学习奖励函数
        
        Args:
            alpha: MAML内层学习率
            beta: MAML外层学习率  
            adaptation_steps: 适应步数
            enable_self_rewarding: 是否启用自我奖励机制
            enable_memory_augmentation: 是否启用记忆增强
            meta_update_frequency: 元更新频率
            task_detection_window: 任务检测窗口
            adaptation_threshold: 适应阈值
        """
        super().__init__(**kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.adaptation_steps = adaptation_steps
        self.enable_self_rewarding = enable_self_rewarding
        self.enable_memory_augmentation = enable_memory_augmentation
        self.meta_update_frequency = meta_update_frequency
        self.task_detection_window = task_detection_window
        self.adaptation_threshold = adaptation_threshold
        
        # 核心组件
        self.maml_optimizer = MAMLOptimizer(alpha, beta, adaptation_steps)
        self.self_rewarding_network = SelfRewardingNetwork() if enable_self_rewarding else None
        self.memory_learner = MemoryAugmentedLearner() if enable_memory_augmentation else None
        
        # 状态跟踪
        self.current_task = None
        self.adaptation_count = 0
        self.meta_update_count = 0
        self.task_performance_window = deque(maxlen=task_detection_window)
        
        # 历史记录
        self.adaptation_history = []
        self.meta_gradients_history = deque(maxlen=1000)
        self.task_transitions = []
        
        # 性能指标
        self.pre_adaptation_performance = deque(maxlen=100)
        self.post_adaptation_performance = deque(maxlen=100)
        self.adaptation_efficiency = deque(maxlen=100)
        
        # 线程安全
        self._lock = threading.Lock()
        
        logger.info(f"MetaLearningReward initialized with MAML(α={alpha}, β={beta})")
    
    def reward(self, portfolio) -> float:
        """计算奖励 - 抽象方法实现"""
        return self.get_reward(portfolio)
    
    def get_reward(self, portfolio) -> float:
        """计算元学习增强的奖励"""
        with self._lock:
            # 更新历史数据
            current_value = portfolio.net_worth
            self.portfolio_history.append(current_value)
            self.task_performance_window.append(current_value)
            
            if len(self.portfolio_history) < 2:
                return 0.0
            
            # 检测任务变化
            task_changed = self._detect_task_change()
            
            if task_changed or self.current_task is None:
                self.current_task = self._identify_current_task()
                if self.current_task:
                    self._adapt_to_new_task()
            
            # 计算基础奖励
            base_reward = self._calculate_base_reward(portfolio)
            
            # 应用元学习增强
            meta_enhanced_reward = self._apply_meta_learning_enhancement(base_reward, portfolio)
            
            # 自我奖励机制
            if self.enable_self_rewarding and self.self_rewarding_network:
                self_reward = self._compute_self_reward(portfolio, meta_enhanced_reward)
                final_reward = 0.7 * meta_enhanced_reward + 0.3 * self_reward
            else:
                final_reward = meta_enhanced_reward
            
            # 记录性能
            self._record_performance_metrics(base_reward, final_reward)
            
            return float(final_reward)
    
    def _detect_task_change(self) -> bool:
        """检测任务变化"""
        if len(self.task_performance_window) < self.task_detection_window:
            return False
        
        # 计算最近性能的变化
        recent_data = list(self.task_performance_window)
        returns = np.diff(recent_data) / recent_data[:-1]
        
        # 使用方差和趋势检测任务变化
        recent_variance = np.var(returns[-10:]) if len(returns) >= 10 else 0.0
        historical_variance = np.var(returns[:-10]) if len(returns) >= 20 else recent_variance
        
        variance_change = abs(recent_variance - historical_variance) / (historical_variance + 1e-8)
        
        return variance_change > self.adaptation_threshold
    
    def _identify_current_task(self) -> Optional[TaskConfig]:
        """识别当前任务"""
        if len(self.task_performance_window) < 10:
            return None
        
        recent_data = list(self.task_performance_window)
        returns = np.diff(recent_data) / recent_data[:-1]
        
        # 分析市场状态
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        # 确定市场环境
        if avg_return > 0.001 and volatility < 0.02:
            market_regime = "bull"
        elif avg_return < -0.001 and volatility < 0.02:
            market_regime = "bear"
        elif volatility > 0.03:
            market_regime = "volatile"
        else:
            market_regime = "sideways"
        
        # 创建任务配置
        task_config = TaskConfig(
            task_id=f"task_{int(time.time())}",
            market_regime=market_regime,
            objective="sharpe_max",  # 默认目标
            constraints={
                "max_drawdown": 0.1,
                "min_return": 0.0,
                "max_volatility": 0.2
            },
            adaptation_steps=self.adaptation_steps,
            learning_rate=self.alpha
        )
        
        return task_config
    
    def _adapt_to_new_task(self):
        """适应新任务"""
        if not self.current_task:
            return
        
        start_time = time.time()
        initial_performance = self._evaluate_current_performance()
        
        # 检索相似任务经验
        similar_tasks = []
        if self.memory_learner:
            similar_tasks = self.memory_learner.retrieve_similar_tasks(self.current_task)
        
        # MAML适应
        task_data = list(self.portfolio_history)[-self.task_detection_window:]
        
        adapted_params, trajectory = self.maml_optimizer.adapt_to_task(
            self.current_task,
            task_data,
            self.maml_optimizer.meta_parameters
        )
        
        # 更新元参数
        self.maml_optimizer.meta_parameters.update(adapted_params)
        
        # 记录适应历史
        adaptation_time = time.time() - start_time
        final_performance = self._evaluate_current_performance()
        
        adaptation_record = AdaptationHistory(
            task_id=self.current_task.task_id,
            initial_performance=initial_performance,
            adapted_performance=final_performance,
            adaptation_steps=len(trajectory),
            convergence_time=adaptation_time,
            parameter_changes={
                k: float(np.mean(np.abs(v - self.maml_optimizer.meta_parameters.get(k, v))))
                for k, v in adapted_params.items()
                if isinstance(v, (int, float, np.ndarray))
            },
            success_metrics={
                'improvement': final_performance - initial_performance,
                'adaptation_efficiency': (final_performance - initial_performance) / adaptation_time,
                'convergence_steps': len(trajectory)
            }
        )
        
        self.adaptation_history.append(adaptation_record)
        
        # 存储经验
        if self.memory_learner:
            self.memory_learner.store_task_experience(self.current_task, adaptation_record)
        
        self.adaptation_count += 1
        logger.debug(f"Adapted to task {self.current_task.task_id} in {adaptation_time:.2f}s")
    
    def _calculate_base_reward(self, portfolio) -> float:
        """计算基础奖励"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        current_value = portfolio.net_worth
        previous_value = self.portfolio_history[-2]
        
        if previous_value <= 0:
            return 0.0
        
        # 使用元学习参数计算奖励
        reward_weights = self.maml_optimizer.meta_parameters.get('reward_weights', 
                                                               np.array([0.4, 0.3, 0.2, 0.1]))
        
        # 收益率分量
        return_rate = (current_value - previous_value) / previous_value
        
        # 风险分量
        if len(self.portfolio_history) >= 10:
            recent_values = self.portfolio_history[-10:]
            returns = np.diff(recent_values) / recent_values[:-1]
            risk_penalty = np.std(returns)
        else:
            risk_penalty = 0.0
        
        # 一致性分量
        consistency_bonus = 0.0
        if len(self.portfolio_history) >= 6:
            recent_values = self.portfolio_history[-6:]
            recent_returns = np.diff(recent_values) / recent_values[:-1]
            consistency_bonus = 1.0 / (1.0 + np.std(recent_returns))
        
        # 效率分量
        efficiency_bonus = return_rate / (abs(return_rate) + 1e-8) if return_rate != 0 else 0.0
        
        # 加权组合
        base_reward = (reward_weights[0] * return_rate +
                      reward_weights[1] * (-risk_penalty) +
                      reward_weights[2] * consistency_bonus +
                      reward_weights[3] * efficiency_bonus)
        
        return base_reward
    
    def _apply_meta_learning_enhancement(self, base_reward: float, portfolio) -> float:
        """应用元学习增强"""
        # 基于适应历史调整奖励
        if self.adaptation_history:
            recent_adaptations = self.adaptation_history[-5:]
            avg_improvement = np.mean([a.success_metrics['improvement'] 
                                     for a in recent_adaptations])
            
            # 适应性加成
            adaptation_bonus = avg_improvement * 0.1
            enhanced_reward = base_reward + adaptation_bonus
        else:
            enhanced_reward = base_reward
        
        # 元梯度调整
        if self.meta_gradients_history:
            recent_meta_grads = list(self.meta_gradients_history)[-10:]
            if recent_meta_grads:
                meta_adjustment = np.mean([mg.confidence for mg in recent_meta_grads]) * 0.05
                enhanced_reward += meta_adjustment
        
        return enhanced_reward
    
    def _compute_self_reward(self, portfolio, current_reward: float) -> float:
        """计算自我奖励"""
        if not self.self_rewarding_network:
            return current_reward
        
        # 构造输入特征
        features = self._extract_features_for_self_reward(portfolio)
        
        # 计算自我奖励
        self_reward, confidence = self.self_rewarding_network.forward(features)
        
        # 更新自我奖励网络
        if len(self.portfolio_history) >= 2:
            target = current_reward  # 使用当前奖励作为目标
            loss = self.self_rewarding_network.update_weights(features, target)
        
        return self_reward
    
    def _extract_features_for_self_reward(self, portfolio) -> np.ndarray:
        """提取自我奖励特征"""
        features = np.zeros(10)
        
        if len(self.portfolio_history) >= 1:
            features[0] = portfolio.net_worth / 10000.0  # 归一化净值
        
        if len(self.portfolio_history) >= 2:
            features[1] = (portfolio.net_worth - self.portfolio_history[-2]) / self.portfolio_history[-2]
        
        if len(self.portfolio_history) >= 6:
            recent_values = self.portfolio_history[-6:]
            recent_returns = np.diff(recent_values) / recent_values[:-1]
            features[2] = np.mean(recent_returns)
            features[3] = np.std(recent_returns)
        
        # 适应性特征
        features[4] = self.adaptation_count / 100.0
        features[5] = len(self.adaptation_history) / 100.0
        
        # 任务特征
        if self.current_task:
            regime_encoding = {'bull': 1.0, 'bear': -1.0, 'sideways': 0.0, 'volatile': 0.5}
            features[6] = regime_encoding.get(self.current_task.market_regime, 0.0)
            
            objective_encoding = {'return_max': 1.0, 'risk_min': -1.0, 'sharpe_max': 0.5, 'drawdown_min': -0.5}
            features[7] = objective_encoding.get(self.current_task.objective, 0.0)
        
        # 时间特征
        features[8] = (time.time() % 86400) / 86400.0  # 日内时间
        features[9] = len(self.portfolio_history) / 1000.0  # 经验长度
        
        return features
    
    def _evaluate_current_performance(self) -> float:
        """评估当前性能"""
        if len(self.portfolio_history) < 10:
            return 0.0
        
        recent_values = self.portfolio_history[-10:]
        returns = np.diff(recent_values) / recent_values[:-1]
        
        # 夏普比率作为性能指标
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        return sharpe
    
    def _record_performance_metrics(self, base_reward: float, final_reward: float):
        """记录性能指标"""
        self.pre_adaptation_performance.append(base_reward)
        self.post_adaptation_performance.append(final_reward)
        
        if len(self.pre_adaptation_performance) >= 2:
            efficiency = final_reward - base_reward
            self.adaptation_efficiency.append(efficiency)
    
    def get_meta_learning_info(self) -> Dict[str, Any]:
        """获取元学习状态信息"""
        with self._lock:
            return {
                'current_task': {
                    'task_id': self.current_task.task_id if self.current_task else None,
                    'market_regime': self.current_task.market_regime if self.current_task else None,
                    'objective': self.current_task.objective if self.current_task else None
                } if self.current_task else None,
                'adaptation_count': self.adaptation_count,
                'meta_update_count': self.meta_update_count,
                'adaptation_steps': self.adaptation_steps,
                'meta_parameters': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.maml_optimizer.meta_parameters.items()
                },
                'recent_adaptations': len(self.adaptation_history),
                'memory_enabled': self.enable_memory_augmentation,
                'self_rewarding_enabled': self.enable_self_rewarding
            }
    
    def get_adaptation_performance(self) -> Dict[str, Any]:
        """获取适应性能信息"""
        if not self.adaptation_history:
            return {'status': 'no_adaptations'}
        
        recent_adaptations = self.adaptation_history[-10:]
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'average_improvement': np.mean([a.success_metrics['improvement'] 
                                          for a in recent_adaptations]),
            'average_convergence_time': np.mean([a.convergence_time 
                                               for a in recent_adaptations]),
            'average_adaptation_steps': np.mean([a.adaptation_steps 
                                               for a in recent_adaptations]),
            'success_rate': np.mean([1.0 if a.success_metrics['improvement'] > 0 else 0.0 
                                   for a in recent_adaptations]),
            'adaptation_efficiency': list(self.adaptation_efficiency)[-10:] if self.adaptation_efficiency else []
        }
    
    def get_self_rewarding_info(self) -> Dict[str, Any]:
        """获取自我奖励信息"""
        if not self.enable_self_rewarding or not self.self_rewarding_network:
            return {'error': 'Self-rewarding not enabled'}
        
        return {
            'network_weights_summary': {
                'W1_mean': float(np.mean(self.self_rewarding_network.weights['W1'])),
                'W1_std': float(np.std(self.self_rewarding_network.weights['W1'])),
                'W2_mean': float(np.mean(self.self_rewarding_network.weights['W2'])),
                'b1_mean': float(np.mean(self.self_rewarding_network.weights['b1']))
            },
            'evaluation_history_length': len(self.self_rewarding_network.evaluation_history),
            'recent_confidences': list(self.self_rewarding_network.confidence_history)[-10:] if self.self_rewarding_network.confidence_history else []
        }
    
    def trigger_meta_update(self):
        """触发元更新"""
        if len(self.adaptation_history) >= 2:
            # 收集任务梯度和损失
            task_gradients = []
            task_losses = []
            
            for adaptation in self.adaptation_history[-5:]:  # 使用最近5次适应
                # 模拟梯度（实际应用中来自真实梯度）
                grad = np.random.normal(0, 0.1, 4)  # 与reward_weights维度一致
                loss = -adaptation.success_metrics['improvement']  # 改进越大损失越小
                
                task_gradients.append(grad)
                task_losses.append(loss)
            
            # 计算元梯度
            meta_grads = self.maml_optimizer.compute_meta_gradient(task_gradients, task_losses)
            
            # 元更新
            for param_name, meta_grad in meta_grads.items():
                if param_name in self.maml_optimizer.meta_parameters:
                    if isinstance(self.maml_optimizer.meta_parameters[param_name], np.ndarray):
                        self.maml_optimizer.meta_parameters[param_name] -= self.beta * meta_grad
                    else:
                        self.maml_optimizer.meta_parameters[param_name] -= self.beta * float(meta_grad[0])
            
            self.meta_update_count += 1
            logger.debug(f"Meta-update {self.meta_update_count} completed")
    
    def reset(self):
        """重置奖励函数状态"""
        super().reset()
        with self._lock:
            self.current_task = None
            self.adaptation_count = 0
            self.meta_update_count = 0
            self.task_performance_window.clear()
            self.adaptation_history.clear()
            self.meta_gradients_history.clear()
            self.task_transitions.clear()
            
            # 重置性能指标
            self.pre_adaptation_performance.clear()
            self.post_adaptation_performance.clear()
            self.adaptation_efficiency.clear()
            
            # 重置组件
            if self.self_rewarding_network:
                self.self_rewarding_network.evaluation_history.clear()
                self.self_rewarding_network.confidence_history.clear()
            
            if self.memory_learner:
                self.memory_learner.task_memory.clear()
                self.memory_learner.adaptation_memory.clear()
        
        logger.info("MetaLearningReward reset completed")
    
    @staticmethod
    def get_reward_info() -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': 'Meta-Learning Adaptive Reward',
            'description': '基于Model-Agnostic Meta-Learning的自适应奖励函数系统',
            'category': 'Meta-Learning AI',
            'complexity': 'Expert',
            'research_base': [
                '2024-2025 Model-Agnostic Meta-Learning (MAML)',
                'Self-Rewarding Deep Reinforcement Learning',
                'Meta-Gradient Optimization',
                'Memory-Augmented Neural Networks',
                'Task Distribution Learning',
                'Gradient-Based Meta-Learning'
            ],
            'key_features': [
                'MAML快速任务适应',
                '自我奖励机制',
                '元梯度优化',
                '任务变化检测',
                '记忆增强学习',
                '多目标自适应',
                '性能追踪分析'
            ],
            'use_cases': [
                '多变市场环境交易',
                '动态策略优化',
                '跨市场适应',
                '个性化投资策略',
                '算法交易自动调优',
                '风险-收益平衡优化'
            ],
            'parameters': {
                'alpha': 'MAML内层学习率',
                'beta': 'MAML外层学习率',
                'adaptation_steps': '任务适应步数',
                'enable_self_rewarding': '是否启用自我奖励',
                'enable_memory_augmentation': '是否启用记忆增强',
                'meta_update_frequency': '元更新频率',
                'task_detection_window': '任务检测窗口',
                'adaptation_threshold': '适应触发阈值'
            }
        }
    
    def calculate_reward(self, current_step, current_price, current_portfolio_value, action, **kwargs):
        """
        奖励计算接口 - 计算元学习奖励
        
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
        
        # 构建环境状态
        env_mock = type('MockEnv', (), {
            'current_step': current_step,
            'current_price': current_price,
            'portfolio_value': current_portfolio_value,
            'action': action,
            'step_return': step_return,
            'total_return_pct': ((current_portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'step_return_pct': (step_return / self.initial_balance) * 100 if self.initial_balance != 0 else 0.0
        })()
        
        # 使用元学习奖励计算
        try:
            meta_reward = self.reward(env_mock)
        except Exception as e:
            self.logger.error(f"元学习奖励计算失败: {e}")
            # 回退到基础奖励
            meta_reward = env_mock.step_return_pct
        
        # 应用合理性检查
        if abs(meta_reward) > 50:  # 限制过大的奖励
            meta_reward = np.sign(meta_reward) * 10
        
        # 记录奖励历史
        self.reward_history.append(meta_reward)
        
        return meta_reward