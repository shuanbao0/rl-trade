#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPO人类对齐算法模块

实现基于人类反馈的PPO（Proximal Policy Optimization）算法，包括：
- PPO与人类对齐的集成训练
- 在线适应性学习机制
- 宪法AI安全框架
- 人类价值对齐优化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
from abc import ABC, abstractmethod
import warnings
from copy import deepcopy

logger = logging.getLogger(__name__)


class AlignmentObjective(Enum):
    """对齐目标枚举"""
    HUMAN_PREFERENCE = "human_preference"
    CONSTITUTIONAL_AI = "constitutional_ai"
    VALUE_ALIGNMENT = "value_alignment"
    SAFETY_FIRST = "safety_first"


@dataclass
class PPOHumanConfig:
    """PPO人类对齐配置"""
    # PPO基础参数
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    
    # 人类对齐特定参数
    alignment_objective: AlignmentObjective = AlignmentObjective.HUMAN_PREFERENCE
    human_feedback_weight: float = 1.0
    preference_learning_rate: float = 1e-4
    constitutional_weight: float = 0.5
    safety_constraint_threshold: float = 0.1
    
    # 训练参数
    num_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048
    target_kl: float = 0.01
    
    # 在线学习参数
    adaptation_frequency: int = 100
    feedback_collection_interval: int = 50
    human_evaluation_threshold: float = 0.8
    
    # 宪法AI参数
    constitutional_principles: List[str] = field(default_factory=lambda: [
        "maximize_trading_profit",
        "minimize_risk_exposure", 
        "ensure_market_fairness",
        "avoid_market_manipulation",
        "maintain_transparency"
    ])
    
    # 价值对齐参数
    value_alignment_weight: float = 0.3
    ethical_constraint_weight: float = 0.2


@dataclass
class PPOBatch:
    """PPO训练批次数据"""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    human_feedback: Optional[torch.Tensor] = None
    constitutional_scores: Optional[torch.Tensor] = None


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # 共享特征提取层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 策略头（连续动作空间）
        self.policy_mean = nn.Linear(prev_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 价值头
        self.value_head = nn.Linear(prev_dim, 1)
        
        # 人类偏好预测头
        self.preference_head = nn.Linear(prev_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Returns:
            Tuple[action_distribution, value, preference_score]
        """
        features = self.feature_extractor(state)
        
        # 策略分布
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std.clamp(-20, 2))
        action_dist = Normal(action_mean, action_std)
        
        # 状态价值
        value = self.value_head(features)
        
        # 人类偏好预测
        preference_score = torch.sigmoid(self.preference_head(features))
        
        return action_dist, value, preference_score
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        action_dist, value, preference_score = self.forward(state)
        
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作"""
        action_dist, value, preference_score = self.forward(state)
        
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy, preference_score


class ConstitutionalAI:
    """宪法AI框架
    
    实现基于原则的约束和安全机制
    """
    
    def __init__(self, config: PPOHumanConfig):
        self.config = config
        self.principles = config.constitutional_principles
        self.violation_history = defaultdict(list)
        
        # 原则权重（可学习）
        self.principle_weights = {
            principle: 1.0 for principle in self.principles
        }
        
    def evaluate_action(self, state: np.ndarray, action: np.ndarray, 
                       market_context: Dict[str, Any]) -> Dict[str, float]:
        """评估动作的宪法合规性"""
        scores = {}
        
        # 交易利润最大化检查
        if "maximize_trading_profit" in self.principles:
            profit_potential = self._evaluate_profit_potential(state, action, market_context)
            scores["maximize_trading_profit"] = profit_potential
        
        # 风险敞口最小化检查
        if "minimize_risk_exposure" in self.principles:
            risk_score = self._evaluate_risk_exposure(state, action, market_context)
            scores["minimize_risk_exposure"] = 1.0 - risk_score  # 反转，低风险得高分
        
        # 市场公平性检查
        if "ensure_market_fairness" in self.principles:
            fairness_score = self._evaluate_market_fairness(state, action, market_context)
            scores["ensure_market_fairness"] = fairness_score
        
        # 避免市场操纵检查
        if "avoid_market_manipulation" in self.principles:
            manipulation_score = self._evaluate_manipulation_risk(state, action, market_context)
            scores["avoid_market_manipulation"] = 1.0 - manipulation_score
        
        # 透明度维护检查
        if "maintain_transparency" in self.principles:
            transparency_score = self._evaluate_transparency(state, action, market_context)
            scores["maintain_transparency"] = transparency_score
        
        return scores
    
    def _evaluate_profit_potential(self, state: np.ndarray, action: np.ndarray, 
                                 market_context: Dict[str, Any]) -> float:
        """评估利润潜力"""
        # 基于动作大小和市场趋势
        position_size = abs(action[0]) if len(action) > 0 else 0
        market_trend = market_context.get('trend', 0)
        
        # 简化的利润潜力评估
        if position_size > 0:
            # 与趋势一致的大仓位获得更高评分
            profit_score = min(1.0, position_size * abs(market_trend) * 2)
        else:
            profit_score = 0.1  # 不交易获得较低评分
        
        return profit_score
    
    def _evaluate_risk_exposure(self, state: np.ndarray, action: np.ndarray, 
                              market_context: Dict[str, Any]) -> float:
        """评估风险敞口"""
        position_size = abs(action[0]) if len(action) > 0 else 0
        volatility = market_context.get('volatility', 0.5)
        
        # 风险 = 仓位大小 × 波动率
        risk_score = position_size * volatility
        return min(1.0, risk_score)
    
    def _evaluate_market_fairness(self, state: np.ndarray, action: np.ndarray, 
                                market_context: Dict[str, Any]) -> float:
        """评估市场公平性"""
        # 检查是否存在不公平优势利用
        # 简化实现：避免过度频繁交易
        trading_frequency = market_context.get('recent_trades', 0)
        
        if trading_frequency > 10:  # 过度频繁
            return 0.3
        elif trading_frequency > 5:  # 较频繁
            return 0.7
        else:
            return 1.0
    
    def _evaluate_manipulation_risk(self, state: np.ndarray, action: np.ndarray, 
                                  market_context: Dict[str, Any]) -> float:
        """评估市场操纵风险"""
        position_size = abs(action[0]) if len(action) > 0 else 0
        market_volume = market_context.get('volume', 1000000)
        
        # 如果交易量相对于市场容量过大，可能构成操纵
        volume_ratio = position_size / max(market_volume, 1)
        
        if volume_ratio > 0.1:  # 超过市场容量10%
            return 0.8
        elif volume_ratio > 0.05:  # 超过5%
            return 0.3
        else:
            return 0.0
    
    def _evaluate_transparency(self, state: np.ndarray, action: np.ndarray, 
                             market_context: Dict[str, Any]) -> float:
        """评估透明度"""
        # 简化实现：所有算法交易都认为是透明的
        return 1.0
    
    def compute_constitutional_score(self, scores: Dict[str, float]) -> float:
        """计算综合宪法得分"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for principle, score in scores.items():
            weight = self.principle_weights.get(principle, 1.0)
            weighted_sum += weight * score
            total_weight += weight
        
        return weighted_sum / max(total_weight, 1e-8)
    
    def update_principle_weights(self, feedback: Dict[str, float]):
        """根据人类反馈更新原则权重"""
        learning_rate = 0.01
        
        for principle, fb_score in feedback.items():
            if principle in self.principle_weights:
                current_weight = self.principle_weights[principle]
                # 简单的梯度更新
                self.principle_weights[principle] = max(0.1, 
                    current_weight + learning_rate * (fb_score - 0.5))


class PPOWithHumanAlignment:
    """带人类对齐的PPO算法"""
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOHumanConfig):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # 宪法AI
        self.constitutional_ai = ConstitutionalAI(config)
        
        # 经验缓冲区
        self.buffer = deque(maxlen=config.buffer_size)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'human_alignment_scores': [],
            'constitutional_scores': [],
            'kl_divergences': []
        }
        
        # 人类反馈历史
        self.human_feedback_history = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor, deterministic)
            _, value, preference_score = self.policy.forward(state_tensor)
        
        action_np = action.squeeze(0).numpy()
        
        # 计算宪法得分（需要市场上下文）
        market_context = self._extract_market_context(state)
        constitutional_scores = self.constitutional_ai.evaluate_action(
            state, action_np, market_context
        )
        constitutional_score = self.constitutional_ai.compute_constitutional_score(constitutional_scores)
        
        action_info = {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'preference_score': preference_score.item(),
            'constitutional_score': constitutional_score,
            'constitutional_details': constitutional_scores
        }
        
        return action_np, action_info
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool, action_info: Dict[str, Any]):
        """存储转换"""
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': action_info['log_prob'],
            'value': action_info['value'],
            'preference_score': action_info['preference_score'],
            'constitutional_score': action_info['constitutional_score']
        }
        
        self.buffer.append(transition)
    
    def update_policy(self, human_feedback: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """更新策略"""
        if len(self.buffer) < self.config.batch_size:
            return {}
        
        # 准备训练数据
        batch = self._prepare_batch()
        
        # 计算人类对齐奖励
        if human_feedback:
            human_rewards = self._compute_human_alignment_rewards(batch, human_feedback)
            batch.rewards = batch.rewards + self.config.human_feedback_weight * human_rewards
        
        # PPO更新
        update_stats = self._ppo_update(batch)
        
        # 更新训练统计
        self._update_training_stats(update_stats)
        
        return update_stats
    
    def _prepare_batch(self) -> PPOBatch:
        """准备训练批次"""
        # 从缓冲区采样
        batch_size = min(self.config.batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        preference_scores = []
        constitutional_scores = []
        
        for idx in indices:
            transition = self.buffer[idx]
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            values.append(transition['value'])
            log_probs.append(transition['log_prob'])
            preference_scores.append(transition['preference_score'])
            constitutional_scores.append(transition['constitutional_score'])
        
        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(np.array(rewards))
        values_tensor = torch.FloatTensor(np.array(values))
        log_probs_tensor = torch.FloatTensor(np.array(log_probs))
        
        # 计算优势和回报
        advantages, returns = self._compute_gae(rewards_tensor, values_tensor)
        
        return PPOBatch(
            observations=states_tensor,
            actions=actions_tensor,
            rewards=rewards_tensor,
            values=values_tensor,
            log_probs=log_probs_tensor,
            advantages=advantages,
            returns=returns,
            constitutional_scores=torch.FloatTensor(np.array(constitutional_scores))
        )
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计(GAE)"""
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # 简化的GAE计算（假设所有episode都结束）
        gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0  # 假设episode结束
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _compute_human_alignment_rewards(self, batch: PPOBatch, 
                                       human_feedback: List[Dict[str, Any]]) -> torch.Tensor:
        """计算人类对齐奖励"""
        human_rewards = torch.zeros(batch.rewards.size(0))
        
        # 根据人类反馈调整奖励
        for feedback in human_feedback:
            feedback_type = feedback.get('type', 'preference')
            
            if feedback_type == 'preference':
                # 偏好反馈
                preferred_action = feedback.get('preferred_action')
                confidence = feedback.get('confidence', 1.0)
                
                # 简化实现：基于动作相似性给予奖励
                for i in range(batch.actions.size(0)):
                    action_similarity = self._compute_action_similarity(
                        batch.actions[i].numpy(), preferred_action
                    )
                    human_rewards[i] += confidence * action_similarity
            
            elif feedback_type == 'scalar':
                # 标量评分反馈
                action_id = feedback.get('action_id')
                rating = feedback.get('rating', 0.0)
                
                # 基于评分调整对应动作的奖励
                if action_id is not None and action_id < len(human_rewards):
                    human_rewards[action_id] += rating
        
        return human_rewards
    
    def _ppo_update(self, batch: PPOBatch) -> Dict[str, float]:
        """PPO算法更新"""
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'constitutional_loss': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0
        }
        
        for epoch in range(self.config.num_epochs):
            # 随机打乱批次
            indices = torch.randperm(batch.observations.size(0))
            
            for start_idx in range(0, batch.observations.size(0), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, batch.observations.size(0))
                mini_batch_indices = indices[start_idx:end_idx]
                
                # 提取小批次
                mini_obs = batch.observations[mini_batch_indices]
                mini_actions = batch.actions[mini_batch_indices]
                mini_log_probs_old = batch.log_probs[mini_batch_indices]
                mini_advantages = batch.advantages[mini_batch_indices]
                mini_returns = batch.returns[mini_batch_indices]
                mini_constitutional = batch.constitutional_scores[mini_batch_indices]
                
                # 前向传播
                log_probs_new, values_new, entropy, preference_scores = self.policy.evaluate_actions(
                    mini_obs, mini_actions
                )
                
                # 策略损失（PPO clipped objective）
                ratio = torch.exp(log_probs_new - mini_log_probs_old)
                surr1 = ratio * mini_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 
                                   1.0 + self.config.clip_epsilon) * mini_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values_new.squeeze(), mini_returns)
                
                # 熵损失（鼓励探索）
                entropy_loss = -entropy.mean()
                
                # 宪法对齐损失
                constitutional_loss = F.mse_loss(
                    preference_scores.squeeze(), 
                    mini_constitutional
                )
                
                # 总损失
                total_loss = (
                    policy_loss + 
                    self.config.value_loss_coefficient * value_loss +
                    self.config.entropy_coefficient * entropy_loss +
                    self.config.constitutional_weight * constitutional_loss
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                # 更新统计
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy_loss'] += entropy_loss.item()
                stats['constitutional_loss'] += constitutional_loss.item()
                stats['total_loss'] += total_loss.item()
                
                # 计算KL散度（用于早停）
                kl_div = (mini_log_probs_old - log_probs_new).mean()
                stats['kl_divergence'] += kl_div.item()
                
                # 早停检查
                if abs(kl_div.item()) > self.config.target_kl:
                    logger.info(f"Early stopping due to KL divergence: {kl_div.item()}")
                    break
        
        # 平均化统计
        num_updates = self.config.num_epochs * max(1, batch.observations.size(0) // self.config.batch_size)
        for key in stats:
            stats[key] /= num_updates
        
        return stats
    
    def _extract_market_context(self, state: np.ndarray) -> Dict[str, Any]:
        """从状态中提取市场上下文"""
        # 简化实现，从状态向量中提取关键信息
        context = {
            'trend': 0.0,
            'volatility': 0.5,
            'volume': 1000000,
            'recent_trades': 0
        }
        
        if len(state) > 0:
            # 假设状态向量的某些维度对应市场信息
            context['trend'] = float(state[0]) if len(state) > 0 else 0.0
            context['volatility'] = abs(float(state[1])) if len(state) > 1 else 0.5
            context['volume'] = abs(float(state[2])) * 1000000 if len(state) > 2 else 1000000
        
        return context
    
    def _compute_action_similarity(self, action1: np.ndarray, action2: np.ndarray) -> float:
        """计算动作相似性"""
        if action2 is None:
            return 0.0
        
        action2 = np.array(action2)
        if action1.shape != action2.shape:
            return 0.0
        
        # 使用余弦相似性
        norm1 = np.linalg.norm(action1)
        norm2 = np.linalg.norm(action2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0
        
        similarity = np.dot(action1, action2) / (norm1 * norm2)
        return max(0.0, similarity)  # 只考虑正相似性
    
    def _update_training_stats(self, update_stats: Dict[str, float]):
        """更新训练统计"""
        for key, value in update_stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
    
    def collect_human_feedback(self, states: List[np.ndarray], actions: List[np.ndarray], 
                             rewards: List[float]) -> List[Dict[str, Any]]:
        """收集人类反馈（模拟）"""
        feedback_list = []
        
        for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
            # 模拟人类反馈收集
            market_context = self._extract_market_context(state)
            constitutional_scores = self.constitutional_ai.evaluate_action(
                state, action, market_context
            )
            
            # 基于宪法得分和实际奖励生成模拟反馈
            constitutional_score = self.constitutional_ai.compute_constitutional_score(constitutional_scores)
            
            # 模拟偏好反馈
            if np.random.random() < 0.3:  # 30%概率收到反馈
                feedback = {
                    'type': 'preference',
                    'state': state,
                    'preferred_action': action if reward > 0 else -action,
                    'confidence': min(1.0, abs(reward) + constitutional_score),
                    'expert_id': f"simulated_expert_{np.random.randint(3)}",
                    'timestamp': datetime.now()
                }
                feedback_list.append(feedback)
        
        return feedback_list
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'constitutional_weights': self.constitutional_ai.principle_weights
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.constitutional_ai.principle_weights = checkpoint.get('constitutional_weights', {})
        self.training_stats = checkpoint.get('training_stats', {})


class OnlineAdaptiveLearning:
    """在线适应性学习管理器"""
    
    def __init__(self, ppo_agent: PPOWithHumanAlignment, config: PPOHumanConfig):
        self.ppo_agent = ppo_agent
        self.config = config
        
        # 适应性学习状态
        self.adaptation_step = 0
        self.performance_history = deque(maxlen=100)
        self.feedback_buffer = deque(maxlen=config.buffer_size)
        
        # 性能监控
        self.performance_metrics = {
            'average_reward': 0.0,
            'human_satisfaction': 0.0,
            'constitutional_compliance': 0.0,
            'policy_stability': 0.0
        }
        
    def step(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool, action_info: Dict[str, Any]):
        """在线学习步骤"""
        
        # 存储转换
        self.ppo_agent.store_transition(state, action, reward, next_state, done, action_info)
        
        # 收集性能数据
        self.performance_history.append({
            'reward': reward,
            'constitutional_score': action_info['constitutional_score'],
            'preference_score': action_info['preference_score'],
            'timestamp': datetime.now()
        })
        
        self.adaptation_step += 1
        
        # 定期适应性更新
        if self.adaptation_step % self.config.adaptation_frequency == 0:
            self._perform_adaptation()
        
        # 定期收集人类反馈
        if self.adaptation_step % self.config.feedback_collection_interval == 0:
            self._collect_and_process_feedback()
    
    def _perform_adaptation(self):
        """执行适应性更新"""
        logger.info(f"Performing adaptive learning update at step {self.adaptation_step}")
        
        # 分析最近表现
        recent_performance = self._analyze_recent_performance()
        
        # 调整学习参数
        self._adjust_learning_parameters(recent_performance)
        
        # 执行策略更新
        if len(self.feedback_buffer) > 0:
            feedback_list = list(self.feedback_buffer)
            update_stats = self.ppo_agent.update_policy(feedback_list)
            logger.info(f"Policy update stats: {update_stats}")
            
            # 清空反馈缓冲区
            self.feedback_buffer.clear()
    
    def _analyze_recent_performance(self) -> Dict[str, float]:
        """分析最近表现"""
        if len(self.performance_history) == 0:
            return {}
        
        recent_data = list(self.performance_history)
        
        metrics = {
            'average_reward': np.mean([d['reward'] for d in recent_data]),
            'reward_volatility': np.std([d['reward'] for d in recent_data]),
            'constitutional_compliance': np.mean([d['constitutional_score'] for d in recent_data]),
            'preference_alignment': np.mean([d['preference_score'] for d in recent_data])
        }
        
        return metrics
    
    def _adjust_learning_parameters(self, performance: Dict[str, float]):
        """调整学习参数"""
        # 根据表现动态调整学习率
        avg_reward = performance.get('average_reward', 0.0)
        
        if avg_reward > 0.1:  # 表现良好，降低学习率稳定策略
            self.ppo_agent.config.learning_rate *= 0.95
        elif avg_reward < -0.1:  # 表现较差，提高学习率加速学习
            self.ppo_agent.config.learning_rate *= 1.05
        
        # 限制学习率范围
        self.ppo_agent.config.learning_rate = np.clip(
            self.ppo_agent.config.learning_rate, 1e-5, 1e-2
        )
        
        # 更新优化器学习率
        for param_group in self.ppo_agent.optimizer.param_groups:
            param_group['lr'] = self.ppo_agent.config.learning_rate
        
        logger.info(f"Adjusted learning rate to: {self.ppo_agent.config.learning_rate:.6f}")
    
    def _collect_and_process_feedback(self):
        """收集和处理人类反馈"""
        # 从最近的经验中收集状态、动作、奖励
        recent_transitions = list(self.ppo_agent.buffer)[-self.config.feedback_collection_interval:]
        
        if len(recent_transitions) == 0:
            return
        
        states = [t['state'] for t in recent_transitions]
        actions = [t['action'] for t in recent_transitions]
        rewards = [t['reward'] for t in recent_transitions]
        
        # 收集模拟人类反馈
        feedback_list = self.ppo_agent.collect_human_feedback(states, actions, rewards)
        
        # 添加到反馈缓冲区
        self.feedback_buffer.extend(feedback_list)
        
        logger.info(f"Collected {len(feedback_list)} feedback entries")
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """获取适应状态"""
        recent_performance = self._analyze_recent_performance()
        
        return {
            'adaptation_step': self.adaptation_step,
            'performance_metrics': recent_performance,
            'feedback_buffer_size': len(self.feedback_buffer),
            'experience_buffer_size': len(self.ppo_agent.buffer),
            'current_learning_rate': self.ppo_agent.config.learning_rate,
            'constitutional_weights': self.ppo_agent.constitutional_ai.principle_weights
        }


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建PPO人类对齐代理
    state_dim = 20
    action_dim = 3
    config = PPOHumanConfig(
        learning_rate=3e-4,
        num_epochs=5,
        batch_size=32,
        buffer_size=512
    )
    
    ppo_agent = PPOWithHumanAlignment(state_dim, action_dim, config)
    
    # 创建在线学习管理器
    online_learner = OnlineAdaptiveLearning(ppo_agent, config)
    
    print("Testing PPO with Human Alignment...")
    
    # 模拟训练循环
    for episode in range(10):
        state = np.random.randn(state_dim)
        total_reward = 0
        
        for step in range(50):
            # 选择动作
            action, action_info = ppo_agent.select_action(state)
            
            # 模拟环境反应
            reward = np.random.randn() + action_info['constitutional_score']
            next_state = np.random.randn(state_dim)
            done = step == 49
            
            # 在线学习步骤
            online_learner.step(state, action, reward, next_state, done, action_info)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.3f}")
        
        # 每几个episode输出适应状态
        if (episode + 1) % 3 == 0:
            status = online_learner.get_adaptation_status()
            print(f"Adaptation Status: {status['performance_metrics']}")
    
    print("\nPPO Human Alignment training completed!")
    
    # 测试模型保存
    ppo_agent.save_model("test_ppo_human_alignment.pth")
    print("Model saved successfully!")