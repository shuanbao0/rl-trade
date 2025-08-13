#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF奖励函数

基于人类反馈的强化学习奖励函数，集成了所有RLHF组件：
- 专家反馈收集系统
- 偏好学习模型
- 奖励模型训练
- PPO人类对齐算法
- 宪法AI安全框架
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import pickle
import os
from pathlib import Path

from .base_reward import BaseRewardScheme
from .rlhf_components import (
    ExpertFeedbackInterface,
    ExpertiseLevel,
    FeedbackType,
    BradleyTerryPreferenceModel,
    PreferenceLearningTrainer,
    MultiExpertPreferenceFusion,
    PreferenceLearningMethod,
    PreferenceTrainingConfig,
    PreferenceData,
    create_preference_learner,
    CritiqueGuidedRewardModel,
    RewardModelTrainer,
    RewardModelType,
    RewardModelConfig,
    RewardTrainingData,
    create_reward_model_trainer,
    PPOWithHumanAlignment,
    OnlineAdaptiveLearning,
    ConstitutionalAI,
    PPOHumanConfig,
    AlignmentObjective,
    PolicyNetwork
)

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """RLHF奖励函数配置"""
    
    # 基础奖励参数
    base_reward_weight: float = 0.4
    human_alignment_weight: float = 0.3
    constitutional_weight: float = 0.2
    preference_weight: float = 0.1
    
    # 专家反馈配置
    expert_feedback_enabled: bool = True
    min_expert_confidence: float = 0.6
    expert_feedback_decay: float = 0.95
    feedback_collection_frequency: int = 100
    
    # 偏好学习配置
    preference_learning_method: PreferenceLearningMethod = PreferenceLearningMethod.BRADLEY_TERRY
    preference_model_update_frequency: int = 200
    preference_batch_size: int = 32
    
    # 奖励模型配置
    reward_model_type: RewardModelType = RewardModelType.CRITIQUE_GUIDED
    reward_model_update_frequency: int = 500
    reward_learning_rate: float = 0.001
    
    # PPO人类对齐配置
    ppo_enabled: bool = True
    ppo_learning_rate: float = 3e-4
    ppo_update_frequency: int = 50
    alignment_objective: AlignmentObjective = AlignmentObjective.HUMAN_PREFERENCE
    
    # 宪法AI配置
    constitutional_principles: List[str] = field(default_factory=lambda: [
        "maximize_trading_profit",
        "minimize_risk_exposure",
        "ensure_market_fairness",
        "avoid_market_manipulation",
        "maintain_transparency"
    ])
    
    # 在线学习配置
    online_adaptation_enabled: bool = True
    adaptation_learning_rate: float = 0.01
    memory_buffer_size: int = 1000
    
    # 数据持久化配置
    save_feedback_data: bool = True
    model_save_frequency: int = 1000
    feedback_database_path: str = "rlhf_feedback.db"
    model_save_path: str = "rlhf_models/"


class RLHFReward(BaseRewardScheme):
    """
    RLHF (Reinforcement Learning from Human Feedback) 奖励函数
    
    这是一个先进的奖励函数，结合了人类专家反馈、偏好学习、
    奖励模型训练、PPO人类对齐算法和宪法AI安全框架。
    
    主要特性：
    1. 专家反馈收集和管理
    2. 多种偏好学习方法（Bradley-Terry、神经网络）
    3. 多专家偏好融合机制
    4. 奖励模型训练（点评引导、分层、适应性）
    5. PPO人类对齐算法
    6. 宪法AI安全约束
    7. 在线适应性学习
    8. 完整的数据持久化
    """
    
    def __init__(self, config: Optional[RLHFConfig] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.config = config or RLHFConfig()
        self.step_count = 0
        self.last_update_step = 0
        
        # 初始化RLHF组件
        self._initialize_components()
        
        # 性能监控
        self.performance_metrics = {
            'total_rewards': [],
            'human_alignment_scores': [],
            'constitutional_scores': [],
            'preference_accuracies': [],
            'expert_feedback_counts': [],
            'model_update_counts': 0
        }
        
        # 数据缓存
        self.state_action_cache = []
        self.reward_cache = []
        self.feedback_cache = []
        
        logger.info(f"Initialized RLHFReward with config: {self.config}")
    
    def _initialize_components(self):
        """初始化所有RLHF组件"""
        
        # 1. 专家反馈收集接口
        if self.config.expert_feedback_enabled:
            self.expert_interface = ExpertFeedbackInterface(
                db_path=self.config.feedback_database_path
            )
            logger.info("Initialized Expert Feedback Interface")
        else:
            self.expert_interface = None
        
        # 2. 偏好学习模型
        preference_config = PreferenceTrainingConfig(
            method=self.config.preference_learning_method,
            batch_size=self.config.preference_batch_size,
            learning_rate=self.config.ppo_learning_rate
        )
        
        self.preference_learner = PreferenceLearningTrainer(preference_config)
        self.preference_fusion = MultiExpertPreferenceFusion(preference_config)
        logger.info(f"Initialized Preference Learning: {self.config.preference_learning_method.value}")
        
        # 3. 奖励模型
        reward_config = RewardModelConfig(
            model_type=self.config.reward_model_type,
            learning_rate=self.config.reward_learning_rate,
            num_epochs=50,
            batch_size=32
        )
        
        self.reward_model_trainer = RewardModelTrainer(reward_config)
        self.reward_model_trained = False
        logger.info(f"Initialized Reward Model: {self.config.reward_model_type.value}")
        
        # 4. PPO人类对齐算法
        if self.config.ppo_enabled:
            ppo_config = PPOHumanConfig(
                learning_rate=self.config.ppo_learning_rate,
                alignment_objective=self.config.alignment_objective,
                constitutional_principles=self.config.constitutional_principles,
                adaptation_frequency=self.config.ppo_update_frequency
            )
            
            # 估算状态和动作维度
            state_dim = 50  # 假设50维状态空间
            action_dim = 3   # 假设3维动作空间 [position, size, timing]
            
            self.ppo_agent = PPOWithHumanAlignment(state_dim, action_dim, ppo_config)
            
            # 在线适应性学习
            if self.config.online_adaptation_enabled:
                self.online_learner = OnlineAdaptiveLearning(self.ppo_agent, ppo_config)
            else:
                self.online_learner = None
            
            logger.info("Initialized PPO Human Alignment")
        else:
            self.ppo_agent = None
            self.online_learner = None
        
        # 5. 创建模型保存目录
        if self.config.save_feedback_data:
            save_dir = Path(self.config.model_save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def reward(self, env) -> float:
        """
        TensorTrade框架要求的reward方法
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 计算得到的奖励值
        """
        # 更新基类状态
        state = self.update_state(env)
        
        # 从环境中提取信息
        observation = self._extract_observation_from_env(env)
        action = self._extract_action_from_env(env)
        info = self._extract_info_from_env(env, state)
        
        return self.calculate_reward(observation, action, info)
    
    def get_reward(self, portfolio) -> float:
        """
        TensorTrade框架要求的get_reward方法
        
        Args:
            portfolio: 投资组合对象
            
        Returns:
            float: 计算得到的奖励值
        """
        # 从portfolio中提取必要信息
        current_value = float(portfolio.net_worth)
        
        # 构建简化的观察、动作和信息
        observation = {
            'close_price': current_value,
            'volume': 1000000,
            'returns': 0.0,
            'volatility': 0.2
        }
        
        action = {
            'position': 0.0,
            'size': 0.0,
            'direction': 0
        }
        
        info = {
            'portfolio_return': 0.0,
            'pnl': 0.0
        }
        
        return self.calculate_reward(observation, action, info)
    
    def _extract_observation_from_env(self, env) -> Dict[str, Any]:
        """从环境中提取观察信息"""
        observation = {}
        
        try:
            # 尝试从观察器获取
            if hasattr(env, 'observer') and hasattr(env.observer, '_current_observation'):
                obs = env.observer._current_observation
                if isinstance(obs, dict):
                    observation.update(obs)
                elif hasattr(obs, 'values'):
                    # 如果是pandas Series或类似结构
                    observation.update(obs.to_dict() if hasattr(obs, 'to_dict') else {})
            
            # 基本市场数据
            if 'close' not in observation:
                observation['close_price'] = 100.0
            if 'volume' not in observation:
                observation['volume'] = 1000000
            if 'returns' not in observation:
                observation['returns'] = 0.0
            
        except Exception:
            # 默认观察
            observation = {
                'close_price': 100.0,
                'volume': 1000000,
                'returns': 0.0,
                'volatility': 0.2
            }
        
        return observation
    
    def _extract_action_from_env(self, env) -> Dict[str, Any]:
        """从环境中提取动作信息"""
        action = {}
        
        try:
            current_action = self.get_current_action(env)
            action = {
                'position': current_action,
                'size': abs(current_action),
                'direction': 1 if current_action > 0 else -1 if current_action < 0 else 0
            }
        except Exception:
            action = {
                'position': 0.0,
                'size': 0.0,
                'direction': 0
            }
        
        return action
    
    def _extract_info_from_env(self, env, state: Dict[str, float]) -> Dict[str, Any]:
        """从环境状态中提取信息"""
        info = {
            'portfolio_return': state.get('step_return_pct', 0.0) / 100.0,
            'pnl': state.get('current_value', 0) - state.get('previous_value', 0),
            'total_return': state.get('total_return_pct', 0.0) / 100.0,
            'step_count': state.get('step_count', 0)
        }
        
        return info

    def calculate_reward(self, observation: Dict[str, Any], action: Dict[str, Any], 
                        info: Dict[str, Any]) -> float:
        """
        计算RLHF奖励
        
        组合多个奖励源：
        1. 基础交易奖励
        2. 人类对齐奖励
        3. 宪法AI奖励
        4. 偏好学习奖励
        """
        self.step_count += 1
        
        try:
            # 提取特征
            state_features = self._extract_state_features(observation)
            action_features = self._extract_action_features(action)
            
            # 1. 计算基础奖励
            base_reward = self._calculate_base_reward(observation, action, info)
            
            # 2. 计算人类对齐奖励
            human_alignment_reward = self._calculate_human_alignment_reward(
                state_features, action_features, info
            )
            
            # 3. 计算宪法AI奖励
            constitutional_reward = self._calculate_constitutional_reward(
                state_features, action_features, info
            )
            
            # 4. 计算偏好学习奖励
            preference_reward = self._calculate_preference_reward(
                state_features, action_features
            )
            
            # 5. 组合奖励
            total_reward = (
                self.config.base_reward_weight * base_reward +
                self.config.human_alignment_weight * human_alignment_reward +
                self.config.constitutional_weight * constitutional_reward +
                self.config.preference_weight * preference_reward
            )
            
            # 6. 缓存数据用于后续学习
            self._cache_experience(state_features, action_features, total_reward, info)
            
            # 7. 定期更新模型
            self._maybe_update_models()
            
            # 8. 记录性能指标
            self._update_performance_metrics(
                total_reward, human_alignment_reward, constitutional_reward
            )
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Error calculating RLHF reward: {e}")
            return 0.0
    
    def _extract_state_features(self, observation: Dict[str, Any]) -> np.ndarray:
        """从观察中提取状态特征"""
        features = []
        
        # 价格相关特征
        if 'close_price' in observation:
            features.append(observation['close_price'])
        if 'volume' in observation:
            features.append(observation['volume'])
        if 'returns' in observation:
            features.append(observation['returns'])
        
        # 技术指标特征
        technical_indicators = [
            'sma_10', 'sma_20', 'ema_12', 'ema_26', 'macd',
            'rsi', 'bb_upper', 'bb_lower', 'atr', 'obv'
        ]
        
        for indicator in technical_indicators:
            if indicator in observation:
                features.append(observation[indicator])
            else:
                features.append(0.0)  # 默认值
        
        # 市场状态特征
        market_features = ['volatility', 'trend', 'momentum', 'volume_profile']
        for feature in market_features:
            if feature in observation:
                features.append(observation[feature])
            else:
                features.append(0.0)
        
        # 组合成固定长度的特征向量（50维）
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)
    
    def _extract_action_features(self, action: Dict[str, Any]) -> np.ndarray:
        """从动作中提取特征"""
        features = []
        
        # 基础动作特征
        if 'position' in action:
            features.append(action['position'])
        if 'size' in action:
            features.append(action['size'])
        if 'direction' in action:
            features.append(action['direction'])
        
        # 确保固定长度（3维）
        while len(features) < 3:
            features.append(0.0)
        
        return np.array(features[:3], dtype=np.float32)
    
    def _calculate_base_reward(self, observation: Dict[str, Any], 
                             action: Dict[str, Any], info: Dict[str, Any]) -> float:
        """计算基础交易奖励"""
        # 简化的基础奖励：基于收益率
        if 'portfolio_return' in info:
            return info['portfolio_return']
        elif 'pnl' in info:
            return info['pnl'] / 10000.0  # 标准化
        else:
            return 0.0
    
    def _calculate_human_alignment_reward(self, state_features: np.ndarray, 
                                        action_features: np.ndarray, 
                                        info: Dict[str, Any]) -> float:
        """计算人类对齐奖励"""
        if not self.ppo_agent:
            return 0.0
        
        try:
            # 使用PPO代理评估动作
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
            with torch.no_grad():
                _, _, preference_score = self.ppo_agent.policy.forward(state_tensor)
                alignment_score = preference_score.item()
            
            return alignment_score - 0.5  # 转换为 [-0.5, 0.5] 范围
            
        except Exception as e:
            logger.warning(f"Error calculating human alignment reward: {e}")
            return 0.0
    
    def _calculate_constitutional_reward(self, state_features: np.ndarray, 
                                       action_features: np.ndarray, 
                                       info: Dict[str, Any]) -> float:
        """计算宪法AI奖励"""
        if not self.ppo_agent:
            return 0.0
        
        try:
            # 提取市场上下文
            market_context = {
                'trend': float(state_features[15]) if len(state_features) > 15 else 0.0,
                'volatility': float(state_features[16]) if len(state_features) > 16 else 0.5,
                'volume': float(state_features[1]) if len(state_features) > 1 else 1000000,
                'recent_trades': 0
            }
            
            # 评估宪法合规性
            constitutional_scores = self.ppo_agent.constitutional_ai.evaluate_action(
                state_features, action_features, market_context
            )
            
            overall_score = self.ppo_agent.constitutional_ai.compute_constitutional_score(
                constitutional_scores
            )
            
            return overall_score - 0.5  # 转换为 [-0.5, 0.5] 范围
            
        except Exception as e:
            logger.warning(f"Error calculating constitutional reward: {e}")
            return 0.0
    
    def _calculate_preference_reward(self, state_features: np.ndarray, 
                                   action_features: np.ndarray) -> float:
        """计算偏好学习奖励"""
        try:
            if not hasattr(self.preference_learner, 'model') or self.preference_learner.model is None:
                return 0.0
            
            # 生成对比动作
            alternative_action = action_features + np.random.normal(0, 0.1, action_features.shape)
            
            # 预测偏好
            prob_preferred, confidence = self.preference_learner.predict_preference(
                state_features, alternative_action
            )
            
            # 转换为奖励信号
            preference_reward = (prob_preferred - 0.5) * confidence
            
            return preference_reward
            
        except Exception as e:
            logger.warning(f"Error calculating preference reward: {e}")
            return 0.0
    
    def _cache_experience(self, state_features: np.ndarray, action_features: np.ndarray, 
                         reward: float, info: Dict[str, Any]):
        """缓存经验用于后续学习"""
        experience = {
            'state': state_features,
            'action': action_features,
            'reward': reward,
            'timestamp': datetime.now(),
            'info': info
        }
        
        self.state_action_cache.append(experience)
        self.reward_cache.append(reward)
        
        # 限制缓存大小
        max_cache_size = self.config.memory_buffer_size
        if len(self.state_action_cache) > max_cache_size:
            self.state_action_cache = self.state_action_cache[-max_cache_size:]
            self.reward_cache = self.reward_cache[-max_cache_size:]
    
    def _maybe_update_models(self):
        """根据配置定期更新模型"""
        
        # 检查是否需要更新
        steps_since_update = self.step_count - self.last_update_step
        
        # 更新偏好模型
        if (steps_since_update >= self.config.preference_model_update_frequency and 
            len(self.state_action_cache) >= self.config.preference_batch_size):
            self._update_preference_model()
        
        # 更新奖励模型
        if (steps_since_update >= self.config.reward_model_update_frequency and 
            len(self.state_action_cache) >= 100):
            self._update_reward_model()
        
        # 更新PPO模型
        if (self.ppo_agent and steps_since_update >= self.config.ppo_update_frequency and 
            len(self.ppo_agent.buffer) >= self.config.preference_batch_size):
            self._update_ppo_model()
        
        # 保存模型
        if (self.config.save_feedback_data and 
            steps_since_update >= self.config.model_save_frequency):
            self._save_models()
            self.last_update_step = self.step_count
    
    def _update_preference_model(self):
        """更新偏好学习模型"""
        try:
            if len(self.state_action_cache) < 20:  # 需要足够的数据
                return
            
            # 生成偏好数据
            preference_data = self._generate_preference_data()
            
            if len(preference_data) > 0:
                # 多专家融合
                fused_data = self.preference_fusion.fuse_preferences(preference_data)
                
                # 训练偏好模型
                if len(fused_data) >= 5:
                    results = self.preference_learner.train(fused_data)
                    logger.info(f"Updated preference model: {results}")
                    
                    self.performance_metrics['preference_accuracies'].append(
                        results.get('accuracy', 0.0)
                    )
            
        except Exception as e:
            logger.error(f"Error updating preference model: {e}")
    
    def _update_reward_model(self):
        """更新奖励模型"""
        try:
            if len(self.state_action_cache) < 50:
                return
            
            # 准备训练数据
            training_data = self._prepare_reward_training_data()
            
            # 训练奖励模型
            results = self.reward_model_trainer.train(training_data)
            self.reward_model_trained = True
            
            logger.info(f"Updated reward model: {results}")
            self.performance_metrics['model_update_counts'] += 1
            
        except Exception as e:
            logger.error(f"Error updating reward model: {e}")
    
    def _update_ppo_model(self):
        """更新PPO模型"""
        try:
            if not self.ppo_agent:
                return
            
            # 收集人类反馈
            feedback_list = self._collect_simulated_feedback()
            
            # 更新PPO策略
            if len(feedback_list) > 0:
                update_stats = self.ppo_agent.update_policy(feedback_list)
                logger.info(f"Updated PPO model: {update_stats}")
            
            # 在线适应性学习
            if self.online_learner:
                # 模拟一步在线学习
                if len(self.state_action_cache) > 0:
                    latest_experience = self.state_action_cache[-1]
                    next_state = np.random.randn(50)  # 模拟下一状态
                    
                    action_info = {
                        'constitutional_score': 0.8,
                        'preference_score': 0.6
                    }
                    
                    self.online_learner.step(
                        latest_experience['state'],
                        latest_experience['action'], 
                        latest_experience['reward'],
                        next_state,
                        False,  # not done
                        action_info
                    )
            
        except Exception as e:
            logger.error(f"Error updating PPO model: {e}")
    
    def _generate_preference_data(self) -> List[PreferenceData]:
        """生成偏好数据用于训练"""
        preference_data = []
        
        if len(self.state_action_cache) < 2:
            return preference_data
        
        # 随机选择经验对进行比较
        for _ in range(min(10, len(self.state_action_cache) // 2)):
            try:
                indices = np.random.choice(len(self.state_action_cache), 2, replace=False)
                exp_a = self.state_action_cache[indices[0]]
                exp_b = self.state_action_cache[indices[1]]
                
                # 基于奖励确定偏好
                preference = 0 if exp_a['reward'] > exp_b['reward'] else 1
                confidence = min(0.9, abs(exp_a['reward'] - exp_b['reward']) + 0.6)
                
                pref_data = PreferenceData(
                    expert_id="simulated_expert",
                    scenario_a_features=exp_a['state'],
                    scenario_b_features=exp_b['state'],
                    preference=preference,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    expert_reliability=0.8
                )
                
                preference_data.append(pref_data)
                
            except Exception as e:
                logger.warning(f"Error generating preference data: {e}")
                continue
        
        return preference_data
    
    def _prepare_reward_training_data(self) -> RewardTrainingData:
        """准备奖励模型训练数据"""
        
        states = [exp['state'] for exp in self.state_action_cache]
        actions = [exp['action'] for exp in self.state_action_cache]
        rewards = [exp['reward'] for exp in self.state_action_cache]
        
        # 生成置信度分数
        confidence_scores = np.clip(np.abs(rewards) + 0.5, 0.5, 1.0)
        
        expert_ids = ["rlhf_system"] * len(rewards)
        timestamps = [exp['timestamp'] for exp in self.state_action_cache]
        
        return RewardTrainingData(
            state_features=np.array(states),
            action_features=np.array(actions),
            reward_labels=np.array(rewards),
            confidence_scores=confidence_scores,
            expert_ids=expert_ids,
            timestamps=timestamps
        )
    
    def _collect_simulated_feedback(self) -> List[Dict[str, Any]]:
        """收集模拟的人类反馈"""
        feedback_list = []
        
        if len(self.state_action_cache) < 5:
            return feedback_list
        
        # 基于最近的经验生成反馈
        recent_experiences = self.state_action_cache[-5:]
        
        for exp in recent_experiences:
            if np.random.random() < 0.3:  # 30%概率生成反馈
                feedback = {
                    'type': 'scalar',
                    'action_id': 0,
                    'rating': np.clip(exp['reward'] + 0.5, 0.0, 1.0),
                    'confidence': 0.8,
                    'expert_id': 'simulated_expert'
                }
                feedback_list.append(feedback)
        
        return feedback_list
    
    def _save_models(self):
        """保存所有模型"""
        try:
            save_dir = Path(self.config.model_save_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存PPO模型
            if self.ppo_agent:
                ppo_path = save_dir / f"ppo_model_{timestamp}.pth"
                self.ppo_agent.save_model(str(ppo_path))
            
            # 保存偏好学习训练历史
            if hasattr(self.preference_learner, 'training_history'):
                pref_path = save_dir / f"preference_history_{timestamp}.json"
                with open(pref_path, 'w') as f:
                    json.dump(self.preference_learner.training_history, f, default=str)
            
            # 保存性能指标
            metrics_path = save_dir / f"performance_metrics_{timestamp}.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, default=str)
            
            # 保存经验缓存
            cache_path = save_dir / f"experience_cache_{timestamp}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(self.state_action_cache[-1000:], f)  # 只保存最近1000条
            
            logger.info(f"Saved RLHF models to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _update_performance_metrics(self, total_reward: float, 
                                  human_alignment_reward: float, 
                                  constitutional_reward: float):
        """更新性能指标"""
        self.performance_metrics['total_rewards'].append(total_reward)
        self.performance_metrics['human_alignment_scores'].append(human_alignment_reward)
        self.performance_metrics['constitutional_scores'].append(constitutional_reward)
        
        # 限制历史长度
        max_history = 1000
        for key in ['total_rewards', 'human_alignment_scores', 'constitutional_scores']:
            if len(self.performance_metrics[key]) > max_history:
                self.performance_metrics[key] = self.performance_metrics[key][-max_history:]
    
    def add_expert_feedback(self, expert_name: str, expertise_level: ExpertiseLevel,
                          specialization: List[str], contact_info: Dict[str, str]) -> str:
        """添加人类专家到系统"""
        if not self.expert_interface:
            logger.warning("Expert feedback interface not enabled")
            return ""
        
        try:
            expert = self.expert_interface.register_expert(
                name=expert_name,
                expertise_level=expertise_level,
                specialization=specialization,
                contact_info=contact_info
            )
            
            logger.info(f"Added expert: {expert.name} ({expert.expert_id})")
            return expert.expert_id
            
        except Exception as e:
            logger.error(f"Error adding expert: {e}")
            return ""
    
    def collect_human_preference(self, expert_id: str, scenario_a_features: np.ndarray,
                               scenario_b_features: np.ndarray, preferred_scenario: str,
                               confidence: float, reasoning: str = "") -> bool:
        """收集人类偏好反馈"""
        if not self.expert_interface:
            return False
        
        try:
            # 这里需要将特征转换为场景对象
            # 简化实现，直接创建偏好数据
            preference_data = PreferenceData(
                expert_id=expert_id,
                scenario_a_features=scenario_a_features,
                scenario_b_features=scenario_b_features,
                preference=0 if preferred_scenario == "A" else 1,
                confidence=confidence,
                timestamp=datetime.now(),
                expert_reliability=0.9
            )
            
            self.feedback_cache.append(preference_data)
            
            # 如果收集了足够的反馈，立即更新偏好模型
            if len(self.feedback_cache) >= 5:
                fused_data = self.preference_fusion.fuse_preferences(self.feedback_cache)
                if len(fused_data) > 0:
                    self.preference_learner.train(fused_data)
                    self.feedback_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting human preference: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'total_steps': self.step_count,
            'models_trained': self.reward_model_trained,
            'ppo_enabled': self.config.ppo_enabled,
            'expert_feedback_enabled': self.config.expert_feedback_enabled,
        }
        
        # 计算统计指标
        if self.performance_metrics['total_rewards']:
            rewards = self.performance_metrics['total_rewards']
            summary['average_reward'] = np.mean(rewards)
            summary['reward_std'] = np.std(rewards)
            summary['latest_reward'] = rewards[-1] if rewards else 0.0
        
        if self.performance_metrics['human_alignment_scores']:
            alignment_scores = self.performance_metrics['human_alignment_scores']
            summary['average_alignment'] = np.mean(alignment_scores)
            summary['alignment_trend'] = np.mean(alignment_scores[-10:]) if len(alignment_scores) >= 10 else 0.0
        
        if self.performance_metrics['constitutional_scores']:
            constitutional_scores = self.performance_metrics['constitutional_scores']
            summary['average_constitutional'] = np.mean(constitutional_scores)
            summary['constitutional_compliance'] = np.mean(np.array(constitutional_scores) > 0)
        
        # PPO状态
        if self.online_learner:
            ppo_status = self.online_learner.get_adaptation_status()
            summary['ppo_adaptation_step'] = ppo_status['adaptation_step']
            summary['ppo_learning_rate'] = ppo_status['current_learning_rate']
        
        return summary
    
    def reset(self):
        """重置奖励函数状态"""
        self.step_count = 0
        self.last_update_step = 0
        
        # 清空缓存（但保留模型）
        self.state_action_cache.clear()
        self.reward_cache.clear()
        self.feedback_cache.clear()
        
        # 重置性能指标
        for key in self.performance_metrics:
            if isinstance(self.performance_metrics[key], list):
                self.performance_metrics[key].clear()
            else:
                self.performance_metrics[key] = 0
        
        logger.info("Reset RLHF reward function state")


# 工厂函数
def create_rlhf_reward(config: Optional[RLHFConfig] = None, **kwargs) -> RLHFReward:
    """创建RLHF奖励函数实例"""
    return RLHFReward(config=config, **kwargs)


# 使用示例
if __name__ == "__main__":
    # 创建RLHF奖励函数
    config = RLHFConfig(
        base_reward_weight=0.4,
        human_alignment_weight=0.3,
        constitutional_weight=0.2,
        preference_weight=0.1,
        expert_feedback_enabled=True,
        ppo_enabled=True,
        online_adaptation_enabled=True
    )
    
    rlhf_reward = create_rlhf_reward(config)
    
    print("Testing RLHF Reward Function...")
    
    # 模拟交易环境
    for step in range(100):
        # 模拟观察
        observation = {
            'close_price': 100 + np.random.randn(),
            'volume': 1000000 + np.random.randint(-100000, 100000),
            'returns': np.random.randn() * 0.01,
            'sma_10': 100 + np.random.randn(),
            'rsi': np.random.uniform(30, 70),
            'volatility': np.random.uniform(0.1, 0.5)
        }
        
        # 模拟动作
        action = {
            'position': np.random.uniform(-1, 1),
            'size': np.random.uniform(0, 1),
            'direction': np.random.choice([-1, 0, 1])
        }
        
        # 模拟信息
        info = {
            'portfolio_return': np.random.randn() * 0.02,
            'drawdown': np.random.uniform(0, 0.1)
        }
        
        # 计算奖励
        reward = rlhf_reward.calculate_reward(observation, action, info)
        
        if step % 20 == 0:
            print(f"Step {step}: Reward = {reward:.4f}")
    
    # 获取性能摘要
    summary = rlhf_reward.get_performance_summary()
    print(f"\nPerformance Summary: {summary}")
    
    print("\nRLHF Reward Function test completed!")