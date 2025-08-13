#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
奖励模型训练模块

实现基于人类反馈的奖励模型学习，包括：
- 点评引导的奖励模型
- 分层奖励模型
- 在线适应性学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class RewardModelType(Enum):
    """奖励模型类型枚举"""
    CRITIQUE_GUIDED = "critique_guided"
    HIERARCHICAL = "hierarchical"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


@dataclass
class RewardModelConfig:
    """奖励模型配置"""
    model_type: RewardModelType = RewardModelType.CRITIQUE_GUIDED
    
    # 网络架构参数
    input_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    output_dim: int = 1
    dropout_rate: float = 0.3
    
    # 训练参数
    learning_rate: float = 0.0003
    batch_size: int = 64
    num_epochs: int = 200
    validation_split: float = 0.2
    early_stopping_patience: int = 20
    
    # 正则化参数
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # 分层模型特定参数
    hierarchy_levels: int = 3
    level_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # 适应性学习参数
    adaptation_rate: float = 0.1
    memory_size: int = 1000
    update_frequency: int = 10
    
    # 特征工程参数
    feature_normalization: bool = True
    feature_selection_threshold: float = 0.01


@dataclass
class RewardTrainingData:
    """奖励训练数据结构"""
    state_features: np.ndarray
    action_features: np.ndarray
    reward_labels: np.ndarray
    confidence_scores: np.ndarray
    expert_ids: List[str]
    timestamps: List[datetime]
    context_features: Optional[np.ndarray] = None
    critique_text: Optional[List[str]] = None


class RewardDataset(Dataset):
    """PyTorch奖励数据集"""
    
    def __init__(self, training_data: RewardTrainingData, config: RewardModelConfig):
        self.config = config
        self.reward_labels = torch.FloatTensor(training_data.reward_labels)
        self.confidence_scores = torch.FloatTensor(training_data.confidence_scores)
        
        # 检查是否已经是预处理过的组合特征
        if training_data.action_features.shape[1] == 1 and np.all(training_data.action_features == 0):
            # 已经预处理过，state_features 包含组合特征
            self.combined_features = torch.FloatTensor(training_data.state_features)
        else:
            # 原始数据，需要组合
            self.state_features = torch.FloatTensor(training_data.state_features)
            self.action_features = torch.FloatTensor(training_data.action_features)
            self.combined_features = torch.cat([self.state_features, self.action_features], dim=1)
            
            # 上下文特征（如果有）
            if training_data.context_features is not None:
                context_tensor = torch.FloatTensor(training_data.context_features)
                self.combined_features = torch.cat([self.combined_features, context_tensor], dim=1)
    
    def __len__(self):
        return len(self.reward_labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.combined_features[idx],
            'reward': self.reward_labels[idx],
            'confidence': self.confidence_scores[idx]
        }


class CritiqueGuidedRewardModel(nn.Module):
    """点评引导的奖励模型
    
    结合专家文本点评和数值评分训练奖励函数
    """
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        
        # 主要奖励网络
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, config.output_dim))
        
        self.reward_network = nn.Sequential(*layers)
        
        # 置信度估计网络
        self.confidence_network = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 文本编码器（用于处理专家点评）
        self.text_encoder = nn.LSTM(
            input_size=768,  # BERT embedding dimension
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.output_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, config.output_dim)
        )
    
    def forward(self, features: torch.Tensor, 
                text_embeddings: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            features: 状态-动作特征
            text_embeddings: 专家点评文本嵌入
            
        Returns:
            Tuple[reward, confidence]
        """
        # 基础奖励预测
        base_reward = self.reward_network(features)
        
        # 置信度估计
        confidence = self.confidence_network(features)
        
        if text_embeddings is not None:
            # 处理文本嵌入
            text_output, _ = self.text_encoder(text_embeddings)
            text_features = text_output.mean(dim=1)  # 池化
            
            # 融合数值和文本信息
            combined = torch.cat([base_reward, text_features], dim=1)
            final_reward = self.fusion_layer(combined)
        else:
            final_reward = base_reward
        
        return final_reward, confidence


class HierarchicalRewardModel(nn.Module):
    """分层奖励模型
    
    在不同时间尺度上学习奖励函数
    """
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        self.hierarchy_levels = config.hierarchy_levels
        
        # 不同层级的奖励网络
        self.level_networks = nn.ModuleList()
        
        for level in range(self.hierarchy_levels):
            # 每层有不同的感受野和复杂度
            hidden_dims = [dim // (level + 1) for dim in config.hidden_dims]
            
            layers = []
            prev_dim = config.input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            
            self.level_networks.append(nn.Sequential(*layers))
        
        # 层级权重（可学习）
        self.level_weights = nn.Parameter(
            torch.tensor(config.level_weights[:self.hierarchy_levels])
        )
        
        # 时间注意力机制 - 确保embed_dim能被num_heads整除
        num_heads = 8
        if config.input_dim % num_heads != 0:
            # 调整embed_dim为最接近的可被num_heads整除的数
            adjusted_dim = ((config.input_dim // num_heads) + 1) * num_heads
            self.input_projection = nn.Linear(config.input_dim, adjusted_dim)
            attention_dim = adjusted_dim
        else:
            self.input_projection = None
            attention_dim = config.input_dim
            
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=0.1
        )
    
    def forward(self, features: torch.Tensor, 
                temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """分层前向传播"""
        
        # 如果有时间上下文，应用注意力机制
        if temporal_context is not None:
            # 如果需要投影到合适的维度
            if self.input_projection is not None:
                features = self.input_projection(features)
                temporal_context = self.input_projection(temporal_context)
            
            attended_features, _ = self.temporal_attention(
                features.unsqueeze(0), 
                temporal_context, 
                temporal_context
            )
            features = attended_features.squeeze(0)
        
        # 计算各层级奖励
        level_rewards = []
        for level_network in self.level_networks:
            level_reward = level_network(features)
            level_rewards.append(level_reward)
        
        # 加权组合
        level_rewards = torch.stack(level_rewards, dim=2)
        weights = torch.softmax(self.level_weights, dim=0)
        
        final_reward = torch.sum(level_rewards * weights, dim=2)
        
        return final_reward


class AdaptiveRewardModel(nn.Module):
    """适应性奖励模型
    
    能够根据新的反馈在线更新的奖励模型
    """
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        
        # 核心网络
        self.core_network = CritiqueGuidedRewardModel(config)
        
        # 适应层
        self.adaptation_layer = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 记忆缓冲区
        self.memory_buffer = deque(maxlen=config.memory_size)
        self.adaptation_optimizer = None
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        base_reward, confidence = self.core_network(features)
        adaptation = self.adaptation_layer(features)
        
        final_reward = base_reward + adaptation
        return final_reward, confidence
    
    def update_memory(self, features: np.ndarray, reward: float, confidence: float):
        """更新记忆缓冲区"""
        self.memory_buffer.append({
            'features': features,
            'reward': reward,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
    
    def adapt(self):
        """在线适应更新"""
        if len(self.memory_buffer) < self.config.batch_size:
            return
        
        # 从记忆中采样
        recent_data = list(self.memory_buffer)[-self.config.batch_size:]
        
        features = torch.FloatTensor([d['features'] for d in recent_data])
        rewards = torch.FloatTensor([d['reward'] for d in recent_data])
        confidences = torch.FloatTensor([d['confidence'] for d in recent_data])
        
        # 快速适应更新
        if self.adaptation_optimizer is None:
            self.adaptation_optimizer = optim.Adam(
                self.adaptation_layer.parameters(),
                lr=self.config.adaptation_rate
            )
        
        self.adaptation_optimizer.zero_grad()
        
        predicted_rewards = self.adaptation_layer(features).squeeze()
        loss = nn.MSELoss()(predicted_rewards, rewards)
        
        loss.backward()
        self.adaptation_optimizer.step()


class RewardModelTrainer:
    """奖励模型训练器"""
    
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if config.feature_normalization else None
        self.training_history = []
        
    def train(self, training_data: RewardTrainingData) -> Dict[str, Any]:
        """训练奖励模型"""
        logger.info(f"Training reward model: {self.config.model_type.value}")
        
        # 数据预处理
        processed_data = self._preprocess_data(training_data)
        
        # 创建模型
        self.model = self._create_model(processed_data)
        
        # 训练
        if self.config.model_type == RewardModelType.CRITIQUE_GUIDED:
            return self._train_critique_guided(processed_data)
        elif self.config.model_type == RewardModelType.HIERARCHICAL:
            return self._train_hierarchical(processed_data)
        elif self.config.model_type == RewardModelType.ADAPTIVE:
            return self._train_adaptive(processed_data)
        else:
            raise NotImplementedError(f"Model type {self.config.model_type.value} not implemented")
    
    def predict_reward(self, state_features: np.ndarray, 
                      action_features: np.ndarray,
                      context_features: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """预测奖励值
        
        Returns:
            Tuple[reward, confidence]
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # 组合特征
        combined = np.concatenate([state_features, action_features])
        if context_features is not None:
            combined = np.concatenate([combined, context_features])
        
        # 标准化
        if self.scaler is not None:
            combined = self.scaler.transform(combined.reshape(1, -1)).flatten()
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(combined).unsqueeze(0)
            
            if isinstance(self.model, CritiqueGuidedRewardModel):
                reward, confidence = self.model(features_tensor)
                return reward.item(), confidence.item()
            else:
                reward = self.model(features_tensor)
                return reward.item(), 0.8  # 默认置信度
    
    def _preprocess_data(self, training_data: RewardTrainingData) -> RewardTrainingData:
        """数据预处理"""
        # 组合特征
        combined_features = np.concatenate([
            training_data.state_features,
            training_data.action_features
        ], axis=1)
        
        if training_data.context_features is not None:
            combined_features = np.concatenate([
                combined_features,
                training_data.context_features
            ], axis=1)
        
        # 特征标准化
        if self.scaler is not None:
            combined_features = self.scaler.fit_transform(combined_features)
        
        # 更新配置中的输入维度
        self.config.input_dim = combined_features.shape[1]
        logger.info(f"Updated input dimension to: {self.config.input_dim}")
        
        # 创建新的训练数据对象
        processed_data = RewardTrainingData(
            state_features=combined_features,  # 实际上是组合特征
            action_features=np.zeros((len(combined_features), 1)),  # 占位符
            reward_labels=training_data.reward_labels,
            confidence_scores=training_data.confidence_scores,
            expert_ids=training_data.expert_ids,
            timestamps=training_data.timestamps,
            context_features=None,  # 已经组合
            critique_text=training_data.critique_text
        )
        
        return processed_data
    
    def _create_model(self, training_data: RewardTrainingData) -> nn.Module:
        """创建模型"""
        # 确保输入维度已正确设置
        input_dim = training_data.state_features.shape[1]
        self.config.input_dim = input_dim
        logger.info(f"Creating model with input dimension: {input_dim}")
        
        if self.config.model_type == RewardModelType.CRITIQUE_GUIDED:
            return CritiqueGuidedRewardModel(self.config)
        elif self.config.model_type == RewardModelType.HIERARCHICAL:
            return HierarchicalRewardModel(self.config)
        elif self.config.model_type == RewardModelType.ADAPTIVE:
            return AdaptiveRewardModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _train_critique_guided(self, training_data: RewardTrainingData) -> Dict[str, Any]:
        """训练点评引导模型"""
        dataset = RewardDataset(training_data, self.config)
        
        # 划分训练和验证集
        train_size = int((1 - self.config.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        # 优化器和损失函数
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        reward_criterion = nn.MSELoss(reduction='none')
        confidence_criterion = nn.BCELoss(reduction='none')
        
        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # 前向传播
                reward_pred, confidence_pred = self.model(batch['features'])
                
                # 计算损失
                reward_loss = reward_criterion(
                    reward_pred.squeeze(), 
                    batch['reward']
                )
                
                # 基于真实置信度的标签
                confidence_target = (batch['confidence'] > 0.7).float()
                confidence_loss = confidence_criterion(
                    confidence_pred.squeeze(),
                    confidence_target
                )
                
                # 加权损失
                weights = batch['confidence']
                total_loss = (
                    (reward_loss * weights).mean() + 
                    0.1 * (confidence_loss * weights).mean()
                )
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
                
                optimizer.step()
                epoch_train_loss += total_loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    reward_pred, confidence_pred = self.model(batch['features'])
                    
                    reward_loss = reward_criterion(
                        reward_pred.squeeze(), 
                        batch['reward']
                    )
                    
                    confidence_target = (batch['confidence'] > 0.7).float()
                    confidence_loss = confidence_criterion(
                        confidence_pred.squeeze(),
                        confidence_target
                    )
                    
                    weights = batch['confidence']
                    total_loss = (
                        (reward_loss * weights).mean() + 
                        0.1 * (confidence_loss * weights).mean()
                    )
                    
                    epoch_val_loss += total_loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}], "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}")
        
        # 计算最终指标
        final_metrics = self._compute_final_metrics(val_loader)
        
        results = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'num_epochs_trained': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses,
            **final_metrics
        }
        
        self.training_history.append({
            'model_type': 'critique_guided',
            'timestamp': datetime.now(),
            'results': results,
            'config': self.config.__dict__
        })
        
        logger.info(f"Critique-guided model training completed. "
                   f"Best val loss: {best_val_loss:.4f}")
        
        return results
    
    def _train_hierarchical(self, training_data: RewardTrainingData) -> Dict[str, Any]:
        """训练分层模型"""
        # 简化实现，主要训练逻辑类似
        dataset = RewardDataset(training_data, self.config)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                predicted_reward = self.model(batch['features'])
                loss = criterion(predicted_reward.squeeze(), batch['reward'])
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Hierarchical - Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}")
        
        results = {
            'final_loss': losses[-1],
            'training_losses': losses,
            'num_epochs': self.config.num_epochs
        }
        
        return results
    
    def _train_adaptive(self, training_data: RewardTrainingData) -> Dict[str, Any]:
        """训练适应性模型"""
        # 先训练核心网络
        base_results = self._train_critique_guided(training_data)
        
        # 初始化适应性组件
        if hasattr(self.model, 'adaptation_optimizer'):
            self.model.adaptation_optimizer = optim.Adam(
                self.model.adaptation_layer.parameters(),
                lr=self.config.adaptation_rate
            )
        
        results = {
            **base_results,
            'adaptation_ready': True,
            'memory_size': self.config.memory_size
        }
        
        return results
    
    def _compute_final_metrics(self, val_loader: DataLoader) -> Dict[str, float]:
        """计算最终评估指标"""
        self.model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 处理不同模型的返回值
                model_output = self.model(batch['features'])
                
                if isinstance(model_output, tuple):
                    reward_pred, _ = model_output
                else:
                    reward_pred = model_output
                
                mse = torch.mean((reward_pred.squeeze() - batch['reward']) ** 2)
                mae = torch.mean(torch.abs(reward_pred.squeeze() - batch['reward']))
                
                total_mse += mse.item() * len(batch['reward'])
                total_mae += mae.item() * len(batch['reward'])
                num_samples += len(batch['reward'])
        
        return {
            'mse': total_mse / num_samples,
            'mae': total_mae / num_samples,
            'rmse': np.sqrt(total_mse / num_samples)
        }


def create_reward_model_trainer(
    model_type: RewardModelType = RewardModelType.CRITIQUE_GUIDED,
    **kwargs
) -> RewardModelTrainer:
    """工厂函数创建奖励模型训练器"""
    
    config = RewardModelConfig(model_type=model_type, **kwargs)
    return RewardModelTrainer(config)


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建模拟训练数据
    np.random.seed(42)
    
    num_samples = 1000
    state_dim = 50
    action_dim = 10
    
    # 模拟数据
    state_features = np.random.randn(num_samples, state_dim)
    action_features = np.random.randn(num_samples, action_dim)
    
    # 模拟真实奖励（基于某种模式）
    true_rewards = (
        np.sum(state_features[:, :5], axis=1) + 
        np.sum(action_features[:, :3], axis=1) +
        np.random.normal(0, 0.1, num_samples)
    )
    
    confidence_scores = np.random.uniform(0.6, 0.95, num_samples)
    expert_ids = [f"expert_{i % 5}" for i in range(num_samples)]
    timestamps = [datetime.now() for _ in range(num_samples)]
    
    training_data = RewardTrainingData(
        state_features=state_features,
        action_features=action_features,
        reward_labels=true_rewards,
        confidence_scores=confidence_scores,
        expert_ids=expert_ids,
        timestamps=timestamps
    )
    
    # 测试点评引导模型
    print("Testing Critique-Guided Reward Model...")
    cg_trainer = create_reward_model_trainer(
        RewardModelType.CRITIQUE_GUIDED,
        num_epochs=50,
        batch_size=32
    )
    cg_results = cg_trainer.train(training_data)
    print(f"Critique-Guided Results: {cg_results}")
    
    # 测试预测
    test_state = np.random.randn(state_dim)
    test_action = np.random.randn(action_dim)
    reward, confidence = cg_trainer.predict_reward(test_state, test_action)
    print(f"Prediction: Reward={reward:.3f}, Confidence={confidence:.3f}")
    
    # 测试分层模型
    print("\nTesting Hierarchical Reward Model...")
    hr_trainer = create_reward_model_trainer(
        RewardModelType.HIERARCHICAL,
        num_epochs=30,
        hierarchy_levels=3
    )
    hr_results = hr_trainer.train(training_data)
    print(f"Hierarchical Results: {hr_results}")
    
    print("\nReward Model Training Module Test Completed!")