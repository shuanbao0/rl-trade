#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
偏好学习模块

实现Bradley-Terry偏好模型和多专家偏好融合机制
用于从人类专家反馈中学习奖励函数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
from datetime import datetime, timedelta
import scipy.optimize
from collections import defaultdict, Counter
import pandas as pd
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


class PreferenceLearningMethod(Enum):
    """偏好学习方法枚举"""
    BRADLEY_TERRY = "bradley_terry"
    GAUSSIAN_PROCESS = "gaussian_process"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class PreferenceTrainingConfig:
    """偏好学习训练配置"""
    method: PreferenceLearningMethod = PreferenceLearningMethod.BRADLEY_TERRY
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    regularization_weight: float = 0.01
    confidence_threshold: float = 0.5
    
    # Bradley-Terry特定参数
    bt_regularization: float = 0.1
    bt_convergence_tol: float = 1e-6
    bt_max_iterations: int = 1000
    
    # 神经网络特定参数
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout_rate: float = 0.2
    
    # 专家加权参数
    expert_weight_decay: float = 0.95
    min_expert_weight: float = 0.1
    reliability_weight: float = 0.8


@dataclass
class PreferenceData:
    """偏好数据结构"""
    expert_id: str
    scenario_a_features: np.ndarray
    scenario_b_features: np.ndarray
    preference: int  # 0: A preferred, 1: B preferred
    confidence: float
    timestamp: datetime
    expert_reliability: float = 1.0
    context_features: Optional[np.ndarray] = None


class PreferenceDataset(Dataset):
    """PyTorch偏好数据集"""
    
    def __init__(self, preference_data: List[PreferenceData]):
        self.data = preference_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 组合特征
        feature_a = torch.FloatTensor(item.scenario_a_features)
        feature_b = torch.FloatTensor(item.scenario_b_features)
        
        # 偏好标签和权重
        preference = torch.LongTensor([item.preference])
        confidence = torch.FloatTensor([item.confidence])
        expert_weight = torch.FloatTensor([item.expert_reliability])
        
        return {
            'feature_a': feature_a,
            'feature_b': feature_b,
            'preference': preference,
            'confidence': confidence,
            'expert_weight': expert_weight
        }


class BradleyTerryPreferenceModel:
    """Bradley-Terry偏好模型
    
    实现经典的Bradley-Terry模型用于从成对比较中学习偏好
    P(A > B) = exp(θ_A) / (exp(θ_A) + exp(θ_B))
    """
    
    def __init__(self, config: PreferenceTrainingConfig):
        self.config = config
        self.feature_weights = None
        self.fitted = False
        
    def fit(self, preference_data: List[PreferenceData]) -> Dict[str, Any]:
        """训练Bradley-Terry模型"""
        logger.info("Training Bradley-Terry preference model...")
        
        # 准备数据
        X_diff, y, weights = self._prepare_data(preference_data)
        
        if len(X_diff) == 0:
            raise ValueError("No valid preference data provided")
        
        # 使用scipy优化器求解
        def objective(w):
            logits = X_diff @ w
            # Bradley-Terry对数似然
            loss = -np.sum(weights * (y * logits - np.log(1 + np.exp(logits))))
            # L2正则化
            loss += self.config.bt_regularization * np.sum(w**2)
            return loss
        
        def gradient(w):
            logits = X_diff @ w
            probs = 1 / (1 + np.exp(-logits))
            grad = -X_diff.T @ (weights * (y - probs))
            grad += 2 * self.config.bt_regularization * w
            return grad
        
        # 初始化权重
        n_features = X_diff.shape[1]
        w0 = np.random.normal(0, 0.1, n_features)
        
        # 优化
        result = scipy.optimize.minimize(
            objective, w0, method='L-BFGS-B', 
            jac=gradient,
            options={'maxiter': self.config.bt_max_iterations}
        )
        
        self.feature_weights = result.x
        self.fitted = True
        
        # 计算训练统计
        final_loss = result.fun
        accuracy = self._compute_accuracy(X_diff, y, weights)
        
        logger.info(f"Bradley-Terry training completed. Loss: {final_loss:.4f}, Accuracy: {accuracy:.3f}")
        
        return {
            'final_loss': final_loss,
            'accuracy': accuracy,
            'convergence': result.success,
            'iterations': result.nit,
            'feature_weights': self.feature_weights.tolist()
        }
    
    def predict_preference(self, feature_a: np.ndarray, feature_b: np.ndarray) -> Tuple[float, float]:
        """预测偏好概率
        
        Returns:
            Tuple[float, float]: (P(A > B), confidence)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        feature_diff = feature_a - feature_b
        logit = feature_diff @ self.feature_weights
        prob_a = 1 / (1 + np.exp(-logit))
        
        # 置信度基于预测的确定性
        confidence = abs(prob_a - 0.5) * 2
        
        return prob_a, confidence
    
    def _prepare_data(self, preference_data: List[PreferenceData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备训练数据"""
        X_diff = []
        y = []
        weights = []
        
        for item in preference_data:
            if item.confidence < self.config.confidence_threshold:
                continue
            
            # 特征差异
            feature_diff = item.scenario_a_features - item.scenario_b_features
            X_diff.append(feature_diff)
            
            # 标签 (0: A preferred, 1: B preferred)
            y.append(1 - item.preference)  # 转换为Bradley-Terry格式
            
            # 权重 (置信度 × 专家可靠性)
            weight = item.confidence * item.expert_reliability
            weights.append(weight)
        
        return np.array(X_diff), np.array(y), np.array(weights)
    
    def _compute_accuracy(self, X_diff: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """计算加权准确率"""
        logits = X_diff @ self.feature_weights
        probs = 1 / (1 + np.exp(-logits))
        predictions = (probs > 0.5).astype(int)
        
        weighted_correct = np.sum(weights * (predictions == y))
        total_weight = np.sum(weights)
        
        return weighted_correct / total_weight if total_weight > 0 else 0.0


class NeuralPreferenceModel(nn.Module):
    """神经网络偏好模型
    
    使用深度神经网络学习复杂的偏好模式
    """
    
    def __init__(self, input_dim: int, config: PreferenceTrainingConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层 - 单个奖励值
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(features)
    
    def predict_preference(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        """预测偏好概率"""
        reward_a = self.forward(feature_a)
        reward_b = self.forward(feature_b)
        
        # Bradley-Terry概率
        logit_diff = reward_a - reward_b
        prob_a = torch.sigmoid(logit_diff)
        
        return prob_a


class PreferenceLearningTrainer:
    """偏好学习训练器
    
    统一的训练接口，支持多种偏好学习方法
    """
    
    def __init__(self, config: PreferenceTrainingConfig):
        self.config = config
        self.model = None
        self.training_history = []
        
    def train(self, preference_data: List[PreferenceData]) -> Dict[str, Any]:
        """训练偏好模型"""
        logger.info(f"Starting preference learning with method: {self.config.method.value}")
        
        if self.config.method == PreferenceLearningMethod.BRADLEY_TERRY:
            return self._train_bradley_terry(preference_data)
        elif self.config.method == PreferenceLearningMethod.NEURAL_NETWORK:
            return self._train_neural_network(preference_data)
        else:
            raise NotImplementedError(f"Method {self.config.method.value} not implemented")
    
    def predict_preference(self, feature_a: np.ndarray, feature_b: np.ndarray) -> Tuple[float, float]:
        """预测偏好"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_preference(feature_a, feature_b)
    
    def _train_bradley_terry(self, preference_data: List[PreferenceData]) -> Dict[str, Any]:
        """训练Bradley-Terry模型"""
        self.model = BradleyTerryPreferenceModel(self.config)
        results = self.model.fit(preference_data)
        
        self.training_history.append({
            'method': 'bradley_terry',
            'timestamp': datetime.now(),
            'results': results,
            'num_samples': len(preference_data)
        })
        
        return results
    
    def _train_neural_network(self, preference_data: List[PreferenceData]) -> Dict[str, Any]:
        """训练神经网络模型"""
        if len(preference_data) == 0:
            raise ValueError("No preference data provided")
        
        # 确定输入维度
        input_dim = len(preference_data[0].scenario_a_features)
        
        # 创建模型
        self.model = NeuralPreferenceModel(input_dim, self.config)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss(reduction='none')
        
        # 创建数据加载器
        dataset = PreferenceDataset(preference_data)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # 训练循环
        training_losses = []
        
        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # 前向传播
                prob_a = self.model.predict_preference(
                    batch['feature_a'], 
                    batch['feature_b']
                )
                
                # 计算损失
                target = (1 - batch['preference']).float()  # 转换标签
                loss = criterion(prob_a.squeeze(), target.squeeze())
                
                # 加权损失
                weights = batch['confidence'] * batch['expert_weight']
                weighted_loss = (loss * weights.squeeze()).mean()
                
                # 反向传播
                weighted_loss.backward()
                optimizer.step()
                
                epoch_loss += weighted_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            training_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}")
        
        # 计算最终准确率
        self.model.eval()
        accuracy = self._compute_neural_accuracy(dataloader)
        
        results = {
            'final_loss': training_losses[-1],
            'accuracy': accuracy,
            'training_losses': training_losses,
            'num_epochs': self.config.num_epochs
        }
        
        self.training_history.append({
            'method': 'neural_network',
            'timestamp': datetime.now(),
            'results': results,
            'num_samples': len(preference_data)
        })
        
        logger.info(f"Neural preference model training completed. Final loss: {results['final_loss']:.4f}, Accuracy: {accuracy:.3f}")
        
        return results
    
    def _compute_neural_accuracy(self, dataloader: DataLoader) -> float:
        """计算神经网络模型准确率"""
        correct = 0
        total = 0
        total_weight = 0
        
        with torch.no_grad():
            for batch in dataloader:
                prob_a = self.model.predict_preference(
                    batch['feature_a'], 
                    batch['feature_b']
                )
                
                predictions = (prob_a > 0.5).long().squeeze()
                targets = (1 - batch['preference']).long().squeeze()
                weights = (batch['confidence'] * batch['expert_weight']).squeeze()
                
                correct += (weights * (predictions == targets)).sum().item()
                total_weight += weights.sum().item()
        
        return correct / total_weight if total_weight > 0 else 0.0


class MultiExpertPreferenceFusion:
    """多专家偏好融合器
    
    融合多个专家的偏好数据，处理专家间的分歧和权重分配
    """
    
    def __init__(self, config: PreferenceTrainingConfig):
        self.config = config
        self.expert_weights = defaultdict(float)
        self.expert_reliability = defaultdict(float)
        self.consensus_threshold = 0.7
        
    def update_expert_weights(self, expert_feedback_history: Dict[str, List[Dict]]):
        """更新专家权重基于历史表现"""
        logger.info("Updating expert weights based on historical performance...")
        
        for expert_id, history in expert_feedback_history.items():
            if len(history) == 0:
                continue
            
            # 计算专家可靠性指标
            confidences = [h.get('confidence', 0.5) for h in history]
            consistency = self._calculate_consistency(history)
            recency_weight = self._calculate_recency_weight(history)
            
            # 综合评分
            reliability = (
                np.mean(confidences) * 0.4 +
                consistency * 0.4 +
                recency_weight * 0.2
            )
            
            # 应用衰减
            current_weight = self.expert_weights.get(expert_id, 1.0)
            new_weight = (
                current_weight * self.config.expert_weight_decay +
                reliability * (1 - self.config.expert_weight_decay)
            )
            
            # 应用最小权重约束
            self.expert_weights[expert_id] = max(new_weight, self.config.min_expert_weight)
            self.expert_reliability[expert_id] = reliability
            
        logger.info(f"Updated weights for {len(self.expert_weights)} experts")
    
    def fuse_preferences(self, preference_data: List[PreferenceData]) -> List[PreferenceData]:
        """融合多专家偏好数据"""
        logger.info("Fusing multi-expert preference data...")
        
        # 按场景对进行分组
        scenario_groups = defaultdict(list)
        
        for pref in preference_data:
            # 创建场景对的键
            scenario_key = self._create_scenario_key(
                pref.scenario_a_features, 
                pref.scenario_b_features
            )
            scenario_groups[scenario_key].append(pref)
        
        fused_data = []
        
        for scenario_key, group_prefs in scenario_groups.items():
            if len(group_prefs) == 1:
                # 单个专家，直接使用
                fused_data.append(group_prefs[0])
            else:
                # 多个专家，进行融合
                fused_pref = self._fuse_group_preferences(group_prefs)
                if fused_pref is not None:
                    fused_data.append(fused_pref)
        
        logger.info(f"Fused {len(preference_data)} preferences into {len(fused_data)} consensus preferences")
        return fused_data
    
    def _calculate_consistency(self, history: List[Dict]) -> float:
        """计算专家一致性"""
        if len(history) < 2:
            return 1.0
        
        # 基于置信度变异系数
        confidences = [h.get('confidence', 0.5) for h in history]
        if np.std(confidences) == 0:
            return 1.0
        
        cv = np.std(confidences) / np.mean(confidences)
        consistency = max(0, 1 - cv)
        
        return consistency
    
    def _calculate_recency_weight(self, history: List[Dict]) -> float:
        """计算时间衰减权重"""
        if len(history) == 0:
            return 0.0
        
        # 最近的反馈权重更高
        recent_feedbacks = [
            h for h in history 
            if datetime.now() - h.get('timestamp', datetime.now()) < timedelta(days=30)
        ]
        
        recency_ratio = len(recent_feedbacks) / len(history)
        return recency_ratio
    
    def _create_scenario_key(self, features_a: np.ndarray, features_b: np.ndarray) -> str:
        """创建场景对的唯一键"""
        # 使用特征向量的哈希
        combined = np.concatenate([features_a, features_b])
        return str(hash(combined.tobytes()))
    
    def _fuse_group_preferences(self, group_prefs: List[PreferenceData]) -> Optional[PreferenceData]:
        """融合同一场景对的多专家偏好"""
        if len(group_prefs) == 0:
            return None
        
        # 收集专家意见
        expert_votes = []
        expert_weights = []
        expert_confidences = []
        
        for pref in group_prefs:
            expert_weight = self.expert_weights.get(pref.expert_id, 1.0)
            
            expert_votes.append(pref.preference)
            expert_weights.append(expert_weight)
            expert_confidences.append(pref.confidence)
        
        expert_weights = np.array(expert_weights)
        expert_votes = np.array(expert_votes)
        expert_confidences = np.array(expert_confidences)
        
        # 计算加权投票
        weighted_vote = np.average(expert_votes, weights=expert_weights)
        final_preference = 1 if weighted_vote > 0.5 else 0
        
        # 计算共识强度
        consensus_strength = self._calculate_consensus_strength(
            expert_votes, expert_weights
        )
        
        # 只有达到共识阈值才返回融合结果
        if consensus_strength < self.consensus_threshold:
            return None
        
        # 融合置信度
        fused_confidence = np.average(
            expert_confidences, 
            weights=expert_weights
        ) * consensus_strength
        
        # 创建融合的偏好数据
        base_pref = group_prefs[0]
        fused_pref = PreferenceData(
            expert_id="multi_expert_consensus",
            scenario_a_features=base_pref.scenario_a_features,
            scenario_b_features=base_pref.scenario_b_features,
            preference=final_preference,
            confidence=fused_confidence,
            timestamp=datetime.now(),
            expert_reliability=consensus_strength
        )
        
        return fused_pref
    
    def _calculate_consensus_strength(self, votes: np.ndarray, weights: np.ndarray) -> float:
        """计算专家共识强度"""
        # 加权方差越小，共识越强
        weighted_mean = np.average(votes, weights=weights)
        weighted_variance = np.average((votes - weighted_mean)**2, weights=weights)
        
        # 转换为0-1的共识强度
        consensus_strength = max(0, 1 - 4 * weighted_variance)  # 4倍方差归一化
        
        return consensus_strength


def create_preference_learner(
    method: PreferenceLearningMethod = PreferenceLearningMethod.BRADLEY_TERRY,
    **kwargs
) -> PreferenceLearningTrainer:
    """工厂函数创建偏好学习器"""
    
    config = PreferenceTrainingConfig(method=method, **kwargs)
    return PreferenceLearningTrainer(config)


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建模拟偏好数据
    np.random.seed(42)
    
    sample_data = []
    for i in range(100):
        features_a = np.random.randn(10)
        features_b = np.random.randn(10)
        
        # 模拟真实偏好（基于某种模式）
        true_preference = 1 if np.sum(features_a) > np.sum(features_b) else 0
        noise_preference = true_preference if np.random.random() > 0.1 else 1 - true_preference
        
        sample_data.append(PreferenceData(
            expert_id=f"expert_{i % 5}",
            scenario_a_features=features_a,
            scenario_b_features=features_b,
            preference=noise_preference,
            confidence=np.random.uniform(0.6, 0.95),
            timestamp=datetime.now(),
            expert_reliability=np.random.uniform(0.7, 1.0)
        ))
    
    # 测试Bradley-Terry模型
    print("Testing Bradley-Terry Preference Model...")
    bt_trainer = create_preference_learner(PreferenceLearningMethod.BRADLEY_TERRY)
    bt_results = bt_trainer.train(sample_data)
    print(f"Bradley-Terry Results: {bt_results}")
    
    # 测试神经网络模型
    print("\nTesting Neural Preference Model...")
    nn_trainer = create_preference_learner(
        PreferenceLearningMethod.NEURAL_NETWORK,
        num_epochs=50,
        batch_size=16
    )
    nn_results = nn_trainer.train(sample_data)
    print(f"Neural Network Results: {nn_results}")
    
    # 测试多专家融合
    print("\nTesting Multi-Expert Fusion...")
    fusion = MultiExpertPreferenceFusion(PreferenceTrainingConfig())
    
    # 模拟专家历史
    expert_history = {
        f"expert_{i}": [
            {'confidence': np.random.uniform(0.6, 0.9), 'timestamp': datetime.now()}
            for _ in range(np.random.randint(5, 20))
        ]
        for i in range(5)
    }
    
    fusion.update_expert_weights(expert_history)
    fused_data = fusion.fuse_preferences(sample_data)
    print(f"Fused {len(sample_data)} preferences into {len(fused_data)} consensus preferences")
    
    print("\nPreference Learning Module Test Completed!")