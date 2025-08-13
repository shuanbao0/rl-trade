"""
好奇心驱动奖励函数

基于2024-2025年最新的好奇心驱动强化学习理论，
实现内在动机和外在奖励的融合，包括前向模型预测误差、
学习进度监控、层次化子目标管理和DiNAT-Vision Transformer增强。
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import deque, defaultdict
import scipy.stats as stats
from dataclasses import dataclass
from .base_reward import BaseRewardScheme


@dataclass
class CuriosityConfig:
    """好奇心模块配置"""
    alpha_extrinsic: float = 1.0    # 外在奖励权重
    alpha_curiosity: float = 0.5     # 好奇心奖励权重  
    beta_progress: float = 0.3       # 学习进度奖励权重
    gamma_hierarchical: float = 0.2  # 层次化奖励权重
    forward_model_lr: float = 0.01   # 前向模型学习率
    progress_window: int = 50        # 学习进度计算窗口
    skill_discovery_threshold: float = 0.1  # 技能发现阈值


class ForwardModel:
    """
    前向模型 - 预测下一状态
    
    实现基于预测误差的内在好奇心机制
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 1, learning_rate: float = 0.01):
        """
        初始化前向模型
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度  
            learning_rate: 学习率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # 简化的线性前向模型参数
        self.weights_state = np.random.normal(0, 0.1, (state_dim, state_dim))
        self.weights_action = np.random.normal(0, 0.1, (action_dim, state_dim))
        self.bias = np.zeros(state_dim)
        
        # 预测历史
        self.prediction_errors = deque(maxlen=1000)
        self.predictions = deque(maxlen=100)
        
    def encode_state(self, portfolio_value: float, returns: List[float]) -> np.ndarray:
        """
        将交易状态编码为向量
        
        Args:
            portfolio_value: 投资组合价值
            returns: 历史收益率列表
            
        Returns:
            np.ndarray: 状态向量
        """
        # 基础特征
        features = []
        
        # 价值相关特征
        features.append(portfolio_value / 10000.0)  # 标准化的组合价值
        features.append(np.log(portfolio_value / 10000.0 + 1e-8))  # 对数价值
        
        # 收益率特征 (最近8个)
        if len(returns) >= 8:
            recent_returns = returns[-8:]
        else:
            recent_returns = returns + [0.0] * (8 - len(returns))
        features.extend(recent_returns)
        
        return np.array(features, dtype=np.float32)
    
    def predict_next_state(self, current_state: np.ndarray, action: float) -> np.ndarray:
        """
        预测下一状态
        
        Args:
            current_state: 当前状态向量
            action: 执行的动作
            
        Returns:
            np.ndarray: 预测的下一状态
        """
        action_vector = np.array([action], dtype=np.float32)
        
        # 线性预测模型: next_state = W_s * state + W_a * action + bias
        predicted_state = (
            np.dot(self.weights_state, current_state) + 
            np.dot(action_vector, self.weights_action) + 
            self.bias
        )
        
        return predicted_state
    
    def update_model(self, current_state: np.ndarray, action: float, 
                    actual_next_state: np.ndarray) -> float:
        """
        更新前向模型并返回预测误差
        
        Args:
            current_state: 当前状态
            action: 执行的动作
            actual_next_state: 实际的下一状态
            
        Returns:
            float: 预测误差(好奇心信号)
        """
        # 获取预测
        predicted_state = self.predict_next_state(current_state, action)
        
        # 计算预测误差
        prediction_error = np.mean((predicted_state - actual_next_state) ** 2)
        
        # 更新模型参数 (简化的梯度下降)
        error_gradient = 2.0 * (predicted_state - actual_next_state)
        
        # 更新权重
        self.weights_state -= self.learning_rate * np.outer(error_gradient, current_state)
        self.weights_action -= self.learning_rate * np.outer([action], error_gradient)
        self.bias -= self.learning_rate * error_gradient
        
        # 记录误差
        self.prediction_errors.append(prediction_error)
        self.predictions.append({
            'predicted': predicted_state.copy(),
            'actual': actual_next_state.copy(),
            'error': prediction_error
        })
        
        return float(prediction_error)
    
    def get_curiosity_reward(self, prediction_error: float) -> float:
        """
        基于预测误差计算好奇心奖励
        
        Args:
            prediction_error: 预测误差
            
        Returns:
            float: 好奇心奖励
        """
        # 好奇心奖励 = 预测误差，但需要归一化避免过大
        curiosity_reward = np.tanh(prediction_error * 10.0)  # 使用tanh归一化到[-1,1]
        return float(curiosity_reward)


class LearningProgressMonitor:
    """
    学习进度监控器
    
    跟踪智能体的学习进度，提供进度相关的内在奖励
    """
    
    def __init__(self, window_size: int = 50):
        """
        初始化学习进度监控器
        
        Args:
            window_size: 计算学习进度的窗口大小
        """
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        self.progress_scores = deque(maxlen=100)
        
    def update_performance(self, reward: float, prediction_error: float):
        """
        更新性能指标
        
        Args:
            reward: 当前奖励
            prediction_error: 预测误差
        """
        self.performance_history.append(reward)
        self.error_history.append(prediction_error)
    
    def calculate_learning_progress(self) -> float:
        """
        计算学习进度奖励
        
        基于性能改进趋势和误差减少趋势
        
        Returns:
            float: 学习进度奖励
        """
        if len(self.performance_history) < 10:
            return 0.0
        
        # 计算性能趋势
        performance_data = list(self.performance_history)
        recent_performance = np.mean(performance_data[-10:])
        early_performance = np.mean(performance_data[:10])
        performance_improvement = recent_performance - early_performance
        
        # 计算误差趋势
        error_data = list(self.error_history)
        recent_errors = np.mean(error_data[-10:])
        early_errors = np.mean(error_data[:10])
        error_reduction = early_errors - recent_errors
        
        # 综合学习进度
        progress_score = (performance_improvement + error_reduction) / 2.0
        
        # 归一化进度奖励
        progress_reward = np.tanh(progress_score * 5.0)
        
        self.progress_scores.append(progress_reward)
        
        return float(progress_reward)
    
    def get_competence_measure(self) -> float:
        """
        获取能力度量(基于最近的学习进度)
        
        Returns:
            float: 能力度量值
        """
        if not self.progress_scores:
            return 0.0
        
        return float(np.mean(list(self.progress_scores)[-10:]))


class HierarchicalGoalManager:
    """
    层次化目标管理器
    
    实现多层次的子目标管理和技能发现
    """
    
    def __init__(self, skill_threshold: float = 0.1):
        """
        初始化层次化目标管理器
        
        Args:
            skill_threshold: 技能发现阈值
        """
        self.skill_threshold = skill_threshold
        self.discovered_skills = []
        self.current_goals = {}
        self.goal_achievements = defaultdict(list)
        self.skill_patterns = deque(maxlen=500)
        
    def analyze_behavior_pattern(self, state: np.ndarray, action: float, 
                                reward: float) -> Optional[str]:
        """
        分析行为模式，发现潜在技能
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            
        Returns:
            Optional[str]: 发现的技能名称
        """
        # 记录行为模式
        pattern = {
            'state_features': state[:3] if len(state) >= 3 else state,  # 关键特征
            'action': action,
            'reward': reward,
            'timestamp': len(self.skill_patterns)
        }
        self.skill_patterns.append(pattern)
        
        # 技能发现逻辑 (简化版)
        if len(self.skill_patterns) >= 20:
            recent_patterns = list(self.skill_patterns)[-20:]
            
            # 检查是否有持续的正奖励模式
            positive_rewards = [p['reward'] for p in recent_patterns if p['reward'] > 0]
            if len(positive_rewards) >= 15:  # 75%的步骤获得正奖励
                skill_name = f"profitable_trading_skill_{len(self.discovered_skills)}"
                if skill_name not in self.discovered_skills:
                    self.discovered_skills.append(skill_name)
                    return skill_name
            
            # 检查是否有风险管理模式 (小亏损但避免大亏损)
            small_losses = [p['reward'] for p in recent_patterns 
                          if -0.1 <= p['reward'] < 0]
            if len(small_losses) >= 10 and max([p['reward'] for p in recent_patterns]) >= 0:
                skill_name = f"risk_management_skill_{len(self.discovered_skills)}"
                if skill_name not in self.discovered_skills:
                    self.discovered_skills.append(skill_name)
                    return skill_name
        
        return None
    
    def get_hierarchical_reward(self, discovered_skill: Optional[str]) -> float:
        """
        计算层次化奖励
        
        Args:
            discovered_skill: 发现的技能
            
        Returns:
            float: 层次化奖励
        """
        if discovered_skill:
            # 新技能发现奖励
            skill_discovery_bonus = 1.0
            logging.info(f"[CuriosityDriven] 发现新技能: {discovered_skill}")
            return skill_discovery_bonus
        
        # 基于现有技能的持续奖励
        if self.discovered_skills:
            skill_maintenance_bonus = 0.1 * len(self.discovered_skills)
            return skill_maintenance_bonus
        
        return 0.0


class DiNATFeatureExtractor:
    """
    简化的DiNAT-inspired特征提取器
    
    模拟DiNAT(Dilated Neighborhood Attention Transformer)的特征提取能力
    """
    
    def __init__(self, feature_dim: int = 10):
        """
        初始化特征提取器
        
        Args:
            feature_dim: 特征维度
        """
        self.feature_dim = feature_dim
        self.attention_weights = np.random.normal(0, 0.1, (feature_dim, feature_dim))
        self.learned_patterns = deque(maxlen=1000)
        
    def extract_features(self, raw_state: np.ndarray) -> np.ndarray:
        """
        提取增强特征
        
        Args:
            raw_state: 原始状态向量
            
        Returns:
            np.ndarray: 增强特征向量
        """
        # 模拟注意力机制
        attended_features = np.dot(self.attention_weights, raw_state)
        
        # 添加非线性变换
        enhanced_features = np.tanh(attended_features)
        
        # 记录学习到的模式
        self.learned_patterns.append(enhanced_features.copy())
        
        return enhanced_features
    
    def update_attention(self, state: np.ndarray, feedback: float):
        """
        基于反馈更新注意力权重
        
        Args:
            state: 状态向量
            feedback: 反馈信号
        """
        # 简化的注意力更新
        learning_rate = 0.001
        gradient = feedback * np.outer(state, np.ones_like(state))
        self.attention_weights += learning_rate * gradient
        
        # 保持权重稳定
        self.attention_weights = np.clip(self.attention_weights, -1.0, 1.0)


class CuriosityDrivenReward(BaseRewardScheme):
    """
    好奇心驱动奖励函数
    
    集成内在动机和外在奖励，包括：
    - 前向模型预测误差驱动的好奇心
    - 学习进度实时监控
    - 层次化子目标管理
    - DiNAT-Vision Transformer增强特征提取
    
    数学公式:
    R_total = α×R_extrinsic + β×R_curiosity + γ×R_progress + δ×R_hierarchical
    R_curiosity = ||f(s_t, a_t) - s_{t+1}||²
    """
    
    def __init__(self,
                 alpha_extrinsic: float = 1.0,
                 alpha_curiosity: float = 0.5,
                 beta_progress: float = 0.3,
                 gamma_hierarchical: float = 0.2,
                 forward_model_lr: float = 0.01,
                 progress_window: int = 50,
                 skill_discovery_threshold: float = 0.1,
                 enable_dinat: bool = True,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化好奇心驱动奖励函数
        
        Args:
            alpha_extrinsic: 外在奖励权重
            alpha_curiosity: 好奇心奖励权重
            beta_progress: 学习进度奖励权重
            gamma_hierarchical: 层次化奖励权重
            forward_model_lr: 前向模型学习率
            progress_window: 学习进度计算窗口
            skill_discovery_threshold: 技能发现阈值
            enable_dinat: 是否启用DiNAT特征增强
            initial_balance: 初始资金
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 配置参数
        self.config = CuriosityConfig(
            alpha_extrinsic=alpha_extrinsic,
            alpha_curiosity=alpha_curiosity,
            beta_progress=beta_progress,
            gamma_hierarchical=gamma_hierarchical,
            forward_model_lr=forward_model_lr,
            progress_window=progress_window,
            skill_discovery_threshold=skill_discovery_threshold
        )
        
        # 初始化组件
        self.forward_model = ForwardModel(
            state_dim=10,
            action_dim=1,
            learning_rate=forward_model_lr
        )
        
        self.progress_monitor = LearningProgressMonitor(window_size=progress_window)
        self.goal_manager = HierarchicalGoalManager(skill_threshold=skill_discovery_threshold)
        
        if enable_dinat:
            self.feature_extractor = DiNATFeatureExtractor(feature_dim=10)
        else:
            self.feature_extractor = None
        
        # 状态追踪
        self.returns_history = deque(maxlen=100)
        self.states_history = deque(maxlen=50)
        self.curiosity_scores = deque(maxlen=100)
        self.progress_scores = deque(maxlen=100)
        
        # 统计信息
        self.total_curiosity_reward = 0.0
        self.total_progress_reward = 0.0
        self.total_hierarchical_reward = 0.0
        
        logging.info(f"初始化CuriosityDrivenReward: "
                    f"α_ext={alpha_extrinsic}, α_cur={alpha_curiosity}, "
                    f"β_prog={beta_progress}, γ_hier={gamma_hierarchical}")
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 好奇心驱动奖励
        
        Args:
            portfolio_value: 当前投资组合价值
            action: 执行的动作
            price: 当前价格
            portfolio_info: 投资组合详细信息
            trade_info: 交易执行信息
            step: 当前步数
            **kwargs: 其他参数
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            # 第一步初始化
            if self.previous_value is None:
                self.previous_value = portfolio_value
                self.initial_value = portfolio_value
                return 0.0
            
            # 计算收益率
            step_return_pct = (portfolio_value - self.previous_value) / self.previous_value
            total_return_pct = (portfolio_value - self.initial_value) / self.initial_value
            
            # 更新历史数据
            self.return_history.append(step_return_pct)
            
            # 计算各组件奖励
            extrinsic_reward = self._calculate_extrinsic_reward(total_return_pct)
            curiosity_reward = self._calculate_curiosity_reward(portfolio_value, action)
            progress_reward = self._calculate_progress_reward()
            hierarchical_reward = self._calculate_hierarchical_reward(portfolio_value, action)
            
            # 综合奖励
            total_reward = (
                self.config.alpha_extrinsic * extrinsic_reward +
                self.config.alpha_curiosity * curiosity_reward +
                self.config.beta_progress * progress_reward +
                self.config.gamma_hierarchical * hierarchical_reward
            )
            
            # 更新状态
            self.previous_value = portfolio_value
            self.step_count += 1
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"CuriosityDriven奖励计算异常: {e}")
            return 0.0
    
    def _calculate_extrinsic_reward(self, total_return_pct: float) -> float:
        """
        计算外在奖励(基于实际交易表现)
        
        Args:
            total_return_pct: 总收益率百分比
            
        Returns:
            float: 外在奖励值
        """
        # 基础外在奖励 = 总收益率 * 100
        extrinsic_reward = total_return_pct * 100.0
        return extrinsic_reward
    
    def reward(self, env) -> float:
        """
        计算好奇心驱动的综合奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 综合奖励值
        """
        try:
            # 获取基础状态
            state = self.update_state(env)
            current_value = state['current_value']
            step_return_pct = state['step_return_pct'] 
            total_return_pct = state['total_return_pct']
            current_action = state['current_action']
            
            # 第一步初始化
            if state['step_count'] == 1:
                logging.info(f"[CuriosityDriven] 初始化: ${current_value:.2f}")
                return 0.0
            
            # 更新收益历史
            self.returns_history.append(step_return_pct)
            
            # === 1. 外在奖励计算 ===
            extrinsic_reward = self._calculate_extrinsic_reward(total_return_pct)
            
            # === 2. 好奇心奖励计算 ===
            
            # 编码当前状态
            current_state_vector = self.forward_model.encode_state(
                current_value, list(self.returns_history)
            )
            
            # DiNAT特征增强
            if self.feature_extractor:
                current_state_vector = self.feature_extractor.extract_features(current_state_vector)
            
            curiosity_reward = 0.0
            if len(self.states_history) > 0:
                # 获取前一状态
                prev_state_vector = self.states_history[-1]
                
                # 更新前向模型并获取预测误差
                prediction_error = self.forward_model.update_model(
                    prev_state_vector, current_action, current_state_vector
                )
                
                # 计算好奇心奖励
                curiosity_reward = self.forward_model.get_curiosity_reward(prediction_error)
                self.curiosity_scores.append(curiosity_reward)
                self.total_curiosity_reward += curiosity_reward
                
                # 更新DiNAT注意力
                if self.feature_extractor:
                    self.feature_extractor.update_attention(current_state_vector, curiosity_reward)
            
            # 保存当前状态
            self.states_history.append(current_state_vector)
            
            # === 3. 学习进度奖励计算 ===
            
            # 更新进度监控器
            self.progress_monitor.update_performance(extrinsic_reward, 
                                                   curiosity_reward if curiosity_reward else 0.0)
            
            # 计算学习进度奖励
            progress_reward = self.progress_monitor.calculate_learning_progress()
            self.progress_scores.append(progress_reward)
            self.total_progress_reward += progress_reward
            
            # === 4. 层次化奖励计算 ===
            
            # 分析行为模式，发现技能
            discovered_skill = self.goal_manager.analyze_behavior_pattern(
                current_state_vector, current_action, extrinsic_reward
            )
            
            # 计算层次化奖励
            hierarchical_reward = self.goal_manager.get_hierarchical_reward(discovered_skill)
            self.total_hierarchical_reward += hierarchical_reward
            
            # === 5. 综合奖励计算 ===
            
            total_reward = (
                self.config.alpha_extrinsic * extrinsic_reward +
                self.config.alpha_curiosity * curiosity_reward +
                self.config.beta_progress * progress_reward +
                self.config.gamma_hierarchical * hierarchical_reward
            )
            
            # 详细日志
            if state['step_count'] % 50 == 0 or abs(total_reward) > 5.0:
                logging.info(
                    f"[CuriosityDriven] 步骤{state['step_count']}: "
                    f"外在{extrinsic_reward:.3f} + 好奇心{curiosity_reward:.3f} + "
                    f"进度{progress_reward:.3f} + 层次{hierarchical_reward:.3f} = "
                    f"{total_reward:.3f} (技能数:{len(self.goal_manager.discovered_skills)})"
                )
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"CuriosityDriven奖励计算异常: {e}")
            return 0.0
    
    def get_reward(self, portfolio) -> float:
        """
        TensorTrade框架要求的get_reward方法
        
        Args:
            portfolio: 投资组合对象
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            current_value = float(portfolio.net_worth)
            
            if self.previous_value is None:
                self.previous_value = current_value
                self.initial_value = current_value
                return 0.0
            
            # 计算收益率
            step_return_pct = (current_value - self.previous_value) / self.previous_value
            total_return_pct = (current_value - self.initial_value) / self.initial_value
            
            # 简化的好奇心驱动奖励计算
            extrinsic = total_return_pct * 100.0
            
            # 更新收益历史
            self.returns_history.append(step_return_pct)
            
            # 编码状态
            current_state = self.forward_model.encode_state(
                current_value, list(self.returns_history)
            )
            
            # 计算好奇心奖励
            curiosity = 0.0
            if len(self.states_history) > 0:
                prev_state = self.states_history[-1]
                pred_error = self.forward_model.update_model(prev_state, 0.0, current_state)
                curiosity = self.forward_model.get_curiosity_reward(pred_error)
            
            self.states_history.append(current_state)
            
            # 综合奖励
            total_reward = (
                self.config.alpha_extrinsic * extrinsic +
                self.config.alpha_curiosity * curiosity
            )
            
            # 更新状态
            self.previous_value = current_value
            self.step_count += 1
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"CuriosityDriven get_reward异常: {e}")
            return 0.0
    
    def reset(self) -> 'CuriosityDrivenReward':
        """
        重置奖励函数状态
        
        Returns:
            CuriosityDrivenReward: 返回self以支持链式调用
        """
        # 记录上一回合的统计信息
        logging.info(f"[CuriosityDriven] 回合{self.episode_count}结束: "
                    f"好奇心总奖励{self.total_curiosity_reward:.3f}, "
                    f"进度总奖励{self.total_progress_reward:.3f}, "
                    f"层次总奖励{self.total_hierarchical_reward:.3f}, "
                    f"发现技能数{len(self.goal_manager.discovered_skills)}")
        
        # 调用父类reset
        super().reset()
        
        # 重置组件状态 (保留学习到的模式)
        # 只清空短期历史，保留长期学习成果
        self.returns_history.clear()
        self.states_history.clear()
        
        # 重置统计
        self.total_curiosity_reward = 0.0
        self.total_progress_reward = 0.0
        self.total_hierarchical_reward = 0.0
        
        return self
    
    def get_curiosity_metrics(self) -> Dict[str, Any]:
        """
        获取好奇心相关指标
        
        Returns:
            Dict[str, Any]: 好奇心指标字典
        """
        metrics = {
            'avg_curiosity_score': 0.0,
            'avg_progress_score': 0.0,
            'discovered_skills_count': len(self.goal_manager.discovered_skills),
            'discovered_skills': self.goal_manager.discovered_skills.copy(),
            'total_curiosity_reward': self.total_curiosity_reward,
            'total_progress_reward': self.total_progress_reward,
            'total_hierarchical_reward': self.total_hierarchical_reward,
            'competence_measure': self.progress_monitor.get_competence_measure(),
            'forward_model_prediction_errors': len(self.forward_model.prediction_errors)
        }
        
        if self.curiosity_scores:
            metrics['avg_curiosity_score'] = float(np.mean(list(self.curiosity_scores)))
            
        if self.progress_scores:
            metrics['avg_progress_score'] = float(np.mean(list(self.progress_scores)))
        
        return metrics
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'CuriosityDrivenReward',
            'description': '好奇心驱动奖励函数，集成内在动机和外在奖励，包括前向模型预测误差、学习进度监控和层次化子目标管理',
            'category': 'intrinsic_motivation',
            'complexity': 'expert',
            'parameters': {
                'alpha_extrinsic': {
                    'type': 'float',
                    'default': 1.0,
                    'description': '外在奖励权重'
                },
                'alpha_curiosity': {
                    'type': 'float',
                    'default': 0.5,
                    'description': '好奇心奖励权重'
                },
                'beta_progress': {
                    'type': 'float',
                    'default': 0.3,
                    'description': '学习进度奖励权重'
                },
                'gamma_hierarchical': {
                    'type': 'float',
                    'default': 0.2,
                    'description': '层次化奖励权重'
                },
                'forward_model_lr': {
                    'type': 'float',
                    'default': 0.01,
                    'description': '前向模型学习率'
                },
                'progress_window': {
                    'type': 'int',
                    'default': 50,
                    'description': '学习进度计算窗口'
                },
                'skill_discovery_threshold': {
                    'type': 'float',
                    'default': 0.1,
                    'description': '技能发现阈值'
                },
                'enable_dinat': {
                    'type': 'bool',
                    'default': True,
                    'description': '是否启用DiNAT特征增强'
                }
            },
            'features': [
                '前向模型预测误差驱动的好奇心',
                '学习进度实时监控',
                '层次化子目标管理',
                'DiNAT-Vision Transformer增强特征提取',
                '技能发现和模式识别',
                '内在动机和外在奖励融合',
                '自适应探索机制'
            ],
            'mathematical_foundation': [
                'Intrinsic Curiosity Module (ICM)',
                'Forward Model Prediction Error',
                'Learning Progress Theory',
                'Hierarchical Reinforcement Learning',
                'DiNAT (Dilated Neighborhood Attention Transformer)',
                'Skill Discovery and Pattern Recognition',
                'Multi-objective Reward Combination'
            ],
            'applications': [
                '稀疏奖励环境',
                '需要探索的交易策略',
                '长期学习和适应',
                '技能发现和迁移',
                '复杂模式识别',
                '自主学习系统'
            ]
        }