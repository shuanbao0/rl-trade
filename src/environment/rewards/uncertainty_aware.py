"""
基于不确定性感知的奖励函数

基于2024-2025年最新的贝叶斯神经网络和不确定性量化理论，
实现认知不确定性(Epistemic)和任意不确定性(Aleatoric)的估计，
结合条件风险价值(CVaR)优化，提供置信度校准的风险敏感决策。
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from collections import deque
import scipy.stats as stats
from .base_reward import BaseRewardScheme


class UncertaintyEstimator:
    """
    不确定性估计器 - 实现认知和任意不确定性量化
    
    基于Bayesian Deep Learning和Variational Inference理论
    """
    
    def __init__(self, ensemble_size: int = 5, dropout_samples: int = 100):
        """
        初始化不确定性估计器
        
        Args:
            ensemble_size: 集成模型数量
            dropout_samples: Monte Carlo Dropout采样次数
        """
        self.ensemble_size = ensemble_size
        self.dropout_samples = dropout_samples
        self.model_predictions = []
        self.prediction_history = deque(maxlen=1000)
        
    def estimate_epistemic_uncertainty(self, predictions: List[float]) -> float:
        """
        估计认知不确定性(Epistemic Uncertainty) - 模型不确定性
        
        认知不确定性反映了模型对现象理解的不确定性，
        可以通过增加训练数据或改进模型来减少。
        
        Args:
            predictions: 多个模型或采样的预测结果
            
        Returns:
            float: 认知不确定性值
        """
        if len(predictions) < 2:
            return 0.0
            
        # 使用方差作为认知不确定性的度量
        epistemic = np.var(predictions)
        
        # 归一化到[0, 1]区间
        epistemic_normalized = min(epistemic / (1.0 + epistemic), 0.95)
        
        return float(epistemic_normalized)
    
    def estimate_aleatoric_uncertainty(self, prediction: float, 
                                     historical_volatility: float) -> float:
        """
        估计任意不确定性(Aleatoric Uncertainty) - 数据固有噪声
        
        任意不确定性反映了数据中的固有随机性和噪声，
        无法通过增加数据量来减少。
        
        Args:
            prediction: 当前预测值
            historical_volatility: 历史波动率
            
        Returns:
            float: 任意不确定性值
        """
        # 基于历史波动率估计数据噪声
        base_aleatoric = historical_volatility
        
        # 考虑预测值的大小对不确定性的影响
        magnitude_factor = min(abs(prediction) * 0.1, 0.5)
        
        aleatoric = base_aleatoric + magnitude_factor
        
        # 归一化到[0, 1]区间
        aleatoric_normalized = min(aleatoric, 0.95)
        
        return float(aleatoric_normalized)
    
    def get_confidence_weight(self, epistemic: float, aleatoric: float) -> float:
        """
        计算置信度权重
        
        Args:
            epistemic: 认知不确定性
            aleatoric: 任意不确定性
            
        Returns:
            float: 置信度权重 [0, 1]
        """
        total_uncertainty = epistemic + aleatoric
        confidence = 1.0 / (1.0 + total_uncertainty)
        
        return float(confidence)


class CVaROptimizer:
    """
    条件风险价值(CVaR)优化器
    
    实现风险敏感的奖励优化，专注于尾部风险控制
    """
    
    def __init__(self, alpha: float = 0.05, window_size: int = 100):
        """
        初始化CVaR优化器
        
        Args:
            alpha: 置信水平 (默认5%，即95%置信度)
            window_size: 计算窗口大小
        """
        self.alpha = alpha
        self.window_size = window_size
        self.returns_history = deque(maxlen=window_size)
        
    def calculate_var(self, returns: List[float], alpha: float) -> float:
        """
        计算风险价值(VaR)
        
        Args:
            returns: 收益率历史
            alpha: 置信水平
            
        Returns:
            float: VaR值
        """
        if len(returns) < 10:
            return 0.0
            
        returns_array = np.array(returns)
        var = np.percentile(returns_array, alpha * 100)
        
        return float(var)
    
    def calculate_cvar(self, returns: List[float], alpha: float) -> float:
        """
        计算条件风险价值(CVaR)
        
        CVaR = E[Loss | Loss >= VaR]
        
        Args:
            returns: 收益率历史
            alpha: 置信水平
            
        Returns:
            float: CVaR值
        """
        if len(returns) < 10:
            return 0.0
            
        var = self.calculate_var(returns, alpha)
        returns_array = np.array(returns)
        
        # 找出超过VaR的损失
        tail_losses = returns_array[returns_array <= var]
        
        if len(tail_losses) == 0:
            return var
        
        cvar = np.mean(tail_losses)
        
        return float(cvar)
    
    def get_risk_penalty(self, current_return: float) -> float:
        """
        获取基于CVaR的风险惩罚
        
        Args:
            current_return: 当前收益率
            
        Returns:
            float: 风险惩罚值
        """
        self.returns_history.append(current_return)
        
        if len(self.returns_history) < 20:
            return 0.0
        
        returns_list = list(self.returns_history)
        cvar = self.calculate_cvar(returns_list, self.alpha)
        
        # 如果当前收益率落在CVaR尾部风险区域，给予额外惩罚
        if current_return <= cvar:
            risk_penalty = abs(cvar - current_return) * 10.0
            return risk_penalty
        
        return 0.0


class UncertaintyAwareReward(BaseRewardScheme):
    """
    不确定性感知奖励函数
    
    整合认知不确定性、任意不确定性估计和CVaR风险优化，
    提供置信度校准的风险敏感决策奖励。
    
    数学公式:
    R_adjusted = confidence_weight × R_base - λ × uncertainty_penalty - γ × cvar_penalty
    其中: confidence_weight = 1 / (1 + epistemic + aleatoric)
    """
    
    def __init__(self, 
                 uncertainty_lambda: float = 1.0,
                 cvar_gamma: float = 0.5,
                 cvar_alpha: float = 0.05,
                 ensemble_size: int = 5,
                 confidence_threshold: float = 0.3,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化不确定性感知奖励函数
        
        Args:
            uncertainty_lambda: 不确定性惩罚权重
            cvar_gamma: CVaR风险惩罚权重
            cvar_alpha: CVaR置信水平
            ensemble_size: 集成模型数量
            confidence_threshold: 最低置信度阈值
            initial_balance: 初始资金
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        self.uncertainty_lambda = uncertainty_lambda
        self.cvar_gamma = cvar_gamma
        self.confidence_threshold = confidence_threshold
        
        # 初始化组件
        self.uncertainty_estimator = UncertaintyEstimator(
            ensemble_size=ensemble_size,
            dropout_samples=100
        )
        self.cvar_optimizer = CVaROptimizer(
            alpha=cvar_alpha,
            window_size=100
        )
        
        # 状态追踪
        self.return_predictions = deque(maxlen=50)
        self.volatility_history = deque(maxlen=100)
        self.confidence_scores = deque(maxlen=100)
        
        logging.info(f"初始化UncertaintyAwareReward: λ={uncertainty_lambda}, γ={cvar_gamma}, α={cvar_alpha}")
    
    def _generate_ensemble_predictions(self, current_return: float) -> List[float]:
        """
        生成集成预测 (简化版本 - 实际应用中需要真实的模型集成)
        
        Args:
            current_return: 当前收益率
            
        Returns:
            List[float]: 集成预测结果
        """
        base_prediction = current_return
        predictions = []
        
        # 模拟不同模型的预测偏差
        noise_levels = [0.001, 0.002, 0.0015, 0.0025, 0.001]
        
        for i in range(self.uncertainty_estimator.ensemble_size):
            noise = np.random.normal(0, noise_levels[i])
            prediction = base_prediction + noise
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_historical_volatility(self) -> float:
        """
        计算历史波动率
        
        Returns:
            float: 历史波动率
        """
        if len(self.volatility_history) < 10:
            return 0.02  # 默认值
        
        volatility = np.std(list(self.volatility_history))
        return float(volatility)
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 不确定性感知奖励
        
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
            self.volatility_history.append(abs(step_return_pct))
            
            # 计算基础奖励
            base_reward = total_return_pct * 100.0  # 基于总收益率
            
            # === 1. 不确定性估计 ===
            
            # 生成集成预测
            predictions = self._generate_ensemble_predictions(step_return_pct)
            
            # 估计认知不确定性
            epistemic_uncertainty = self.uncertainty_estimator.estimate_epistemic_uncertainty(predictions)
            
            # 估计任意不确定性
            historical_vol = self._calculate_historical_volatility()
            aleatoric_uncertainty = self.uncertainty_estimator.estimate_aleatoric_uncertainty(
                step_return_pct, historical_vol
            )
            
            # 计算置信度权重
            confidence_weight = self.uncertainty_estimator.get_confidence_weight(
                epistemic_uncertainty, aleatoric_uncertainty
            )
            
            # === 2. CVaR风险优化 ===
            
            # 计算CVaR风险惩罚
            cvar_penalty = self.cvar_optimizer.get_risk_penalty(step_return_pct)
            
            # === 3. 综合奖励计算 ===
            
            # 不确定性惩罚
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            uncertainty_penalty = total_uncertainty * self.uncertainty_lambda
            
            # 最终奖励
            adjusted_reward = (confidence_weight * base_reward - 
                             uncertainty_penalty - 
                             self.cvar_gamma * cvar_penalty)
            
            # 置信度阈值过滤
            if confidence_weight < self.confidence_threshold:
                adjusted_reward *= 0.5  # 低置信度时减半奖励
            
            # 记录状态
            self.confidence_scores.append(confidence_weight)
            self.previous_value = portfolio_value
            self.step_count += 1
            
            return float(adjusted_reward)
            
        except Exception as e:
            logging.error(f"UncertaintyAware奖励计算异常: {e}")
            return 0.0
    
    def reward(self, env) -> float:
        """
        计算不确定性感知奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 奖励值
        """
        try:
            # 获取基础状态
            state = self.update_state(env)
            current_value = state['current_value']
            step_return_pct = state['step_return_pct']
            total_return_pct = state['total_return_pct']
            
            # 第一步初始化
            if state['step_count'] == 1:
                logging.info(f"[UncertaintyAware] 初始化: ${current_value:.2f}")
                return 0.0
            
            # 更新历史数据
            self.volatility_history.append(abs(step_return_pct))
            
            # 计算基础奖励
            base_reward = total_return_pct * 100.0  # 基于总收益率
            
            # === 1. 不确定性估计 ===
            
            # 生成集成预测
            predictions = self._generate_ensemble_predictions(step_return_pct)
            
            # 估计认知不确定性
            epistemic_uncertainty = self.uncertainty_estimator.estimate_epistemic_uncertainty(predictions)
            
            # 估计任意不确定性
            historical_vol = self._calculate_historical_volatility()
            aleatoric_uncertainty = self.uncertainty_estimator.estimate_aleatoric_uncertainty(
                step_return_pct, historical_vol
            )
            
            # 计算置信度权重
            confidence_weight = self.uncertainty_estimator.get_confidence_weight(
                epistemic_uncertainty, aleatoric_uncertainty
            )
            
            # === 2. CVaR风险优化 ===
            
            # 计算CVaR风险惩罚
            cvar_penalty = self.cvar_optimizer.get_risk_penalty(step_return_pct)
            
            # === 3. 综合奖励计算 ===
            
            # 不确定性惩罚
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            uncertainty_penalty = total_uncertainty * self.uncertainty_lambda
            
            # 最终奖励
            adjusted_reward = (confidence_weight * base_reward - 
                             uncertainty_penalty - 
                             self.cvar_gamma * cvar_penalty)
            
            # 置信度阈值过滤
            if confidence_weight < self.confidence_threshold:
                adjusted_reward *= 0.5  # 低置信度时减半奖励
            
            # 记录状态
            self.confidence_scores.append(confidence_weight)
            
            # 详细日志
            if state['step_count'] % 50 == 0 or abs(adjusted_reward) > 5.0:
                logging.info(
                    f"[UncertaintyAware] 步骤{state['step_count']}: "
                    f"基础奖励{base_reward:.3f} × 置信度{confidence_weight:.3f} "
                    f"- 不确定性惩罚{uncertainty_penalty:.3f} - CVaR惩罚{cvar_penalty:.3f} "
                    f"= {adjusted_reward:.3f} "
                    f"(认知:{epistemic_uncertainty:.3f}, 任意:{aleatoric_uncertainty:.3f})"
                )
            
            return float(adjusted_reward)
            
        except Exception as e:
            logging.error(f"UncertaintyAware奖励计算异常: {e}")
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
            
            # 简化的不确定性感知奖励计算
            predictions = self._generate_ensemble_predictions(step_return_pct)
            epistemic = self.uncertainty_estimator.estimate_epistemic_uncertainty(predictions)
            
            historical_vol = self._calculate_historical_volatility()
            aleatoric = self.uncertainty_estimator.estimate_aleatoric_uncertainty(
                step_return_pct, historical_vol
            )
            
            confidence = self.uncertainty_estimator.get_confidence_weight(epistemic, aleatoric)
            
            # 基础奖励
            base_reward = total_return_pct * 100.0
            uncertainty_penalty = (epistemic + aleatoric) * self.uncertainty_lambda
            
            # 最终奖励
            reward = confidence * base_reward - uncertainty_penalty
            
            # 更新状态
            self.previous_value = current_value
            self.step_count += 1
            self.volatility_history.append(abs(step_return_pct))
            
            return float(reward)
            
        except Exception as e:
            logging.error(f"UncertaintyAware get_reward异常: {e}")
            return 0.0
    
    def reset(self) -> 'UncertaintyAwareReward':
        """
        重置奖励函数状态
        
        Returns:
            UncertaintyAwareReward: 返回self以支持链式调用
        """
        # 记录上一回合的最终统计
        if self.confidence_scores:
            avg_confidence = np.mean(list(self.confidence_scores))
            logging.info(f"[UncertaintyAware] 回合{self.episode_count}结束: "
                        f"平均置信度{avg_confidence:.3f}")
        
        # 调用父类reset
        super().reset()
        
        # 重置组件 (保留部分历史数据用于连续学习)
        # 只保留最近的数据
        if len(self.volatility_history) > 50:
            recent_volatility = list(self.volatility_history)[-25:]
            self.volatility_history.clear()
            self.volatility_history.extend(recent_volatility)
        
        return self
    
    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """
        获取不确定性相关指标
        
        Returns:
            Dict[str, float]: 不确定性指标字典
        """
        if not self.confidence_scores:
            return {
                'avg_confidence': 0.0,
                'confidence_std': 0.0,
                'low_confidence_ratio': 0.0
            }
        
        confidence_list = list(self.confidence_scores)
        low_confidence_count = sum(1 for c in confidence_list if c < self.confidence_threshold)
        
        return {
            'avg_confidence': float(np.mean(confidence_list)),
            'confidence_std': float(np.std(confidence_list)),
            'low_confidence_ratio': float(low_confidence_count / len(confidence_list)),
            'historical_volatility': self._calculate_historical_volatility()
        }
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'UncertaintyAwareReward',
            'description': '基于认知和任意不确定性量化的风险敏感奖励函数，结合CVaR优化',
            'category': 'uncertainty_quantification',
            'complexity': 'expert',
            'parameters': {
                'uncertainty_lambda': {
                    'type': 'float',
                    'default': 1.0,
                    'description': '不确定性惩罚权重'
                },
                'cvar_gamma': {
                    'type': 'float',
                    'default': 0.5,
                    'description': 'CVaR风险惩罚权重'
                },
                'cvar_alpha': {
                    'type': 'float',
                    'default': 0.05,
                    'description': 'CVaR置信水平'
                },
                'ensemble_size': {
                    'type': 'int',
                    'default': 5,
                    'description': '集成模型数量'
                },
                'confidence_threshold': {
                    'type': 'float',
                    'default': 0.3,
                    'description': '最低置信度阈值'
                }
            },
            'features': [
                '认知不确定性估计(Epistemic Uncertainty)',
                '任意不确定性估计(Aleatoric Uncertainty)',
                '置信度校准和权重调整',
                'CVaR条件风险价值优化',
                '风险敏感决策支持',
                '集成模型不确定性量化',
                '尾部风险控制'
            ],
            'mathematical_foundation': [
                'Bayesian Neural Networks',
                'Variational Inference',
                'Monte Carlo Dropout',
                'Conditional Value at Risk (CVaR)',
                'Epistemic vs Aleatoric Uncertainty Decomposition'
            ],
            'applications': [
                '高风险交易环境',
                '不确定性量化需求',
                '风险敏感决策',
                '置信度要求严格的场景',
                '尾部风险控制'
            ]
        }