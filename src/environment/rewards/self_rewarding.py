"""
自我奖励函数 (Self-Rewarding Reward)

基于Meta AI 2024年Self-Rewarding Language Models理论，
实现自我评判、迭代改进和直接偏好优化(DPO)的自适应奖励机制。

参考文献:
- Self-Rewarding Language Models (Meta AI, January 2024)
- Meta-Rewarding Language Models (Meta AI, July 2024)
- Direct Preference Optimization (Rafailov et al., 2024)
"""

import numpy as np
import logging
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
import scipy.stats as stats
from enum import Enum
from .base_reward import BaseRewardScheme


class JudgeRole(Enum):
    """评判角色定义"""
    ACTOR = "actor"           # 行动者 - 执行交易决策
    JUDGE = "judge"           # 评判者 - 评估交易质量
    META_JUDGE = "meta_judge" # 元评判者 - 评估评判质量


@dataclass
class SelfRewardConfig:
    """自我奖励配置"""
    # 基础权重
    base_reward_weight: float = 0.6          # 基础奖励权重
    self_evaluation_weight: float = 0.3      # 自我评估权重
    meta_evaluation_weight: float = 0.1      # 元评估权重
    
    # DPO参数
    dpo_beta: float = 0.1                   # DPO温度参数
    preference_strength: float = 1.0        # 偏好强度
    
    # 迭代学习参数
    learning_rate: float = 0.01             # 学习率
    evaluation_window: int = 20             # 评估窗口大小
    improvement_threshold: float = 0.05      # 改进阈值
    
    # 安全性参数
    bias_detection_threshold: float = 0.3    # 偏差检测阈值
    length_bias_penalty: float = 0.1        # 长度偏差惩罚
    overconfidence_penalty: float = 0.2     # 过度自信惩罚


@dataclass
class TradingEvaluation:
    """交易评估结果"""
    score: float                            # 评估分数 [0-10]
    reasoning: str                          # 评估推理
    confidence: float                       # 置信度 [0-1]
    criteria_scores: Dict[str, float] = field(default_factory=dict)  # 各项评估标准分数
    timestamp: float = 0.0                  # 时间戳
    role: JudgeRole = JudgeRole.JUDGE      # 评判角色


class LLMAsJudge:
    """
    LLM-as-a-Judge评估模块
    
    模拟大语言模型的评判能力，对交易决策进行多维度评估
    """
    
    def __init__(self, config: SelfRewardConfig):
        """
        初始化LLM评判模块
        
        Args:
            config: 自我奖励配置
        """
        self.config = config
        
        # 评估标准和权重
        self.evaluation_criteria = {
            'profitability': 0.3,      # 盈利能力
            'risk_management': 0.25,   # 风险管理
            'consistency': 0.2,        # 一致性
            'efficiency': 0.15,        # 效率
            'adaptability': 0.1        # 适应性
        }
        
        # 历史评估记录
        self.evaluation_history = deque(maxlen=1000)
        self.judge_performance = defaultdict(list)
        
        # 偏差检测
        self.bias_detector = BiasDetector(config)
        
        logging.info(f"初始化LLM-as-a-Judge评估模块")
    
    def evaluate_trading_decision(self, 
                                trading_context: Dict[str, Any],
                                role: JudgeRole = JudgeRole.JUDGE) -> TradingEvaluation:
        """
        评估交易决策
        
        Args:
            trading_context: 交易上下文信息
            role: 评判角色
            
        Returns:
            TradingEvaluation: 评估结果
        """
        try:
            # 提取关键指标
            current_value = trading_context.get('current_value', 0.0)
            step_return = trading_context.get('step_return_pct', 0.0)
            total_return = trading_context.get('total_return_pct', 0.0)
            action = trading_context.get('current_action', 0.0)
            recent_returns = trading_context.get('recent_returns', [])
            
            # 多维度评估
            criteria_scores = {}
            
            # 1. 盈利能力评估
            profitability_score = self._evaluate_profitability(
                step_return, total_return, recent_returns
            )
            criteria_scores['profitability'] = profitability_score
            
            # 2. 风险管理评估
            risk_score = self._evaluate_risk_management(
                recent_returns, action, step_return
            )
            criteria_scores['risk_management'] = risk_score
            
            # 3. 一致性评估
            consistency_score = self._evaluate_consistency(recent_returns)
            criteria_scores['consistency'] = consistency_score
            
            # 4. 效率评估
            efficiency_score = self._evaluate_efficiency(
                step_return, action, current_value
            )
            criteria_scores['efficiency'] = efficiency_score
            
            # 5. 适应性评估
            adaptability_score = self._evaluate_adaptability(recent_returns, action)
            criteria_scores['adaptability'] = adaptability_score
            
            # 综合评分
            overall_score = sum(
                criteria_scores[criterion] * weight 
                for criterion, weight in self.evaluation_criteria.items()
            )
            
            # 生成评估推理
            reasoning = self._generate_reasoning(criteria_scores, trading_context)
            
            # 计算置信度
            confidence = self._calculate_confidence(criteria_scores, role)
            
            # 创建评估结果
            evaluation = TradingEvaluation(
                score=overall_score,
                reasoning=reasoning,
                confidence=confidence,
                criteria_scores=criteria_scores,
                timestamp=len(self.evaluation_history),
                role=role
            )
            
            # 偏差检测
            bias_penalty = self.bias_detector.detect_bias(evaluation, trading_context)
            evaluation.score = max(0.0, evaluation.score - bias_penalty)
            
            # 记录评估历史
            self.evaluation_history.append(evaluation)
            self.judge_performance[role].append(evaluation.score)
            
            return evaluation
            
        except Exception as e:
            logging.error(f"LLM评估异常: {e}")
            return TradingEvaluation(
                score=5.0,  # 中性分数
                reasoning="评估过程出现异常，返回中性分数",
                confidence=0.1,
                role=role
            )
    
    def _evaluate_profitability(self, step_return: float, total_return: float, 
                              recent_returns: List[float]) -> float:
        """评估盈利能力"""
        # 当前步骤收益评分
        step_score = np.tanh(step_return * 50) * 5 + 5  # 转换到[0-10]
        
        # 总收益评分
        total_score = np.tanh(total_return * 20) * 5 + 5
        
        # 近期收益趋势评分
        if len(recent_returns) >= 5:
            trend_score = (np.mean(recent_returns[-5:]) > 0) * 2 + 3  # [3-5]
        else:
            trend_score = 5.0
        
        # 综合盈利能力评分
        profitability = (step_score * 0.4 + total_score * 0.4 + trend_score * 0.2)
        return np.clip(profitability, 0.0, 10.0)
    
    def _evaluate_risk_management(self, recent_returns: List[float], 
                                action: float, step_return: float) -> float:
        """评估风险管理"""
        if len(recent_returns) < 3:
            return 5.0
        
        # 波动率控制评分
        volatility = np.std(recent_returns[-10:]) if len(recent_returns) >= 10 else np.std(recent_returns)
        volatility_score = max(0, 10 - volatility * 100)  # 低波动率高分
        
        # 下行风险控制评分
        negative_returns = [r for r in recent_returns[-10:] if r < 0]
        if negative_returns:
            max_loss = min(negative_returns)
            downside_score = max(0, 10 + max_loss * 50)  # 控制最大亏损
        else:
            downside_score = 10.0
        
        # 动作合理性评分
        action_score = 10 - abs(action) * 2  # 极端动作扣分
        
        risk_score = (volatility_score * 0.4 + downside_score * 0.4 + action_score * 0.2)
        return np.clip(risk_score, 0.0, 10.0)
    
    def _evaluate_consistency(self, recent_returns: List[float]) -> float:
        """评估一致性"""
        if len(recent_returns) < 5:
            return 5.0
        
        # 收益稳定性
        returns_array = np.array(recent_returns[-10:])
        stability = 1.0 / (1.0 + np.std(returns_array))
        stability_score = stability * 10
        
        # 策略连贯性 (减少频繁变化)
        sign_changes = sum(1 for i in range(1, len(returns_array)) 
                          if np.sign(returns_array[i]) != np.sign(returns_array[i-1]))
        coherence_score = max(0, 10 - sign_changes)
        
        consistency = (stability_score * 0.6 + coherence_score * 0.4)
        return np.clip(consistency, 0.0, 10.0)
    
    def _evaluate_efficiency(self, step_return: float, action: float, 
                           current_value: float) -> float:
        """评估效率"""
        # 收益-动作比率
        if abs(action) > 1e-6:
            efficiency_ratio = abs(step_return) / abs(action)
            efficiency_score = np.tanh(efficiency_ratio * 10) * 10
        else:
            efficiency_score = 5.0 if abs(step_return) < 1e-6 else 2.0  # 无动作无收益为中性
        
        # 资金利用率 (简化指标)
        utilization_score = min(10, current_value / 10000 * 5)  # 基于初始资金的利用率
        
        efficiency = (efficiency_score * 0.7 + utilization_score * 0.3)
        return np.clip(efficiency, 0.0, 10.0)
    
    def _evaluate_adaptability(self, recent_returns: List[float], action: float) -> float:
        """评估适应性"""
        if len(recent_returns) < 5:
            return 5.0
        
        # 市场适应性 - 基于近期表现变化调整策略
        recent_performance = np.mean(recent_returns[-3:])
        earlier_performance = np.mean(recent_returns[-6:-3]) if len(recent_returns) >= 6 else recent_performance
        
        performance_change = recent_performance - earlier_performance
        
        # 如果表现变差，检查是否有策略调整
        if performance_change < 0:
            action_adjustment = abs(action) > 0.1  # 是否有明显的策略调整
            adaptability_score = 8.0 if action_adjustment else 3.0
        else:
            # 表现良好时保持策略的稳定性
            adaptability_score = 7.0 if abs(action) < 0.5 else 5.0
        
        return np.clip(adaptability_score, 0.0, 10.0)
    
    def _generate_reasoning(self, criteria_scores: Dict[str, float], 
                          trading_context: Dict[str, Any]) -> str:
        """生成评估推理"""
        reasoning_parts = []
        
        # 分析各项指标
        for criterion, score in criteria_scores.items():
            if score >= 7.0:
                reasoning_parts.append(f"{criterion}表现优秀({score:.1f})")
            elif score >= 4.0:
                reasoning_parts.append(f"{criterion}表现一般({score:.1f})")
            else:
                reasoning_parts.append(f"{criterion}需要改进({score:.1f})")
        
        # 整体评价
        overall_score = sum(criteria_scores[c] * w for c, w in self.evaluation_criteria.items())
        if overall_score >= 7.0:
            overall_assessment = "交易决策质量优秀"
        elif overall_score >= 4.0:
            overall_assessment = "交易决策质量中等"
        else:
            overall_assessment = "交易决策需要改进"
        
        reasoning = f"{overall_assessment}。" + "；".join(reasoning_parts) + "。"
        
        # 添加具体建议
        step_return = trading_context.get('step_return_pct', 0.0)
        if step_return < -0.02:  # 亏损超过2%
            reasoning += " 建议加强风险控制。"
        elif step_return > 0.03:  # 盈利超过3%
            reasoning += " 表现良好，建议保持策略。"
        
        return reasoning
    
    def _calculate_confidence(self, criteria_scores: Dict[str, float], 
                            role: JudgeRole) -> float:
        """计算评估置信度"""
        # 基于评分的一致性计算置信度
        scores = list(criteria_scores.values())
        score_std = np.std(scores)
        
        # 评分越一致，置信度越高
        consistency_confidence = 1.0 / (1.0 + score_std)
        
        # 角色调整
        role_adjustment = {
            JudgeRole.ACTOR: 0.7,      # 行动者置信度较低
            JudgeRole.JUDGE: 1.0,      # 评判者基准置信度
            JudgeRole.META_JUDGE: 1.2  # 元评判者置信度较高
        }
        
        confidence = consistency_confidence * role_adjustment.get(role, 1.0)
        return np.clip(confidence, 0.1, 1.0)


class BiasDetector:
    """偏差检测器"""
    
    def __init__(self, config: SelfRewardConfig):
        """
        初始化偏差检测器
        
        Args:
            config: 自我奖励配置
        """
        self.config = config
        self.bias_history = deque(maxlen=100)
        
    def detect_bias(self, evaluation: TradingEvaluation, 
                   trading_context: Dict[str, Any]) -> float:
        """
        检测评估偏差并返回惩罚值
        
        Args:
            evaluation: 评估结果
            trading_context: 交易上下文
            
        Returns:
            float: 偏差惩罚值
        """
        total_penalty = 0.0
        
        # 1. 长度偏差检测 (评估推理长度偏差)
        reasoning_length = len(evaluation.reasoning)
        if reasoning_length > 200 or reasoning_length < 20:
            length_penalty = self.config.length_bias_penalty
            total_penalty += length_penalty
        
        # 2. 过度自信检测
        if evaluation.confidence > 0.9 and evaluation.score > 8.0:
            overconfidence_penalty = self.config.overconfidence_penalty
            total_penalty += overconfidence_penalty
        
        # 3. 分数偏差检测 (避免极端分数)
        if evaluation.score > 9.5 or evaluation.score < 0.5:
            extreme_score_penalty = 0.5
            total_penalty += extreme_score_penalty
        
        # 4. 一致性偏差检测
        if len(self.bias_history) >= 5:
            recent_scores = [h.score for h in list(self.bias_history)[-5:]]
            if abs(evaluation.score - np.mean(recent_scores)) > 3.0:
                consistency_penalty = 0.3
                total_penalty += consistency_penalty
        
        # 记录偏差历史
        self.bias_history.append(evaluation)
        
        return total_penalty


class DirectPreferenceOptimizer:
    """
    直接偏好优化器 (DPO)
    
    基于偏好对比学习，优化奖励函数参数
    """
    
    def __init__(self, config: SelfRewardConfig):
        """
        初始化DPO优化器
        
        Args:
            config: 自我奖励配置
        """
        self.config = config
        self.preference_data = deque(maxlen=500)
        self.reward_parameters = {
            'base_weight': config.base_reward_weight,
            'self_weight': config.self_evaluation_weight,
            'meta_weight': config.meta_evaluation_weight
        }
        
    def collect_preference_pair(self, 
                              evaluation_a: TradingEvaluation,
                              evaluation_b: TradingEvaluation,
                              preference: str) -> None:
        """
        收集偏好对比数据
        
        Args:
            evaluation_a: 评估A
            evaluation_b: 评估B  
            preference: 偏好选择 ('a', 'b', 'neutral')
        """
        preference_pair = {
            'evaluation_a': evaluation_a,
            'evaluation_b': evaluation_b,
            'preference': preference,
            'timestamp': len(self.preference_data)
        }
        
        self.preference_data.append(preference_pair)
    
    def optimize_rewards(self) -> Dict[str, float]:
        """
        基于偏好数据优化奖励参数
        
        Returns:
            Dict[str, float]: 优化后的参数
        """
        if len(self.preference_data) < 10:
            return self.reward_parameters.copy()
        
        # 简化的DPO优化
        preference_trends = self._analyze_preference_trends()
        
        # 更新参数
        for param, trend in preference_trends.items():
            current_value = self.reward_parameters[param]
            adjustment = trend * self.config.learning_rate
            self.reward_parameters[param] = np.clip(
                current_value + adjustment, 0.1, 2.0
            )
        
        logging.info(f"DPO参数更新: {self.reward_parameters}")
        return self.reward_parameters.copy()
    
    def _analyze_preference_trends(self) -> Dict[str, float]:
        """分析偏好趋势"""
        trends = defaultdict(float)
        
        recent_preferences = list(self.preference_data)[-20:]
        
        for pair in recent_preferences:
            eval_a = pair['evaluation_a']
            eval_b = pair['evaluation_b']
            preference = pair['preference']
            
            # 分析偏好与评估特征的关系
            if preference == 'a':
                if eval_a.confidence > eval_b.confidence:
                    trends['self_weight'] += 0.1
                if eval_a.score > eval_b.score:
                    trends['base_weight'] += 0.1
            elif preference == 'b':
                if eval_b.confidence > eval_a.confidence:
                    trends['self_weight'] += 0.1
                if eval_b.score > eval_a.score:
                    trends['base_weight'] += 0.1
        
        # 归一化趋势
        total_preferences = len(recent_preferences)
        if total_preferences > 0:
            for key in trends:
                trends[key] /= total_preferences
        
        return dict(trends)


class SelfRewardingReward(BaseRewardScheme):
    """
    自我奖励函数
    
    基于Meta AI 2024年Self-Rewarding理论，实现：
    - 三角色系统 (Actor-Judge-MetaJudge)
    - LLM-as-a-Judge多维度评估
    - 直接偏好优化 (DPO)
    - 自适应偏差检测和纠正
    - 迭代自我改进机制
    
    数学公式:
    R_total = α×R_base + β×R_self + γ×R_meta
    R_self = Judge(trading_context, criteria)
    R_meta = MetaJudge(Judge_quality, bias_penalty)
    """
    
    def __init__(self,
                 base_reward_weight: float = 0.6,
                 self_evaluation_weight: float = 0.3,
                 meta_evaluation_weight: float = 0.1,
                 dpo_beta: float = 0.1,
                 learning_rate: float = 0.01,
                 evaluation_window: int = 20,
                 bias_detection_threshold: float = 0.3,
                 enable_meta_judge: bool = True,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化自我奖励函数
        
        Args:
            base_reward_weight: 基础奖励权重
            self_evaluation_weight: 自我评估权重
            meta_evaluation_weight: 元评估权重
            dpo_beta: DPO温度参数
            learning_rate: 学习率
            evaluation_window: 评估窗口大小
            bias_detection_threshold: 偏差检测阈值
            enable_meta_judge: 是否启用元评判
            initial_balance: 初始资金
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 配置参数
        self.config = SelfRewardConfig(
            base_reward_weight=base_reward_weight,
            self_evaluation_weight=self_evaluation_weight,
            meta_evaluation_weight=meta_evaluation_weight,
            dpo_beta=dpo_beta,
            learning_rate=learning_rate,
            evaluation_window=evaluation_window,
            bias_detection_threshold=bias_detection_threshold
        )
        
        # 初始化组件
        self.llm_judge = LLMAsJudge(self.config)
        self.dpo_optimizer = DirectPreferenceOptimizer(self.config)
        self.enable_meta_judge = enable_meta_judge
        
        # 状态追踪
        self.trading_history = deque(maxlen=200)
        self.evaluation_history = deque(maxlen=100)
        self.returns_history = deque(maxlen=100)
        
        # 统计信息
        self.total_base_reward = 0.0
        self.total_self_reward = 0.0
        self.total_meta_reward = 0.0
        self.improvement_iterations = 0
        
        logging.info(f"初始化SelfRewardingReward: "
                    f"base_weight={base_reward_weight}, "
                    f"self_weight={self_evaluation_weight}, "
                    f"meta_weight={meta_evaluation_weight}")
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 自我奖励函数
        
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
            self.returns_history.append(step_return_pct)
            
            # 1. 计算基础奖励
            base_reward = self._calculate_base_reward(total_return_pct, step_return_pct)
            
            # 2. 自我评估奖励
            trading_context = {
                'portfolio_value': portfolio_value,
                'action': action,
                'price': price,
                'step_return_pct': step_return_pct,
                'total_return_pct': total_return_pct,
                'recent_returns': list(self.returns_history)[-10:],
                'step': step,
                **portfolio_info,
                **trade_info,
                **kwargs
            }
            
            self_evaluation = self.llm_judge.evaluate_trading(trading_context, JudgeRole.JUDGE)
            self_reward = self_evaluation.score - 5.0  # 将[0,10]转换为[-5,5]
            
            # 3. 元评估奖励（如果启用）
            meta_reward = 0.0
            if self.enable_meta_judge and len(self.evaluation_history) >= 5:
                meta_evaluation = self.llm_judge.evaluate_trading(trading_context, JudgeRole.META_JUDGE)
                meta_reward = (meta_evaluation.score - 5.0) * 0.5  # 缩放元奖励
            
            # 4. 综合奖励
            total_reward = (
                self.config.base_reward_weight * base_reward +
                self.config.self_evaluation_weight * self_reward +
                self.config.meta_evaluation_weight * meta_reward
            )
            
            # 记录历史
            self.evaluation_history.append(self_evaluation)
            self.trading_history.append({
                'portfolio_value': portfolio_value,
                'action': action,
                'reward': total_reward,
                'step': step
            })
            
            # 更新状态
            self.previous_value = portfolio_value
            self.step_count += 1
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"SelfRewarding奖励计算异常: {e}")
            return 0.0
    
    def _calculate_base_reward(self, total_return_pct: float, 
                             step_return_pct: float) -> float:
        """
        计算基础奖励
        
        Args:
            total_return_pct: 总收益率
            step_return_pct: 步骤收益率
            
        Returns:
            float: 基础奖励值
        """
        # 简单的基础奖励 = 总收益率 * 100 + 步骤收益率调整
        base_reward = total_return_pct * 100.0 + step_return_pct * 50.0
        return base_reward
    
    def reward(self, env) -> float:
        """
        计算自我奖励综合奖励
        
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
                logging.info(f"[SelfRewarding] 初始化: ${current_value:.2f}")
                return 0.0
            
            # 更新历史
            self.returns_history.append(step_return_pct)
            
            # === 1. 基础奖励计算 ===
            base_reward = self._calculate_base_reward(total_return_pct, step_return_pct)
            self.total_base_reward += base_reward
            
            # === 2. 自我评估奖励计算 ===
            
            # 构建交易上下文
            trading_context = {
                'current_value': current_value,
                'step_return_pct': step_return_pct,
                'total_return_pct': total_return_pct,
                'current_action': current_action,
                'recent_returns': list(self.returns_history)[-20:],
                'step_count': state['step_count']
            }
            
            # LLM-as-a-Judge评估
            self_evaluation = self.llm_judge.evaluate_trading_decision(
                trading_context, JudgeRole.JUDGE
            )
            
            # 转换评估分数为奖励 (0-10 -> -5 to 5)
            self_reward = (self_evaluation.score - 5.0) * self_evaluation.confidence
            self.total_self_reward += self_reward
            
            # === 3. 元评判奖励计算 ===
            meta_reward = 0.0
            if self.enable_meta_judge and len(self.evaluation_history) >= 3:
                # 元评判：评估评判质量
                meta_evaluation = self._meta_judge_evaluation(
                    self_evaluation, trading_context
                )
                meta_reward = (meta_evaluation.score - 5.0) * meta_evaluation.confidence * 0.5
                self.total_meta_reward += meta_reward
            
            # === 4. 综合奖励计算 ===
            
            # 获取当前优化的权重
            current_weights = self.dpo_optimizer.reward_parameters
            
            total_reward = (
                current_weights['base_weight'] * base_reward +
                current_weights['self_weight'] * self_reward +
                current_weights['meta_weight'] * meta_reward
            )
            
            # === 5. 自适应学习 ===
            
            # 记录评估历史
            self.evaluation_history.append(self_evaluation)
            
            # 定期优化参数
            if state['step_count'] % self.config.evaluation_window == 0:
                self._trigger_self_improvement()
            
            # 详细日志
            if state['step_count'] % 30 == 0 or abs(total_reward) > 3.0:
                logging.info(
                    f"[SelfRewarding] 步骤{state['step_count']}: "
                    f"基础{base_reward:.3f} + 自评{self_reward:.3f} + 元评{meta_reward:.3f} = "
                    f"{total_reward:.3f} (评估分数:{self_evaluation.score:.1f}, "
                    f"置信度:{self_evaluation.confidence:.2f})"
                )
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"SelfRewarding奖励计算异常: {e}")
            return 0.0
    
    def _meta_judge_evaluation(self, 
                             judge_evaluation: TradingEvaluation,
                             trading_context: Dict[str, Any]) -> TradingEvaluation:
        """
        元评判：评估评判质量
        
        Args:
            judge_evaluation: 主评判结果
            trading_context: 交易上下文
            
        Returns:
            TradingEvaluation: 元评判结果
        """
        # 元评判标准
        meta_score = 5.0  # 基准分数
        
        # 1. 评估合理性检查
        actual_return = trading_context['step_return_pct']
        if actual_return > 0.02 and judge_evaluation.score < 5.0:
            meta_score -= 1.0  # 盈利却给低分，扣分
        elif actual_return < -0.02 and judge_evaluation.score > 7.0:
            meta_score -= 1.0  # 亏损却给高分，扣分
        else:
            meta_score += 0.5  # 评估合理，加分
        
        # 2. 置信度合理性
        if judge_evaluation.confidence > 0.9 and abs(actual_return) < 0.005:
            meta_score -= 0.5  # 小变化高置信度不合理
        
        # 3. 评估一致性 (与历史评估对比)
        if len(self.evaluation_history) >= 3:
            recent_scores = [e.score for e in list(self.evaluation_history)[-3:]]
            score_volatility = np.std(recent_scores + [judge_evaluation.score])
            if score_volatility > 2.0:
                meta_score -= 0.3  # 评估过于波动
            else:
                meta_score += 0.2  # 评估稳定
        
        # 4. 详细程度检查
        reasoning_quality = len(judge_evaluation.reasoning) > 30
        if reasoning_quality:
            meta_score += 0.3
        
        meta_score = np.clip(meta_score, 0.0, 10.0)
        
        # 元评判置信度
        meta_confidence = min(0.8, judge_evaluation.confidence * 0.9)
        
        return TradingEvaluation(
            score=meta_score,
            reasoning=f"元评判：主评判分数{judge_evaluation.score:.1f}的质量评估",
            confidence=meta_confidence,
            role=JudgeRole.META_JUDGE
        )
    
    def _trigger_self_improvement(self):
        """触发自我改进机制"""
        try:
            # 分析最近的评估质量
            if len(self.evaluation_history) < 5:
                return
            
            recent_evaluations = list(self.evaluation_history)[-10:]
            
            # 生成偏好对比数据
            for i in range(len(recent_evaluations) - 1):
                eval_a = recent_evaluations[i]
                eval_b = recent_evaluations[i + 1]
                
                # 简化的偏好判断：基于实际表现vs评估分数的一致性
                context_a = self.trading_history[-(len(recent_evaluations) - i)] if len(self.trading_history) > len(recent_evaluations) - i else None
                context_b = self.trading_history[-(len(recent_evaluations) - i - 1)] if len(self.trading_history) > len(recent_evaluations) - i - 1 else None
                
                if context_a and context_b:
                    # 基于实际收益判断偏好
                    actual_a = context_a.get('step_return_pct', 0.0)
                    actual_b = context_b.get('step_return_pct', 0.0)
                    
                    if abs(actual_a - actual_b) > 0.01:  # 收益差异足够大
                        preference = 'a' if actual_a > actual_b else 'b'
                        self.dpo_optimizer.collect_preference_pair(eval_a, eval_b, preference)
            
            # 执行DPO优化
            new_weights = self.dpo_optimizer.optimize_rewards()
            
            # 更新配置
            self.config.base_reward_weight = new_weights.get('base_weight', self.config.base_reward_weight)
            self.config.self_evaluation_weight = new_weights.get('self_weight', self.config.self_evaluation_weight)
            self.config.meta_evaluation_weight = new_weights.get('meta_weight', self.config.meta_evaluation_weight)
            
            self.improvement_iterations += 1
            
            logging.info(f"[SelfRewarding] 自我改进第{self.improvement_iterations}轮: "
                        f"权重更新为 base:{new_weights['base_weight']:.3f}, "
                        f"self:{new_weights['self_weight']:.3f}, "
                        f"meta:{new_weights['meta_weight']:.3f}")
            
        except Exception as e:
            logging.error(f"自我改进过程异常: {e}")
    
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
            
            # 更新历史
            self.returns_history.append(step_return_pct)
            
            # 构建交易上下文
            trading_context = {
                'current_value': current_value,
                'step_return_pct': step_return_pct,
                'total_return_pct': total_return_pct,
                'current_action': 0.0,  # 简化处理
                'recent_returns': list(self.returns_history)[-20:],
                'step_count': self.step_count + 1
            }
            
            self.trading_history.append(trading_context)
            
            # 基础奖励
            base_reward = self._calculate_base_reward(total_return_pct, step_return_pct)
            
            # 自我评估奖励
            self_evaluation = self.llm_judge.evaluate_trading_decision(
                trading_context, JudgeRole.JUDGE
            )
            self_reward = (self_evaluation.score - 5.0) * self_evaluation.confidence
            
            # 综合奖励
            current_weights = self.dpo_optimizer.reward_parameters
            total_reward = (
                current_weights['base_weight'] * base_reward +
                current_weights['self_weight'] * self_reward
            )
            
            # 更新状态
            self.previous_value = current_value
            self.step_count += 1
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"SelfRewarding get_reward异常: {e}")
            return 0.0
    
    def reset(self) -> 'SelfRewardingReward':
        """
        重置奖励函数状态
        
        Returns:
            SelfRewardingReward: 返回self以支持链式调用
        """
        # 记录上一回合的统计信息
        logging.info(f"[SelfRewarding] 回合{self.episode_count}结束: "
                    f"基础总奖励{self.total_base_reward:.3f}, "
                    f"自评总奖励{self.total_self_reward:.3f}, "
                    f"元评总奖励{self.total_meta_reward:.3f}, "
                    f"改进迭代{self.improvement_iterations}次")
        
        # 调用父类reset
        super().reset()
        
        # 重置短期状态，保留学习成果
        self.returns_history.clear()
        # 保留evaluation_history和trading_history用于持续学习
        
        # 重置统计
        self.total_base_reward = 0.0
        self.total_self_reward = 0.0
        self.total_meta_reward = 0.0
        
        return self
    
    def get_self_reward_metrics(self) -> Dict[str, Any]:
        """
        获取自我奖励相关指标
        
        Returns:
            Dict[str, Any]: 自我奖励指标字典
        """
        metrics = {
            'total_base_reward': self.total_base_reward,
            'total_self_reward': self.total_self_reward,
            'total_meta_reward': self.total_meta_reward,
            'improvement_iterations': self.improvement_iterations,
            'evaluation_count': len(self.evaluation_history),
            'current_weights': self.dpo_optimizer.reward_parameters.copy(),
            'avg_evaluation_score': 0.0,
            'avg_confidence': 0.0,
            'bias_detection_count': len(self.llm_judge.bias_detector.bias_history)
        }
        
        if self.evaluation_history:
            recent_evaluations = list(self.evaluation_history)[-20:]
            metrics['avg_evaluation_score'] = float(np.mean([e.score for e in recent_evaluations]))
            metrics['avg_confidence'] = float(np.mean([e.confidence for e in recent_evaluations]))
        
        return metrics
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'SelfRewardingReward',
            'description': '自我奖励函数，基于Meta AI 2024年Self-Rewarding理论，实现三角色系统、LLM-as-a-Judge评估和直接偏好优化',
            'category': 'self_improving',
            'complexity': 'expert',
            'parameters': {
                'base_reward_weight': {
                    'type': 'float',
                    'default': 0.6,
                    'description': '基础奖励权重'
                },
                'self_evaluation_weight': {
                    'type': 'float',
                    'default': 0.3,
                    'description': '自我评估权重'
                },
                'meta_evaluation_weight': {
                    'type': 'float',
                    'default': 0.1,
                    'description': '元评估权重'
                },
                'dpo_beta': {
                    'type': 'float',
                    'default': 0.1,
                    'description': 'DPO温度参数'
                },
                'learning_rate': {
                    'type': 'float',
                    'default': 0.01,
                    'description': '学习率'
                },
                'evaluation_window': {
                    'type': 'int',
                    'default': 20,
                    'description': '评估窗口大小'
                },
                'bias_detection_threshold': {
                    'type': 'float',
                    'default': 0.3,
                    'description': '偏差检测阈值'
                },
                'enable_meta_judge': {
                    'type': 'bool',
                    'default': True,
                    'description': '是否启用元评判'
                }
            },
            'features': [
                '三角色系统 (Actor-Judge-MetaJudge)',
                'LLM-as-a-Judge多维度评估',
                '直接偏好优化 (DPO)',
                '自适应偏差检测和纠正',
                '迭代自我改进机制',
                '多标准交易质量评估',
                '实时权重优化',
                '长度偏差和过度自信检测'
            ],
            'mathematical_foundation': [
                'Self-Rewarding Language Models (Meta AI 2024)',
                'Meta-Rewarding Language Models (Meta AI 2024)',
                'Direct Preference Optimization (DPO)',
                'LLM-as-a-Judge Framework',
                'Iterative Self-Improvement',
                'Bias Detection and Mitigation',
                'Multi-Criteria Decision Analysis',
                'Adaptive Weight Optimization'
            ],
            'applications': [
                '需要自我改进的交易系统',
                '复杂决策质量评估',
                '自适应奖励机制',
                '多标准优化问题',
                '智能偏差检测',
                '持续学习系统',
                '质量自我监控',
                '决策解释和推理'
            ]
        }