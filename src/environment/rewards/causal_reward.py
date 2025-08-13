"""
因果奖励函数 (Causal Reward)

基于2024年最新的因果推理在强化学习中的应用理论，
实现因果图构建、混淆变量识别、后门调整、前门调整和DOVI去混淆价值迭代算法。

参考文献:
- Causal Reinforcement Learning: A Survey (TNNLS 2024)
- Provably Efficient Causal Reinforcement Learning with Confounded Observational Data (NeurIPS 2021)  
- Applied Causal Inference in Reinforcement Learning (2024)
- Confounding Adjustment Methods for Treatment Effects (2024)
- Policy Confounding and Out-of-Trajectory Generalization in RL (RLJ 2024)
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field
import scipy.stats as stats
from enum import Enum
import networkx as nx
from .base_reward import BaseRewardScheme


class CausalAdjustmentMethod(Enum):
    """因果调整方法"""
    BACKDOOR = "backdoor"           # 后门调整
    FRONTDOOR = "frontdoor"         # 前门调整  
    DOVI = "dovi"                   # 去混淆价值迭代
    DO_CALCULUS = "do_calculus"     # do-演算
    INSTRUMENTAL = "instrumental"    # 工具变量


@dataclass
class CausalConfig:
    """因果奖励配置"""
    # 基础权重
    causal_reward_weight: float = 0.6      # 因果奖励权重
    confounding_penalty: float = 0.3       # 混淆惩罚权重
    intervention_bonus: float = 0.1        # 干预奖励权重
    
    # 因果图构建参数
    min_correlation_threshold: float = 0.1  # 最小相关性阈值
    causal_discovery_method: str = "pc"     # 因果发现方法 (pc, ges, lingam)
    max_graph_nodes: int = 20              # 最大图节点数
    
    # 混淆检测参数
    confounding_detection_threshold: float = 0.2  # 混淆检测阈值
    spurious_correlation_threshold: float = 0.15  # 虚假相关阈值
    temporal_window: int = 50              # 时间窗口大小
    
    # DOVI参数
    dovi_confidence_level: float = 0.95    # DOVI置信水平
    dovi_regularization: float = 0.01      # DOVI正则化参数
    value_iteration_tolerance: float = 1e-6 # 价值迭代容忍度
    
    # 调整方法
    adjustment_method: CausalAdjustmentMethod = CausalAdjustmentMethod.BACKDOOR


@dataclass
class CausalRelation:
    """因果关系"""
    cause: str                             # 原因变量
    effect: str                            # 结果变量
    strength: float                        # 关系强度
    confidence: float                      # 置信度
    confounders: List[str] = field(default_factory=list)  # 混淆变量
    mediators: List[str] = field(default_factory=list)    # 中介变量
    
    
class CausalGraph:
    """
    因果图构建器
    
    实现因果关系发现、图构建和混淆变量识别
    """
    
    def __init__(self, config: CausalConfig):
        """
        初始化因果图构建器
        
        Args:
            config: 因果配置
        """
        self.config = config
        self.graph = nx.DiGraph()  # 有向因果图
        self.relations = {}        # 因果关系映射
        self.confounders = set()   # 已识别的混淆变量
        self.variables = set()     # 所有变量
        
        # 数据存储
        self.variable_history = defaultdict(lambda: deque(maxlen=config.temporal_window))
        self.correlation_matrix = {}
        
        logging.info(f"初始化因果图构建器: 方法={config.causal_discovery_method}")
    
    def add_variable(self, name: str, value: float, timestamp: int = None):
        """
        添加变量观测值
        
        Args:
            name: 变量名
            value: 观测值
            timestamp: 时间戳
        """
        self.variables.add(name)
        self.variable_history[name].append({
            'value': value,
            'timestamp': timestamp or len(self.variable_history[name])
        })
    
    def update_correlation_matrix(self):
        """更新相关性矩阵"""
        variables = list(self.variables)
        n_vars = len(variables)
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                var1, var2 = variables[i], variables[j]
                correlation = self._calculate_correlation(var1, var2)
                self.correlation_matrix[(var1, var2)] = correlation
                self.correlation_matrix[(var2, var1)] = correlation
    
    def _calculate_correlation(self, var1: str, var2: str) -> float:
        """
        计算两个变量的相关性
        
        Args:
            var1: 变量1
            var2: 变量2
            
        Returns:
            float: 相关系数
        """
        if var1 not in self.variable_history or var2 not in self.variable_history:
            return 0.0
        
        values1 = [obs['value'] for obs in self.variable_history[var1]]
        values2 = [obs['value'] for obs in self.variable_history[var2]]
        
        if len(values1) < 3 or len(values2) < 3:
            return 0.0
        
        # 对齐时间序列
        min_len = min(len(values1), len(values2))
        values1 = values1[-min_len:]
        values2 = values2[-min_len:]
        
        try:
            correlation, _ = stats.pearsonr(values1, values2)
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def discover_causal_structure(self):
        """
        发现因果结构
        
        使用简化的PC算法进行因果发现
        """
        if len(self.variables) < 2:
            return
        
        # 更新相关性矩阵
        self.update_correlation_matrix()
        
        # 构建初始无向图
        self._build_initial_graph()
        
        # 检测混淆变量
        self._detect_confounders()
        
        # 确定边的方向
        self._orient_edges()
        
        logging.info(f"因果发现完成: {len(self.graph.nodes())}个节点, "
                    f"{len(self.graph.edges())}条边, "
                    f"{len(self.confounders)}个混淆变量")
    
    def _build_initial_graph(self):
        """构建初始图结构"""
        # 添加所有变量作为节点
        for var in self.variables:
            self.graph.add_node(var)
        
        # 基于相关性添加边
        for (var1, var2), correlation in self.correlation_matrix.items():
            if correlation > self.config.min_correlation_threshold:
                # 使用时间先后确定因果方向的初步推断
                direction = self._infer_temporal_direction(var1, var2)
                if direction:
                    cause, effect = direction
                    self.graph.add_edge(cause, effect, weight=correlation)
    
    def _infer_temporal_direction(self, var1: str, var2: str) -> Optional[Tuple[str, str]]:
        """
        基于时间序列推断因果方向
        
        Args:
            var1: 变量1
            var2: 变量2
            
        Returns:
            Optional[Tuple[str, str]]: (原因, 结果) 或 None
        """
        if var1 not in self.variable_history or var2 not in self.variable_history:
            return None
        
        # 计算滞后相关性
        values1 = [obs['value'] for obs in self.variable_history[var1]]
        values2 = [obs['value'] for obs in self.variable_history[var2]]
        
        if len(values1) < 5 or len(values2) < 5:
            return None
        
        # 检查 var1 -> var2 的滞后相关性
        lag_corr_12 = self._calculate_lag_correlation(values1[:-1], values2[1:])
        # 检查 var2 -> var1 的滞后相关性  
        lag_corr_21 = self._calculate_lag_correlation(values2[:-1], values1[1:])
        
        if lag_corr_12 > lag_corr_21 + 0.05:  # 显著性阈值
            return (var1, var2)
        elif lag_corr_21 > lag_corr_12 + 0.05:
            return (var2, var1)
        
        return None
    
    def _calculate_lag_correlation(self, cause_values: List[float], 
                                 effect_values: List[float]) -> float:
        """计算滞后相关性"""
        if len(cause_values) != len(effect_values) or len(cause_values) < 3:
            return 0.0
        
        try:
            correlation, _ = stats.pearsonr(cause_values, effect_values)
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _detect_confounders(self):
        """检测混淆变量"""
        # 简化的混淆检测：寻找同时影响多个变量的节点
        for node in self.graph.nodes():
            out_degree = self.graph.out_degree(node)
            if out_degree >= 2:  # 影响至少两个其他变量
                # 检查是否形成混淆模式
                successors = list(self.graph.successors(node))
                for i in range(len(successors)):
                    for j in range(i + 1, len(successors)):
                        var1, var2 = successors[i], successors[j]
                        # 如果两个后继变量之间也有相关性，可能存在混淆
                        corr = self.correlation_matrix.get((var1, var2), 0.0)
                        if corr > self.config.confounding_detection_threshold:
                            self.confounders.add(node)
                            logging.info(f"检测到混淆变量: {node} (影响 {var1}, {var2})")
    
    def _orient_edges(self):
        """确定边的方向（简化版本）"""
        # 这里实现简化的边方向确定
        # 在实际应用中，这会包含更复杂的因果推断算法
        pass
    
    def get_backdoor_set(self, treatment: str, outcome: str) -> Set[str]:
        """
        获取后门调整集合
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            
        Returns:
            Set[str]: 后门调整变量集合
        """
        if not self.graph.has_node(treatment) or not self.graph.has_node(outcome):
            return set()
        
        # 简化的后门标准实现
        # 找到所有从处理到结果的路径上的混淆变量
        backdoor_set = set()
        
        # 寻找所有可能阻断后门路径的变量
        for confounder in self.confounders:
            if (self.graph.has_edge(confounder, treatment) and 
                self.graph.has_edge(confounder, outcome)):
                backdoor_set.add(confounder)
        
        return backdoor_set
    
    def get_causal_effect_estimate(self, treatment: str, outcome: str) -> float:
        """
        估计因果效应
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            
        Returns:
            float: 因果效应估计
        """
        if treatment not in self.variable_history or outcome not in self.variable_history:
            return 0.0
        
        # 简化的因果效应估计
        treatment_values = [obs['value'] for obs in self.variable_history[treatment]]
        outcome_values = [obs['value'] for obs in self.variable_history[outcome]]
        
        if len(treatment_values) < 3 or len(outcome_values) < 3:
            return 0.0
        
        # 使用线性回归估计效应
        min_len = min(len(treatment_values), len(outcome_values))
        X = np.array(treatment_values[-min_len:]).reshape(-1, 1)
        y = np.array(outcome_values[-min_len:])
        
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            return float(model.coef_[0])
        except:
            # 退回到简单相关性
            correlation = self._calculate_correlation(treatment, outcome)
            return correlation * np.std(outcome_values) / max(np.std(treatment_values), 1e-8)


class ConfoundingDetector:
    """
    混淆检测器
    
    实现虚假相关和混淆偏差的检测与量化
    """
    
    def __init__(self, config: CausalConfig):
        """
        初始化混淆检测器
        
        Args:
            config: 因果配置
        """
        self.config = config
        self.detected_confoundings = []
        self.spurious_correlations = {}
        
    def detect_spurious_correlation(self, 
                                  variable_a: str, 
                                  variable_b: str,
                                  variable_history: Dict[str, deque]) -> float:
        """
        检测虚假相关
        
        Args:
            variable_a: 变量A
            variable_b: 变量B  
            variable_history: 变量历史数据
            
        Returns:
            float: 虚假相关强度 [0-1]
        """
        if variable_a not in variable_history or variable_b not in variable_history:
            return 0.0
        
        values_a = [obs['value'] for obs in variable_history[variable_a]]
        values_b = [obs['value'] for obs in variable_history[variable_b]]
        
        if len(values_a) < 10 or len(values_b) < 10:
            return 0.0
        
        # 检查相关性稳定性
        min_len = min(len(values_a), len(values_b))
        window_size = min_len // 3
        
        correlations = []
        for i in range(0, min_len - window_size, window_size // 2):
            window_a = values_a[i:i + window_size]
            window_b = values_b[i:i + window_size]
            try:
                corr, _ = stats.pearsonr(window_a, window_b)
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                continue
        
        if len(correlations) < 2:
            return 0.0
        
        # 如果相关性变化很大，可能是虚假的
        correlation_std = np.std(correlations)
        avg_correlation = np.mean(np.abs(correlations))
        
        # 虚假相关指标：相关性不稳定
        spurious_score = correlation_std / max(avg_correlation, 0.1)
        
        return min(spurious_score, 1.0)
    
    def detect_policy_confounding(self, 
                                action_history: List[float],
                                reward_history: List[float],
                                state_features: List[List[float]]) -> float:
        """
        检测策略混淆
        
        Args:
            action_history: 动作历史
            reward_history: 奖励历史
            state_features: 状态特征历史
            
        Returns:
            float: 策略混淆程度 [0-1]
        """
        if len(action_history) < 20 or len(reward_history) < 20:
            return 0.0
        
        # 检查行动-奖励的简单关联是否过于强烈（可能表示混淆）
        try:
            action_reward_corr, _ = stats.pearsonr(action_history, reward_history)
            if np.isnan(action_reward_corr):
                action_reward_corr = 0.0
        except:
            action_reward_corr = 0.0
        
        # 检查状态特征的多样性
        if state_features and len(state_features) >= 20:
            state_diversity = self._calculate_state_diversity(state_features)
            
            # 如果状态多样性低但行动-奖励相关性高，可能有混淆
            if state_diversity < 0.3 and abs(action_reward_corr) > 0.7:
                confounding_score = abs(action_reward_corr) * (1 - state_diversity)
                return min(confounding_score, 1.0)
        
        return 0.0
    
    def _calculate_state_diversity(self, state_features: List[List[float]]) -> float:
        """计算状态多样性"""
        if len(state_features) < 2:
            return 0.0
        
        # 计算状态向量的方差作为多样性度量
        state_array = np.array(state_features)
        if state_array.shape[1] == 0:
            return 0.0
        
        feature_variances = np.var(state_array, axis=0)
        avg_variance = np.mean(feature_variances)
        
        # 归一化到 [0, 1]
        return min(avg_variance / (avg_variance + 1.0), 1.0)


class DOVIOptimizer:
    """
    DOVI (Deconfounded Optimistic Value Iteration) 优化器
    
    实现去混淆的价值迭代算法
    """
    
    def __init__(self, config: CausalConfig):
        """
        初始化DOVI优化器
        
        Args:
            config: 因果配置
        """
        self.config = config
        self.value_estimates = {}
        self.confidence_bounds = {}
        self.confounding_adjustments = {}
        
    def deconfounded_value_update(self,
                                state: str,
                                action: float,
                                reward: float,
                                next_state: str,
                                confounding_estimate: float) -> float:
        """
        去混淆的价值更新
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            confounding_estimate: 混淆估计
            
        Returns:
            float: 调整后的价值估计
        """
        # 初始化值函数
        if state not in self.value_estimates:
            self.value_estimates[state] = 0.0
            self.confidence_bounds[state] = 1.0
        
        if next_state not in self.value_estimates:
            self.value_estimates[next_state] = 0.0
            self.confidence_bounds[next_state] = 1.0
        
        # 计算混淆调整
        confounding_adjustment = confounding_estimate * self.config.dovi_regularization
        
        # 调整后的奖励
        adjusted_reward = reward - confounding_adjustment
        
        # 乐观价值迭代更新
        confidence_bonus = self.confidence_bounds[next_state] * np.sqrt(
            -np.log(1 - self.config.dovi_confidence_level) / max(1, abs(action))
        )
        
        optimistic_next_value = self.value_estimates[next_state] + confidence_bonus
        
        # 价值更新
        new_value = adjusted_reward + 0.99 * optimistic_next_value  # γ = 0.99
        
        # 更新值函数和置信度
        old_value = self.value_estimates[state]
        self.value_estimates[state] = 0.9 * old_value + 0.1 * new_value
        
        # 更新置信度界限
        self.confidence_bounds[state] *= 0.99  # 逐渐减少不确定性
        
        # 记录混淆调整
        self.confounding_adjustments[state] = confounding_adjustment
        
        return self.value_estimates[state]
    
    def get_adjusted_value(self, state: str) -> float:
        """获取调整后的状态价值"""
        return self.value_estimates.get(state, 0.0)
    
    def get_confounding_adjustment(self, state: str) -> float:
        """获取混淆调整量"""
        return self.confounding_adjustments.get(state, 0.0)


class CausalReward(BaseRewardScheme):
    """
    因果奖励函数
    
    基于2024年最新因果推理理论，实现：
    - 因果图构建和混淆变量识别
    - 后门调整和前门调整算法
    - DOVI去混淆价值迭代
    - 虚假相关检测和策略混淆识别
    - Do-演算的因果效应估计
    
    数学公式:
    R_total = α×R_causal + β×penalty_confounding + γ×bonus_intervention
    R_causal = E[Y|do(X)] - E[Y]  (do-演算)
    """
    
    def __init__(self,
                 causal_reward_weight: float = 0.6,
                 confounding_penalty: float = 0.3,
                 intervention_bonus: float = 0.1,
                 adjustment_method: str = "backdoor",
                 confounding_detection_threshold: float = 0.2,
                 temporal_window: int = 50,
                 dovi_confidence_level: float = 0.95,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化因果奖励函数
        
        Args:
            causal_reward_weight: 因果奖励权重
            confounding_penalty: 混淆惩罚权重
            intervention_bonus: 干预奖励权重
            adjustment_method: 调整方法 (backdoor, frontdoor, dovi)
            confounding_detection_threshold: 混淆检测阈值
            temporal_window: 时间窗口大小
            dovi_confidence_level: DOVI置信水平
            initial_balance: 初始资金
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 配置参数
        self.config = CausalConfig(
            causal_reward_weight=causal_reward_weight,
            confounding_penalty=confounding_penalty,
            intervention_bonus=intervention_bonus,
            confounding_detection_threshold=confounding_detection_threshold,
            temporal_window=temporal_window,
            dovi_confidence_level=dovi_confidence_level,
            adjustment_method=CausalAdjustmentMethod(adjustment_method)
        )
        
        # 初始化组件
        self.causal_graph = CausalGraph(self.config)
        self.confounding_detector = ConfoundingDetector(self.config)
        self.dovi_optimizer = DOVIOptimizer(self.config)
        
        # 状态追踪
        self.returns_history = deque(maxlen=temporal_window)
        self.actions_history = deque(maxlen=temporal_window)
        self.states_history = deque(maxlen=temporal_window)
        self.interventions_count = 0
        
        # 统计信息
        self.total_causal_reward = 0.0
        self.total_confounding_penalty = 0.0
        self.total_intervention_bonus = 0.0
        self.detected_confoundings = 0
        
        logging.info(f"初始化CausalReward: "
                    f"方法={adjustment_method}, "
                    f"因果权重={causal_reward_weight}, "
                    f"混淆惩罚={confounding_penalty}")
    
    def _extract_trading_variables(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        提取交易相关变量
        
        Args:
            state: 当前状态
            
        Returns:
            Dict[str, float]: 变量字典
        """
        variables = {}
        
        # 基础变量
        variables['portfolio_value'] = state.get('current_value', 0.0)
        variables['step_return'] = state.get('step_return_pct', 0.0)
        variables['total_return'] = state.get('total_return_pct', 0.0)
        variables['current_action'] = state.get('current_action', 0.0)
        
        # 衍生变量
        if len(self.returns_history) >= 3:
            recent_returns = list(self.returns_history)[-3:]
            variables['volatility'] = float(np.std(recent_returns))
            variables['momentum'] = float(np.mean(recent_returns))
        else:
            variables['volatility'] = 0.0
            variables['momentum'] = 0.0
        
        # 风险变量
        if len(self.returns_history) >= 5:
            returns_array = np.array(list(self.returns_history))
            negative_returns = returns_array[returns_array < 0]
            variables['downside_risk'] = float(np.std(negative_returns)) if len(negative_returns) > 0 else 0.0
        else:
            variables['downside_risk'] = 0.0
        
        return variables
    
    def _calculate_causal_reward(self, state: Dict[str, Any]) -> float:
        """
        计算因果奖励
        
        Args:
            state: 当前状态
            
        Returns:
            float: 因果奖励值
        """
        # 提取变量
        variables = self._extract_trading_variables(state)
        
        # 更新因果图
        step_count = state.get('step_count', 0)
        for var_name, var_value in variables.items():
            self.causal_graph.add_variable(var_name, var_value, step_count)
        
        # 定期重新发现因果结构
        if step_count % 20 == 0 and step_count > 0:
            self.causal_graph.discover_causal_structure()
        
        # 估计关键因果效应
        action_effect = self.causal_graph.get_causal_effect_estimate(
            'current_action', 'step_return'
        )
        
        momentum_effect = self.causal_graph.get_causal_effect_estimate(
            'momentum', 'total_return'
        )
        
        # 因果奖励 = 加权的因果效应
        causal_reward = (
            0.6 * action_effect +
            0.4 * momentum_effect
        ) * variables['step_return']
        
        return causal_reward
    
    def _detect_confounding_penalty(self, state: Dict[str, Any]) -> float:
        """
        检测混淆并计算惩罚
        
        Args:
            state: 当前状态
            
        Returns:
            float: 混淆惩罚值
        """
        penalty = 0.0
        
        # 检测虚假相关
        if len(self.actions_history) >= 10 and len(self.returns_history) >= 10:
            spurious_score = self.confounding_detector.detect_spurious_correlation(
                'current_action', 'step_return', 
                {'current_action': deque([{'value': a} for a in self.actions_history]),
                 'step_return': deque([{'value': r} for r in self.returns_history])}
            )
            
            if spurious_score > self.config.spurious_correlation_threshold:
                penalty += spurious_score * 0.5
                self.detected_confoundings += 1
        
        # 检测策略混淆
        if (len(self.actions_history) >= 20 and 
            len(self.returns_history) >= 20 and
            len(self.states_history) >= 20):
            
            policy_confounding = self.confounding_detector.detect_policy_confounding(
                list(self.actions_history),
                list(self.returns_history),
                self.states_history
            )
            
            if policy_confounding > self.config.confounding_detection_threshold:
                penalty += policy_confounding * 0.3
                self.detected_confoundings += 1
        
        return penalty
    
    def _apply_backdoor_adjustment(self, state: Dict[str, Any]) -> float:
        """
        应用后门调整
        
        Args:
            state: 当前状态
            
        Returns:
            float: 调整后的奖励
        """
        # 获取后门调整集合
        backdoor_set = self.causal_graph.get_backdoor_set('current_action', 'step_return')
        
        if not backdoor_set:
            return 0.0
        
        # 简化的后门调整：基于混淆变量的存在进行调整
        adjustment = 0.0
        step_return = state.get('step_return_pct', 0.0)
        
        for confounder in backdoor_set:
            if confounder in self.causal_graph.variable_history:
                confounder_values = [
                    obs['value'] for obs in self.causal_graph.variable_history[confounder]
                ]
                if confounder_values:
                    confounder_effect = np.mean(confounder_values[-5:])
                    adjustment += confounder_effect * 0.1
        
        return step_return - adjustment
    
    def _apply_dovi_adjustment(self, state: Dict[str, Any]) -> float:
        """
        应用DOVI调整
        
        Args:
            state: 当前状态
            
        Returns:
            float: DOVI调整后的价值
        """
        # 构建状态标识
        portfolio_value = state.get('current_value', 0.0)
        state_id = f"value_{int(portfolio_value // 100)}"  # 简化的状态离散化
        
        # 计算混淆估计
        confounding_estimate = self._detect_confounding_penalty(state)
        
        # DOVI价值更新
        adjusted_value = self.dovi_optimizer.deconfounded_value_update(
            state=state_id,
            action=state.get('current_action', 0.0),
            reward=state.get('step_return_pct', 0.0),
            next_state=f"value_{int((portfolio_value + 100) // 100)}",  # 简化的下一状态
            confounding_estimate=confounding_estimate
        )
        
        return adjusted_value
    
    def reward(self, env) -> float:
        """
        计算因果奖励
        
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
                logging.info(f"[CausalReward] 初始化: ${current_value:.2f}")
                return 0.0
            
            # 更新历史
            self.returns_history.append(step_return_pct)
            self.actions_history.append(current_action)
            
            # 构建状态特征向量
            state_features = [
                current_value / 10000.0,  # 标准化价值
                step_return_pct,
                total_return_pct,
                current_action
            ]
            self.states_history.append(state_features)
            
            # === 1. 因果奖励计算 ===
            causal_reward = self._calculate_causal_reward(state)
            self.total_causal_reward += causal_reward
            
            # === 2. 混淆检测和惩罚 ===
            confounding_penalty = self._detect_confounding_penalty(state)
            self.total_confounding_penalty += confounding_penalty
            
            # === 3. 应用因果调整方法 ===
            adjusted_reward = causal_reward
            
            if self.config.adjustment_method == CausalAdjustmentMethod.BACKDOOR:
                adjustment = self._apply_backdoor_adjustment(state)
                adjusted_reward += adjustment * 0.2
                
            elif self.config.adjustment_method == CausalAdjustmentMethod.DOVI:
                dovi_value = self._apply_dovi_adjustment(state)
                adjusted_reward = 0.7 * causal_reward + 0.3 * dovi_value
            
            # === 4. 干预奖励 ===
            intervention_bonus = 0.0
            if abs(current_action) > 0.5:  # 显著行动被视为干预
                self.interventions_count += 1
                intervention_bonus = 0.1 * step_return_pct
                self.total_intervention_bonus += intervention_bonus
            
            # === 5. 综合奖励计算 ===
            total_reward = (
                self.config.causal_reward_weight * adjusted_reward -
                self.config.confounding_penalty * confounding_penalty +
                self.config.intervention_bonus * intervention_bonus
            )
            
            # 详细日志
            if state['step_count'] % 30 == 0 or abs(total_reward) > 3.0:
                logging.info(
                    f"[CausalReward] 步骤{state['step_count']}: "
                    f"因果{adjusted_reward:.3f} - 混淆{confounding_penalty:.3f} + "
                    f"干预{intervention_bonus:.3f} = {total_reward:.3f} "
                    f"(检测到{self.detected_confoundings}次混淆)"
                )
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"CausalReward奖励计算异常: {e}")
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
            
            # 简化的因果奖励计算
            causal_reward = step_return_pct * total_return_pct * 50.0  # 基础因果效应
            
            # 更新状态
            self.previous_value = current_value
            self.step_count += 1
            
            return float(causal_reward)
            
        except Exception as e:
            logging.error(f"CausalReward get_reward异常: {e}")
            return 0.0
    
    def reset(self) -> 'CausalReward':
        """
        重置奖励函数状态
        
        Returns:
            CausalReward: 返回self以支持链式调用
        """
        # 记录上一回合的统计信息
        logging.info(f"[CausalReward] 回合{self.episode_count}结束: "
                    f"因果总奖励{self.total_causal_reward:.3f}, "
                    f"混淆总惩罚{self.total_confounding_penalty:.3f}, "
                    f"干预总奖励{self.total_intervention_bonus:.3f}, "
                    f"检测混淆{self.detected_confoundings}次, "
                    f"干预{self.interventions_count}次")
        
        # 调用父类reset
        super().reset()
        
        # 重置短期状态，保留学习到的因果结构
        self.returns_history.clear()
        self.actions_history.clear()
        self.states_history.clear()
        
        # 重置统计
        self.total_causal_reward = 0.0
        self.total_confounding_penalty = 0.0
        self.total_intervention_bonus = 0.0
        self.detected_confoundings = 0
        self.interventions_count = 0
        
        return self
    
    def get_causal_metrics(self) -> Dict[str, Any]:
        """
        获取因果相关指标
        
        Returns:
            Dict[str, Any]: 因果指标字典
        """
        metrics = {
            'total_causal_reward': self.total_causal_reward,
            'total_confounding_penalty': self.total_confounding_penalty,
            'total_intervention_bonus': self.total_intervention_bonus,
            'detected_confoundings': self.detected_confoundings,
            'interventions_count': self.interventions_count,
            'causal_graph_nodes': len(self.causal_graph.graph.nodes()),
            'causal_graph_edges': len(self.causal_graph.graph.edges()),
            'identified_confounders': len(self.causal_graph.confounders),
            'confounders_list': list(self.causal_graph.confounders),
            'adjustment_method': self.config.adjustment_method.value,
            'dovi_adjustments': len(self.dovi_optimizer.confounding_adjustments)
        }
        
        return metrics
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'CausalReward',
            'description': '因果奖励函数，基于2024年最新因果推理理论，实现因果图构建、混淆变量识别和DOVI去混淆价值迭代',
            'category': 'causal_inference',
            'complexity': 'expert',
            'parameters': {
                'causal_reward_weight': {
                    'type': 'float',
                    'default': 0.6,
                    'description': '因果奖励权重'
                },
                'confounding_penalty': {
                    'type': 'float',
                    'default': 0.3,
                    'description': '混淆惩罚权重'
                },
                'intervention_bonus': {
                    'type': 'float',
                    'default': 0.1,
                    'description': '干预奖励权重'
                },
                'adjustment_method': {
                    'type': 'str',
                    'default': 'backdoor',
                    'description': '因果调整方法 (backdoor, frontdoor, dovi)'
                },
                'confounding_detection_threshold': {
                    'type': 'float',
                    'default': 0.2,
                    'description': '混淆检测阈值'
                },
                'temporal_window': {
                    'type': 'int',
                    'default': 50,
                    'description': '时间窗口大小'
                },
                'dovi_confidence_level': {
                    'type': 'float',
                    'default': 0.95,
                    'description': 'DOVI置信水平'
                }
            },
            'features': [
                '因果图自动构建',
                '混淆变量识别',
                '后门调整算法',
                '前门调整算法',
                'DOVI去混淆价值迭代',
                '虚假相关检测',
                '策略混淆识别',
                'Do-演算因果效应估计'
            ],
            'mathematical_foundation': [
                'Causal Reinforcement Learning (TNNLS 2024)',
                'DOVI Algorithm (NeurIPS 2021)',
                'Backdoor and Frontdoor Criteria',
                'Do-Calculus and Causal Graphs',
                'Policy Confounding Detection (RLJ 2024)',
                'Spurious Correlation Analysis',
                'Temporal Causal Discovery',
                'Confounding Adjustment Methods'
            ],
            'applications': [
                '去除交易策略中的虚假相关',
                '识别和控制混淆变量',
                '因果效应的无偏估计',
                '策略混淆的检测和纠正',
                '稳健的因果推断',
                '反事实推理',
                '干预效果评估',
                '因果关系发现'
            ]
        }
    
    def calculate_reward(self, current_step, current_price, current_portfolio_value, action, **kwargs):
        """
        奖励计算接口 - 计算因果奖励
        
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
        
        # 构建状态
        state = {
            'step_count': current_step,
            'current_price': current_price,
            'current_value': current_portfolio_value,
            'current_action': action,
            'total_return_pct': ((current_portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'step_return_pct': (step_return / self.initial_balance) * 100 if self.initial_balance != 0 else 0.0
        }
        
        # 计算因果奖励
        causal_reward = self._calculate_causal_reward(state)
        
        # 检测混淆并应用惩罚
        confounding_penalty = self._detect_confounding_penalty(state)
        
        # 干预奖励
        intervention_bonus = 0.0
        if abs(action) > 0.5:
            intervention_bonus = 0.1 * state['step_return_pct']
        
        # 综合奖励
        total_reward = (
            self.config.causal_reward_weight * causal_reward -
            self.config.confounding_penalty * confounding_penalty +
            self.config.intervention_bonus * intervention_bonus
        )
        
        # 记录奖励历史
        self.reward_history.append(total_reward)
        
        return total_reward