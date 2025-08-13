"""
专家委员会多目标强化学习奖励函数

基于多目标强化学习(MORL)理论和Pareto前沿优化，构建专家委员会系统，
为强化学习交易策略提供多维度平衡的复合奖励信号。

核心创新：
1. 五位专家委员会：收益、风险、效率、稳定、趋势专家并行决策
2. Pareto前沿学习：寻找多目标权衡的最优解集
3. 动态权重分配：基于专家历史表现的自适应权重调整
4. Tchebycheff标量化：将多目标优化转化为单目标问题
5. 专家竞争机制：优秀专家获得更多决策权重

数学基础：
- 多目标奖励向量：R_t = [r_return, r_risk, r_efficiency, r_stability, r_trend]
- Tchebycheff标量化：R_final = min_i(w_i × (r_i - z_i*)) + ρ × Σ(w_i × r_i)
- 动态权重更新：w_i = softmax(performance_score_i / temperature)
- Pareto支配关系：solution A dominates B if A ≥ B in all objectives and A > B in at least one
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from enum import Enum
from .base_reward import BaseRewardScheme


class ExpertType(Enum):
    """专家类型枚举"""
    RETURN = "return_expert"          # 收益专家
    RISK = "risk_expert"              # 风险专家  
    EFFICIENCY = "efficiency_expert"   # 效率专家
    STABILITY = "stability_expert"     # 稳定专家
    TREND = "trend_expert"            # 趋势专家


class Expert:
    """单个专家基类"""
    
    def __init__(self, expert_type: ExpertType, weight: float = 0.2):
        self.expert_type = expert_type
        self.weight = weight
        self.performance_history = deque(maxlen=100)
        self.success_rate = 0.0
        self.confidence_score = 0.5
        
    def calculate_reward(self, portfolio_data: Dict[str, Any]) -> float:
        """计算专家特定的奖励值"""
        raise NotImplementedError
    
    def update_performance(self, actual_reward: float, predicted_reward: float):
        """更新专家表现"""
        error = abs(actual_reward - predicted_reward)
        success = 1.0 if error < 0.1 else 0.0
        self.performance_history.append(success)
        self.success_rate = np.mean(list(self.performance_history)) if self.performance_history else 0.0


class ReturnExpert(Expert):
    """收益专家：专注最大化绝对收益"""
    
    def __init__(self, weight: float = 0.2, return_scale: float = 100.0):
        super().__init__(ExpertType.RETURN, weight)
        self.return_scale = return_scale
        
    def calculate_reward(self, portfolio_data: Dict[str, Any]) -> float:
        """计算收益奖励"""
        total_return = portfolio_data.get('total_return', 0.0)
        # 对数增长奖励：鼓励持续盈利
        if total_return > 0:
            reward = np.log(1 + total_return) * self.return_scale
        else:
            reward = total_return * self.return_scale * 0.5  # 温和惩罚亏损
        return float(reward)


class RiskExpert(Expert):
    """风险专家：专注最小化风险和回撤"""
    
    def __init__(self, weight: float = 0.2, risk_penalty: float = 200.0, volatility_penalty: float = 50.0):
        super().__init__(ExpertType.RISK, weight)
        self.risk_penalty = risk_penalty
        self.volatility_penalty = volatility_penalty
        
    def calculate_reward(self, portfolio_data: Dict[str, Any]) -> float:
        """计算风险奖励"""
        max_drawdown = portfolio_data.get('max_drawdown', 0.0)
        volatility = portfolio_data.get('volatility', 0.0)
        
        # 回撤惩罚：指数递增
        drawdown_penalty = (max_drawdown ** 2) * self.risk_penalty
        
        # 波动性惩罚：线性惩罚
        volatility_penalty = volatility * self.volatility_penalty
        
        # 风险专家给出负奖励（惩罚）
        reward = -(drawdown_penalty + volatility_penalty)
        return float(reward)


class EfficiencyExpert(Expert):
    """效率专家：专注风险调整收益（夏普比率）"""
    
    def __init__(self, weight: float = 0.2, efficiency_scale: float = 50.0, min_periods: int = 10):
        super().__init__(ExpertType.EFFICIENCY, weight)
        self.efficiency_scale = efficiency_scale
        self.min_periods = min_periods
        
    def calculate_reward(self, portfolio_data: Dict[str, Any]) -> float:
        """计算效率奖励"""
        returns_history = portfolio_data.get('returns_history', [])
        risk_free_rate = portfolio_data.get('risk_free_rate', 0.02)
        
        if len(returns_history) < self.min_periods:
            return 0.0
        
        # 计算夏普比率
        returns_array = np.array(returns_history)
        excess_returns = returns_array - risk_free_rate / 252
        
        if len(excess_returns) < 2:
            return 0.0
            
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess <= 1e-6:
            return 0.0
            
        sharpe_ratio = mean_excess / std_excess
        reward = sharpe_ratio * self.efficiency_scale
        
        # 限制极端值
        reward = np.clip(reward, -20.0, 20.0)
        return float(reward)


class StabilityExpert(Expert):
    """稳定专家：专注降低交易频率和成本"""
    
    def __init__(self, weight: float = 0.2, stability_bonus: float = 10.0, frequency_penalty: float = 1.0):
        super().__init__(ExpertType.STABILITY, weight)
        self.stability_bonus = stability_bonus
        self.frequency_penalty = frequency_penalty
        self.prev_portfolio_composition = None
        
    def calculate_reward(self, portfolio_data: Dict[str, Any]) -> float:
        """计算稳定性奖励"""
        transaction_count = portfolio_data.get('transaction_count', 0)
        portfolio_composition = portfolio_data.get('portfolio_composition', {})
        
        # 交易频率惩罚
        frequency_penalty = transaction_count * self.frequency_penalty
        
        # 持仓稳定性奖励
        stability_bonus = 0.0
        if self.prev_portfolio_composition is not None:
            # 计算组合变化程度
            composition_change = self._calculate_composition_change(
                self.prev_portfolio_composition, portfolio_composition
            )
            stability_bonus = (1.0 - composition_change) * self.stability_bonus
        
        self.prev_portfolio_composition = portfolio_composition.copy()
        
        reward = stability_bonus - frequency_penalty
        return float(reward)
    
    def _calculate_composition_change(self, prev_comp: Dict, curr_comp: Dict) -> float:
        """计算投资组合组成变化程度"""
        try:
            all_assets = set(prev_comp.keys()) | set(curr_comp.keys())
            if not all_assets:
                return 0.0
                
            total_change = 0.0
            for asset in all_assets:
                prev_weight = prev_comp.get(asset, 0.0)
                curr_weight = curr_comp.get(asset, 0.0)
                total_change += abs(curr_weight - prev_weight)
            
            return min(total_change / 2.0, 1.0)  # 归一化到[0, 1]
        except Exception:
            return 0.0


class TrendExpert(Expert):
    """趋势专家：专注捕捉和跟随市场趋势"""
    
    def __init__(self, weight: float = 0.2, trend_scale: float = 30.0, momentum_window: int = 20):
        super().__init__(ExpertType.TREND, weight)
        self.trend_scale = trend_scale
        self.momentum_window = momentum_window
        
    def calculate_reward(self, portfolio_data: Dict[str, Any]) -> float:
        """计算趋势奖励"""
        value_history = portfolio_data.get('value_history', [])
        
        if len(value_history) < self.momentum_window:
            return 0.0
        
        # 计算动量得分
        momentum_score = self._calculate_momentum(value_history)
        
        # 趋势强度奖励
        trend_strength = abs(momentum_score)
        direction_bonus = 1.0 if momentum_score > 0 else 0.5  # 上涨趋势更好
        
        reward = trend_strength * direction_bonus * self.trend_scale
        return float(reward)
    
    def _calculate_momentum(self, value_history: List[float]) -> float:
        """计算动量得分"""
        try:
            if len(value_history) < self.momentum_window:
                return 0.0
            
            recent_values = value_history[-self.momentum_window:]
            
            # 线性回归斜率作为动量指标
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # 归一化到[-1, 1]
            max_value = max(recent_values)
            if max_value > 0:
                normalized_slope = slope / max_value * len(recent_values)
                return np.clip(normalized_slope, -1.0, 1.0)
            return 0.0
        except Exception:
            return 0.0


class ExpertCommitteeReward(BaseRewardScheme):
    """
    专家委员会多目标强化学习奖励函数
    
    该奖励函数构建了一个由5位专家组成的委员会，每位专家专注不同的交易目标。
    通过多目标优化和Pareto前沿学习，寻找收益、风险、效率、稳定性和趋势
    之间的最优平衡点。
    
    特别适用于需要平衡多个冲突目标的复杂交易策略。
    """
    
    def __init__(self,
                 initial_weights: Optional[Dict[str, float]] = None,
                 weight_adaptation_rate: float = 0.1,
                 tchebycheff_rho: float = 0.1,
                 temperature: float = 1.0,
                 pareto_archive_size: int = 100,
                 update_frequency: int = 50,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化专家委员会奖励函数
        
        Args:
            initial_weights: 专家初始权重分配
            weight_adaptation_rate: 权重自适应学习率
            tchebycheff_rho: Tchebycheff标量化参数
            temperature: Softmax温度参数，控制权重更新的激进程度
            pareto_archive_size: Pareto前沿解集大小
            update_frequency: 权重更新频率
            initial_balance: 初始资金
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 专家委员会设置
        self.experts = self._initialize_experts(initial_weights or {})
        self.weight_adaptation_rate = weight_adaptation_rate
        self.tchebycheff_rho = tchebycheff_rho
        self.temperature = temperature
        self.update_frequency = update_frequency
        
        # Pareto前沿管理
        self.pareto_archive = deque(maxlen=pareto_archive_size)
        self.reference_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # 理想点
        
        # 状态跟踪
        self.portfolio_value_history = deque(maxlen=1000)
        self.returns_history = deque(maxlen=252)  # 一年的交易日
        self.transaction_count = 0
        self.max_drawdown = 0.0
        self.peak_value = initial_balance
        
        # 多目标历史
        self.objective_values_history = deque(maxlen=1000)
        self.weights_history = deque(maxlen=1000)
        
        # 性能统计
        self.pareto_improvements = 0
        self.weight_updates = 0
        self.expert_competitions = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 专家委员会奖励
        
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
            # 构建专家数据字典
            portfolio_data = {
                'portfolio_value': portfolio_value,
                'action': action,
                'price': price,
                'total_return': (portfolio_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0.0,
                'max_drawdown': portfolio_info.get('max_drawdown', 0.0),
                'volatility': portfolio_info.get('volatility', 0.0),
                'portfolio_info': portfolio_info,
                'trade_info': trade_info,
                'step': step,
                **kwargs
            }
            
            # 使用现有的专家委员会逻辑
            reward = self._calculate_committee_reward(portfolio_data)
            self.step_count += 1
            
            return float(reward)
            
        except Exception as e:
            self.logger.error(f"ExpertCommittee奖励计算异常: {e}")
            return 0.0
        
    def _initialize_experts(self, initial_weights: Dict[str, float]) -> Dict[str, Expert]:
        """初始化专家委员会"""
        # 默认等权重分配
        default_weight = 1.0 / 5
        
        experts = {
            'return': ReturnExpert(
                weight=initial_weights.get('return', default_weight)
            ),
            'risk': RiskExpert(
                weight=initial_weights.get('risk', default_weight)
            ),
            'efficiency': EfficiencyExpert(
                weight=initial_weights.get('efficiency', default_weight)
            ),
            'stability': StabilityExpert(
                weight=initial_weights.get('stability', default_weight)
            ),
            'trend': TrendExpert(
                weight=initial_weights.get('trend', default_weight)
            )
        }
        
        # 权重归一化
        total_weight = sum(expert.weight for expert in experts.values())
        if total_weight > 0:
            for expert in experts.values():
                expert.weight /= total_weight
        
        return experts
    
    def reward(self, env) -> float:
        """
        计算专家委员会的综合奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 计算得到的综合奖励值
        """
        try:
            # 获取当前投资组合信息
            current_value = self.get_portfolio_value(env)
            portfolio_data = self._prepare_portfolio_data(current_value, env)
            
            # 各专家计算奖励
            expert_rewards = self._calculate_expert_rewards(portfolio_data)
            
            # 多目标向量
            objective_vector = np.array([
                expert_rewards['return'],
                expert_rewards['risk'], 
                expert_rewards['efficiency'],
                expert_rewards['stability'],
                expert_rewards['trend']
            ])
            
            # Tchebycheff标量化
            final_reward = self._tchebycheff_scalarization(objective_vector)
            
            # 更新Pareto前沿
            self._update_pareto_archive(objective_vector)
            
            # 动态权重更新
            if self.step_count % self.update_frequency == 0:
                self._update_expert_weights(expert_rewards)
            
            # 更新跟踪变量
            self._update_tracking_variables(current_value, objective_vector, final_reward)
            self.step_count += 1
            
            # 记录重要信息
            if self.step_count % 100 == 0 or abs(final_reward) > 10:
                self._log_committee_status(expert_rewards, final_reward)
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"ExpertCommittee奖励计算异常: {e}")
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
            
            # 简化的投资组合数据
            portfolio_data = {
                'current_value': current_value,
                'total_return': (current_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0,
                'max_drawdown': self.max_drawdown,
                'volatility': 0.02,  # 简化假设
                'returns_history': list(self.returns_history),
                'risk_free_rate': 0.02,
                'transaction_count': self.transaction_count,
                'portfolio_composition': {},
                'value_history': list(self.portfolio_value_history)
            }
            
            # 简化的专家奖励计算
            expert_rewards = self._calculate_expert_rewards(portfolio_data)
            objective_vector = np.array(list(expert_rewards.values()))
            final_reward = self._tchebycheff_scalarization(objective_vector)
            
            self.step_count += 1
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"Portfolio奖励计算异常: {e}")
            return 0.0
    
    def _prepare_portfolio_data(self, current_value: float, env) -> Dict[str, Any]:
        """准备投资组合数据"""
        # 更新历史数据
        self.portfolio_value_history.append(current_value)
        
        # 计算收益率
        if len(self.portfolio_value_history) > 1:
            prev_value = self.portfolio_value_history[-2]
            if prev_value > 0:
                period_return = (current_value - prev_value) / prev_value
                self.returns_history.append(period_return)
        
        # 更新峰值和回撤
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 构建数据字典
        portfolio_data = {
            'current_value': current_value,
            'total_return': (current_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'volatility': np.std(list(self.returns_history)) if len(self.returns_history) > 1 else 0.0,
            'returns_history': list(self.returns_history),
            'risk_free_rate': 0.02,  # 可配置
            'transaction_count': self.transaction_count,
            'portfolio_composition': {},  # 简化处理
            'value_history': list(self.portfolio_value_history)
        }
        
        return portfolio_data
    
    def _calculate_expert_rewards(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """计算各专家的奖励"""
        expert_rewards = {}
        
        for name, expert in self.experts.items():
            try:
                reward = expert.calculate_reward(portfolio_data)
                expert_rewards[name] = reward
            except Exception as e:
                self.logger.warning(f"专家{name}奖励计算异常: {e}")
                expert_rewards[name] = 0.0
        
        return expert_rewards
    
    def _tchebycheff_scalarization(self, objective_vector: np.ndarray) -> float:
        """
        Tchebycheff标量化方法
        
        Args:
            objective_vector: 多目标奖励向量
            
        Returns:
            float: 标量化后的奖励值
        """
        try:
            # 获取当前权重
            weights = np.array([expert.weight for expert in self.experts.values()])
            
            # Tchebycheff函数：min_i(w_i * (f_i - z_i*)) + ρ * Σ(w_i * f_i)
            # 其中z_i*是参考点（理想点）
            
            # 计算加权差异
            weighted_diff = weights * (objective_vector - self.reference_point)
            
            # Tchebycheff项：最小的加权差异
            tchebycheff_term = np.min(weighted_diff)
            
            # 增强项：加权和
            augmentation_term = self.tchebycheff_rho * np.sum(weights * objective_vector)
            
            # 最终奖励
            final_reward = tchebycheff_term + augmentation_term
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.warning(f"Tchebycheff标量化异常: {e}")
            # 回退到简单加权和
            weights = np.array([expert.weight for expert in self.experts.values()])
            return float(np.sum(weights * objective_vector))
    
    def _update_pareto_archive(self, objective_vector: np.ndarray) -> None:
        """更新Pareto前沿解集"""
        try:
            # 检查是否被现有解支配
            is_dominated = False
            dominated_indices = []
            
            for i, archived_solution in enumerate(self.pareto_archive):
                if self._dominates(archived_solution, objective_vector):
                    is_dominated = True
                    break
                elif self._dominates(objective_vector, archived_solution):
                    dominated_indices.append(i)
            
            # 如果不被支配，加入解集并移除被支配的解
            if not is_dominated:
                # 移除被新解支配的解
                for i in reversed(dominated_indices):
                    del self.pareto_archive[i]
                
                # 加入新解
                self.pareto_archive.append(objective_vector.copy())
                self.pareto_improvements += 1
                
                # 更新参考点（理想点）
                if len(self.pareto_archive) > 1:
                    archive_array = np.array(list(self.pareto_archive))
                    self.reference_point = np.max(archive_array, axis=0)
                    
        except Exception as e:
            self.logger.warning(f"Pareto前沿更新异常: {e}")
    
    def _dominates(self, solution_a: np.ndarray, solution_b: np.ndarray) -> bool:
        """检查解A是否支配解B"""
        try:
            # A支配B当且仅当：A在所有目标上都不劣于B，且至少在一个目标上严格优于B
            all_better_or_equal = np.all(solution_a >= solution_b)
            at_least_one_better = np.any(solution_a > solution_b)
            return all_better_or_equal and at_least_one_better
        except Exception:
            return False
    
    def _update_expert_weights(self, expert_rewards: Dict[str, float]) -> None:
        """动态更新专家权重"""
        try:
            # 计算专家表现得分
            performance_scores = []
            
            for name, expert in self.experts.items():
                # 基于成功率和最近奖励的综合得分
                recent_reward = expert_rewards.get(name, 0.0)
                success_rate = expert.success_rate
                
                # 综合得分：成功率 + 归一化奖励
                normalized_reward = np.tanh(recent_reward / 10.0)  # 归一化到[-1, 1]
                performance_score = 0.7 * success_rate + 0.3 * (normalized_reward + 1) / 2
                
                performance_scores.append(performance_score)
                
                # 更新专家表现历史
                expert.update_performance(recent_reward, expert.confidence_score * 10.0)
            
            # Softmax权重更新
            performance_array = np.array(performance_scores)
            new_weights = self._softmax(performance_array / self.temperature)
            
            # 自适应权重更新
            old_weights = np.array([expert.weight for expert in self.experts.values()])
            updated_weights = (1 - self.weight_adaptation_rate) * old_weights + \
                            self.weight_adaptation_rate * new_weights
            
            # 应用新权重
            for i, (name, expert) in enumerate(self.experts.items()):
                expert.weight = updated_weights[i]
            
            # 记录权重历史
            weight_dict = {name: expert.weight for name, expert in self.experts.items()}
            self.weights_history.append(weight_dict.copy())
            self.weight_updates += 1
            
        except Exception as e:
            self.logger.warning(f"权重更新异常: {e}")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        try:
            exp_x = np.exp(x - np.max(x))  # 数值稳定性
            return exp_x / np.sum(exp_x)
        except Exception:
            return np.ones(len(x)) / len(x)  # 均匀分布作为回退
    
    def _update_tracking_variables(self, current_value: float, 
                                 objective_vector: np.ndarray, reward: float) -> None:
        """更新跟踪变量"""
        # 更新多目标历史
        self.objective_values_history.append(objective_vector.copy())
        
        # 更新奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
    
    def _log_committee_status(self, expert_rewards: Dict[str, float], final_reward: float) -> None:
        """记录委员会状态"""
        weights_str = ", ".join([f"{name}:{expert.weight:.3f}" 
                               for name, expert in self.experts.items()])
        rewards_str = ", ".join([f"{name}:{reward:.3f}" 
                               for name, reward in expert_rewards.items()])
        
        self.logger.info(
            f"[ExpertCommittee] 步骤{self.step_count}: "
            f"最终奖励={final_reward:.4f}, "
            f"权重=[{weights_str}], "
            f"专家奖励=[{rewards_str}], "
            f"Pareto解数={len(self.pareto_archive)}"
        )
    
    def reset(self) -> 'ExpertCommitteeReward':
        """重置奖励函数状态"""
        # 记录回合性能
        if len(self.portfolio_value_history) > 0:
            final_value = self.portfolio_value_history[-1]
            final_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
            avg_objectives = np.mean(list(self.objective_values_history), axis=0) if self.objective_values_history else np.zeros(5)
            
            self.logger.info(
                f"[ExpertCommittee回合{self.episode_count}结束] "
                f"最终收益率: {final_return:.4f}, "
                f"最大回撤: {self.max_drawdown:.4f}, "
                f"平均目标值: {avg_objectives}, "
                f"Pareto改进: {self.pareto_improvements}, "
                f"权重更新: {self.weight_updates}, "
                f"步数: {self.step_count}"
            )
        
        # 调用父类reset
        super().reset()
        
        # 重置状态但保留学习信息
        self.portfolio_value_history.clear()
        if len(self.returns_history) > 50:
            # 保留部分历史用于连续学习
            recent_returns = list(self.returns_history)[-20:]
            self.returns_history.clear()
            self.returns_history.extend(recent_returns)
        else:
            self.returns_history.clear()
        
        # 重置统计
        self.transaction_count = 0
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance
        self.pareto_improvements = 0
        self.weight_updates = 0
        
        # 保留Pareto前沿和专家学习成果
        # (不重置pareto_archive和expert performance_history)
        
        return self
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        base_summary = super().get_performance_summary()
        
        # 计算专家权重统计
        current_weights = {name: expert.weight for name, expert in self.experts.items()}
        expert_success_rates = {name: expert.success_rate for name, expert in self.experts.items()}
        
        # 多目标统计
        if self.objective_values_history:
            objectives_array = np.array(list(self.objective_values_history))
            avg_objectives = np.mean(objectives_array, axis=0)
            std_objectives = np.std(objectives_array, axis=0)
        else:
            avg_objectives = np.zeros(5)
            std_objectives = np.zeros(5)
        
        # 专家委员会特有指标
        committee_metrics = {
            'expert_weights': current_weights,
            'expert_success_rates': expert_success_rates,
            'avg_return_objective': avg_objectives[0],
            'avg_risk_objective': avg_objectives[1],
            'avg_efficiency_objective': avg_objectives[2],
            'avg_stability_objective': avg_objectives[3],
            'avg_trend_objective': avg_objectives[4],
            'objective_volatility': std_objectives.tolist(),
            'pareto_archive_size': len(self.pareto_archive),
            'pareto_improvements': self.pareto_improvements,
            'weight_updates': self.weight_updates,
            'committee_diversity': np.std(list(current_weights.values())),
            'multi_objective_balance': 1.0 - np.std(avg_objectives) / (np.mean(np.abs(avg_objectives)) + 1e-6)
        }
        
        base_summary.update(committee_metrics)
        return base_summary
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': 'ExpertCommitteeReward',
            'description': '专家委员会多目标强化学习奖励函数，通过5位专家的协作决策实现多目标平衡优化',
            'category': 'multi_objective_expert',
            'parameters': {
                'initial_weights': {
                    'type': 'dict',
                    'default': {'return': 0.2, 'risk': 0.2, 'efficiency': 0.2, 'stability': 0.2, 'trend': 0.2},
                    'description': '专家初始权重分配字典'
                },
                'weight_adaptation_rate': {
                    'type': 'float',
                    'default': 0.1,
                    'description': '权重自适应学习率，控制权重更新速度'
                },
                'tchebycheff_rho': {
                    'type': 'float',
                    'default': 0.1,
                    'description': 'Tchebycheff标量化增强参数'
                },
                'temperature': {
                    'type': 'float',
                    'default': 1.0,
                    'description': 'Softmax温度参数，控制权重更新的激进程度'
                },
                'pareto_archive_size': {
                    'type': 'int',
                    'default': 100,
                    'description': 'Pareto前沿解集最大大小'
                },
                'update_frequency': {
                    'type': 'int',
                    'default': 50,
                    'description': '权重更新频率（步数）'
                },
                'initial_balance': {
                    'type': 'float',
                    'default': 10000.0,
                    'description': '初始资金，用于收益率计算'
                }
            },
            'experts': {
                'return_expert': '收益专家：专注最大化绝对收益',
                'risk_expert': '风险专家：专注最小化风险和回撤',
                'efficiency_expert': '效率专家：专注风险调整收益（夏普比率）',
                'stability_expert': '稳定专家：专注降低交易频率和成本',
                'trend_expert': '趋势专家：专注捕捉和跟随市场趋势'
            },
            'advantages': [
                '多目标平衡优化',
                'Pareto前沿学习',
                '动态专家权重自适应',
                'Tchebycheff标量化方法',
                '专家竞争机制',
                '连续学习能力'
            ],
            'use_cases': [
                '复杂多目标交易策略',
                '机构级投资组合管理',
                '风险与收益平衡优化',
                '自适应策略学习',
                '多专家协作决策系统'
            ],
            'mathematical_foundation': 'MORL理论、Pareto最优、Tchebycheff标量化、专家系统集成',
            'complexity': 'expert'
        }