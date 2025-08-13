"""
最终收益与最大回撤结合的奖励函数

基于Calmar比率理论，将最终收益与最大回撤相结合，为强化学习交易系统
提供平衡收益最大化与风险控制的复合奖励信号。

核心特性：
1. 实时收益跟踪：持续监控总收益率变化
2. 动态最大回撤计算：实时更新峰值和回撤幅度  
3. 分层奖励结构：基础收益奖励 + 回撤惩罚 + Calmar比率奖励
4. 自适应权重：根据训练阶段调整收益与风险的权重
5. 风险分级：不同回撤水平的差异化惩罚

数学基础：
- 总收益率：total_return = (current_value - initial_value) / initial_value
- 最大回撤：max_drawdown = (peak_value - current_value) / peak_value
- Calmar比率：calmar_ratio = annualized_return / |max_drawdown|
- 复合奖励：reward = return_component - drawdown_penalty + calmar_bonus
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from .base_reward import BaseRewardScheme


class ReturnDrawdownReward(BaseRewardScheme):
    """
    最终收益与最大回撤结合的奖励函数
    
    该奖励函数将收益最大化与回撤控制相结合，通过多层次的奖励结构
    鼓励策略在追求高收益的同时有效控制下行风险。
    
    适用于需要平衡收益与风险的量化交易策略，特别是长期投资组合管理。
    """
    
    def __init__(self, 
                 return_weight: float = 0.6,
                 drawdown_weight: float = 0.4,
                 calmar_scale: float = 10.0,
                 drawdown_tolerance: float = 0.05,
                 severe_drawdown_threshold: float = 0.15,
                 use_calmar_bonus: bool = True,
                 adaptive_weights: bool = True,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化收益-回撤奖励函数
        
        Args:
            return_weight: 收益奖励的权重，控制收益部分在总奖励中的比重
            drawdown_weight: 回撤惩罚的权重，控制风险控制的重要性
            calmar_scale: Calmar比率的缩放因子，调整Calmar奖励的量级
            drawdown_tolerance: 回撤容忍度，低于此值不进行惩罚
            severe_drawdown_threshold: 严重回撤阈值，超过此值重度惩罚
            use_calmar_bonus: 是否启用Calmar比率奖励机制
            adaptive_weights: 是否根据市场情况自适应调整权重
            initial_balance: 初始资金，用于计算收益率
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 核心参数
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.calmar_scale = calmar_scale
        self.drawdown_tolerance = drawdown_tolerance
        self.severe_drawdown_threshold = severe_drawdown_threshold
        self.use_calmar_bonus = use_calmar_bonus
        self.adaptive_weights = adaptive_weights
        
        # 状态跟踪
        self.peak_value = initial_balance  # 历史峰值
        self.current_drawdown = 0.0        # 当前回撤
        self.max_drawdown = 0.0            # 最大回撤
        self.total_return = 0.0            # 总收益率
        
        # 历史记录
        self.value_history = []            # 价值历史
        self.drawdown_history = []         # 回撤历史
        self.return_history = []           # 收益历史
        
        # 自适应权重
        self.base_return_weight = return_weight
        self.base_drawdown_weight = drawdown_weight
        self.weight_adjustment_window = 100
        
        # 性能统计
        self.positive_return_count = 0
        self.negative_return_count = 0
        self.severe_drawdown_count = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 基于收益-回撤的复合奖励
        
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
            if len(self.value_history) == 0:
                self.peak_value = portfolio_value
                self.value_history.append(portfolio_value)
                return 0.0
            
            # 更新状态
            self._update_portfolio_state(portfolio_value)
            
            # 计算收益组件
            return_component = self._calculate_return_component()
            
            # 计算回撤惩罚
            drawdown_penalty = self._calculate_drawdown_penalty()
            
            # 计算Calmar比率奖励
            calmar_bonus = self._calculate_calmar_bonus() if self.use_calmar_bonus else 0.0
            
            # 自适应权重调整
            if self.adaptive_weights:
                self._adjust_adaptive_weights()
            
            # 组合最终奖励
            final_reward = (
                self.return_weight * return_component
                - self.drawdown_weight * drawdown_penalty
                + calmar_bonus
            )
            
            # 更新历史记录
            self._update_tracking_variables(portfolio_value, final_reward)
            self.step_count += 1
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"ReturnDrawdown奖励计算异常: {e}")
            return 0.0
        
    def reward(self, env) -> float:
        """
        计算基于收益-回撤的复合奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            # 获取当前投资组合价值
            current_value = self.get_portfolio_value(env)
            
            # 第一步初始化
            if len(self.value_history) == 0:
                self.peak_value = current_value
                self.value_history.append(current_value)
                return 0.0
            
            # 更新状态
            self._update_portfolio_state(current_value)
            
            # 计算收益组件
            return_component = self._calculate_return_component()
            
            # 计算回撤惩罚
            drawdown_penalty = self._calculate_drawdown_penalty()
            
            # 计算Calmar比率奖励
            calmar_bonus = self._calculate_calmar_bonus() if self.use_calmar_bonus else 0.0
            
            # 自适应权重调整
            if self.adaptive_weights:
                self._adjust_adaptive_weights()
            
            # 组合最终奖励
            final_reward = (
                self.return_weight * return_component
                - self.drawdown_weight * drawdown_penalty
                + calmar_bonus
            )
            
            # 更新历史记录
            self._update_tracking_variables(current_value, final_reward)
            self.step_count += 1
            
            # 记录重要信息
            if self.step_count % 100 == 0 or abs(final_reward) > 5:
                self.logger.info(
                    f"[ReturnDrawdown] 步骤{self.step_count}: "
                    f"收益={self.total_return:.4f}, 回撤={self.current_drawdown:.4f}, "
                    f"奖励={final_reward:.4f} (收益:{return_component:.2f}, "
                    f"回撤惩罚:{drawdown_penalty:.2f}, Calmar:{calmar_bonus:.2f})"
                )
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"ReturnDrawdown奖励计算异常: {e}")
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
            
            if len(self.value_history) == 0:
                self.peak_value = current_value
                self.value_history.append(current_value)
                return 0.0
            
            # 更新状态
            self._update_portfolio_state(current_value)
            
            # 简化的奖励计算（适用于portfolio接口）
            return_component = self.total_return * 100  # 将百分比转换为更大的数值
            drawdown_penalty = self.current_drawdown * 200  # 加重回撤惩罚
            
            final_reward = (
                self.return_weight * return_component
                - self.drawdown_weight * drawdown_penalty
            )
            
            self.step_count += 1
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"Portfolio奖励计算异常: {e}")
            return 0.0
    
    def _update_portfolio_state(self, current_value: float) -> None:
        """
        更新投资组合状态信息
        
        Args:
            current_value: 当前投资组合价值
        """
        # 更新峰值
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # 计算当前回撤
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0
        
        # 更新最大回撤
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        # 计算总收益率
        if self.initial_balance > 0:
            self.total_return = (current_value - self.initial_balance) / self.initial_balance
        else:
            self.total_return = 0.0
        
        # 更新历史记录
        self.value_history.append(current_value)
        self.drawdown_history.append(self.current_drawdown)
        self.return_history.append(self.total_return)
        
        # 保持历史记录在合理长度
        if len(self.value_history) > 1000:
            self.value_history.pop(0)
            self.drawdown_history.pop(0)
            self.return_history.pop(0)
    
    def _calculate_return_component(self) -> float:
        """
        计算收益奖励组件
        
        Returns:
            float: 收益奖励值
        """
        # 基础收益奖励：总收益率 * 100（放大到合适的数值范围）
        base_return_reward = self.total_return * 100
        
        # 渐进式奖励：正收益加成，负收益惩罚
        if self.total_return > 0:
            # 正收益：给予额外奖励，鼓励盈利
            progressive_bonus = min(self.total_return * 50, 10.0)  # 最大额外奖励10
            self.positive_return_count += 1
        else:
            # 负收益：额外惩罚，但不过度惩罚以免影响探索
            progressive_bonus = max(self.total_return * 30, -5.0)  # 最大额外惩罚5
            self.negative_return_count += 1
        
        return base_return_reward + progressive_bonus
    
    def _calculate_drawdown_penalty(self) -> float:
        """
        计算回撤惩罚组件
        
        Returns:
            float: 回撤惩罚值
        """
        # 如果回撤在容忍范围内，不进行惩罚
        if self.current_drawdown <= self.drawdown_tolerance:
            return 0.0
        
        # 计算超出容忍度的回撤
        excess_drawdown = self.current_drawdown - self.drawdown_tolerance
        
        # 分级惩罚机制
        if self.current_drawdown <= self.severe_drawdown_threshold:
            # 一般回撤：线性惩罚
            penalty = excess_drawdown * 100
        else:
            # 严重回撤：指数惩罚
            normal_penalty = (self.severe_drawdown_threshold - self.drawdown_tolerance) * 100
            severe_excess = self.current_drawdown - self.severe_drawdown_threshold
            severe_penalty = severe_excess * 300  # 更严厉的惩罚
            penalty = normal_penalty + severe_penalty
            self.severe_drawdown_count += 1
        
        # 历史最大回撤额外惩罚
        if self.current_drawdown >= self.max_drawdown * 0.9:
            penalty *= 1.5  # 接近历史最大回撤时加重惩罚
        
        return penalty
    
    def _calculate_calmar_bonus(self) -> float:
        """
        计算Calmar比率奖励组件
        
        Returns:
            float: Calmar比率奖励值
        """
        try:
            # 需要足够的历史数据计算年化收益
            if len(self.return_history) < 50:
                return 0.0
            
            # 计算年化收益率（简化为最近收益的平均值 * 252）
            recent_returns = self.return_history[-50:]
            if len(recent_returns) < 2:
                return 0.0
            
            # 计算期间收益变化
            period_returns = np.diff(recent_returns)
            if len(period_returns) == 0:
                return 0.0
            
            avg_period_return = np.mean(period_returns)
            annualized_return = avg_period_return * 252  # 假设一年252个交易日
            
            # 避免除零错误
            if self.max_drawdown <= 1e-6:
                # 没有显著回撤时，给予小幅Calmar奖励
                return min(abs(annualized_return) * self.calmar_scale * 0.1, 2.0)
            
            # 计算Calmar比率
            calmar_ratio = annualized_return / self.max_drawdown
            
            # 缩放到合适的奖励范围
            calmar_bonus = calmar_ratio * self.calmar_scale
            
            # 限制Calmar奖励的范围，避免极端值
            calmar_bonus = np.clip(calmar_bonus, -5.0, 10.0)
            
            return float(calmar_bonus)
            
        except Exception as e:
            self.logger.warning(f"Calmar比率计算异常: {e}")
            return 0.0
    
    def _adjust_adaptive_weights(self) -> None:
        """
        根据市场表现自适应调整权重
        """
        if self.step_count < self.weight_adjustment_window:
            return
        
        try:
            # 计算最近的表现统计
            recent_steps = min(self.weight_adjustment_window, len(self.return_history))
            recent_returns = self.return_history[-recent_steps:]
            recent_drawdowns = self.drawdown_history[-recent_steps:]
            
            # 计算波动性和趋势
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
            avg_return = np.mean(recent_returns)
            avg_drawdown = np.mean(recent_drawdowns)
            
            # 动态权重调整逻辑
            if avg_return > 0 and avg_drawdown < self.drawdown_tolerance:
                # 表现良好：稍微增加收益权重
                adjustment_factor = 1.1
            elif avg_return < 0 or avg_drawdown > self.severe_drawdown_threshold:
                # 表现不佳：增加风险控制权重
                adjustment_factor = 0.9
            else:
                # 表现一般：保持平衡
                adjustment_factor = 1.0
            
            # 应用调整，但保持在合理范围内
            self.return_weight = np.clip(
                self.base_return_weight * adjustment_factor, 
                0.3, 0.8
            )
            self.drawdown_weight = np.clip(
                self.base_drawdown_weight / adjustment_factor, 
                0.2, 0.7
            )
            
            # 确保权重总和合理
            total_weight = self.return_weight + self.drawdown_weight
            if total_weight > 1.2:
                factor = 1.0 / total_weight
                self.return_weight *= factor
                self.drawdown_weight *= factor
            
        except Exception as e:
            self.logger.warning(f"自适应权重调整异常: {e}")
    
    def _update_tracking_variables(self, current_value: float, reward: float) -> None:
        """
        更新跟踪变量和统计信息
        
        Args:
            current_value: 当前投资组合价值
            reward: 当前奖励值
        """
        # 更新奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
    
    def reset(self) -> 'ReturnDrawdownReward':
        """
        重置奖励函数状态
        
        Returns:
            ReturnDrawdownReward: 返回self以支持链式调用
        """
        # 记录回合性能
        if len(self.value_history) > 0:
            final_value = self.value_history[-1]
            final_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
            calmar_ratio = final_return / self.max_drawdown if self.max_drawdown > 1e-6 else 0
            
            self.logger.info(
                f"[ReturnDrawdown回合{self.episode_count}结束] "
                f"最终收益率: {final_return:.4f}, "
                f"最大回撤: {self.max_drawdown:.4f}, "
                f"Calmar比率: {calmar_ratio:.4f}, "
                f"步数: {self.step_count}, "
                f"正收益次数: {self.positive_return_count}, "
                f"严重回撤次数: {self.severe_drawdown_count}"
            )
        
        # 调用父类reset
        super().reset()
        
        # 重置状态
        self.peak_value = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.total_return = 0.0
        
        # 保留部分历史用于连续学习
        if len(self.value_history) > 100:
            self.value_history = self.value_history[-20:]
            self.drawdown_history = self.drawdown_history[-20:]
            self.return_history = self.return_history[-20:]
        else:
            self.value_history = []
            self.drawdown_history = []
            self.return_history = []
        
        # 重置统计计数器
        self.positive_return_count = 0
        self.negative_return_count = 0
        self.severe_drawdown_count = 0
        
        # 重置自适应权重
        self.return_weight = self.base_return_weight
        self.drawdown_weight = self.base_drawdown_weight
        
        return self
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要，包含ReturnDrawdown特有指标
        
        Returns:
            Dict[str, Any]: 性能摘要信息
        """
        base_summary = super().get_performance_summary()
        
        # 计算Calmar比率
        calmar_ratio = 0.0
        if self.max_drawdown > 1e-6 and len(self.return_history) > 0:
            try:
                final_return = self.return_history[-1] if self.return_history else 0
                calmar_ratio = final_return / self.max_drawdown
            except:
                pass
        
        # 添加ReturnDrawdown特有指标
        return_drawdown_metrics = {
            'total_return': self.total_return,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': calmar_ratio,
            'peak_value': self.peak_value,
            'positive_return_ratio': self.positive_return_count / max(1, self.step_count),
            'severe_drawdown_count': self.severe_drawdown_count,
            'current_return_weight': self.return_weight,
            'current_drawdown_weight': self.drawdown_weight,
            'avg_drawdown': np.mean(self.drawdown_history) if self.drawdown_history else 0.0,
            'drawdown_volatility': np.std(self.drawdown_history) if len(self.drawdown_history) > 1 else 0.0
        }
        
        base_summary.update(return_drawdown_metrics)
        return base_summary
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'ReturnDrawdownReward',
            'description': '基于最终收益与最大回撤结合的奖励函数，平衡收益最大化与下行风险控制',
            'category': 'risk_return_balanced',
            'parameters': {
                'return_weight': {
                    'type': 'float',
                    'default': 0.6,
                    'description': '收益奖励的权重，控制收益部分在总奖励中的比重'
                },
                'drawdown_weight': {
                    'type': 'float',
                    'default': 0.4,
                    'description': '回撤惩罚的权重，控制风险控制的重要性'
                },
                'calmar_scale': {
                    'type': 'float',
                    'default': 10.0,
                    'description': 'Calmar比率的缩放因子，调整Calmar奖励的量级'
                },
                'drawdown_tolerance': {
                    'type': 'float',
                    'default': 0.05,
                    'description': '回撤容忍度，低于此值不进行惩罚'
                },
                'severe_drawdown_threshold': {
                    'type': 'float',
                    'default': 0.15,
                    'description': '严重回撤阈值，超过此值重度惩罚'
                },
                'use_calmar_bonus': {
                    'type': 'bool',
                    'default': True,
                    'description': '是否启用Calmar比率奖励机制'
                },
                'adaptive_weights': {
                    'type': 'bool',
                    'default': True,
                    'description': '是否根据市场情况自适应调整权重'
                },
                'initial_balance': {
                    'type': 'float',
                    'default': 10000.0,
                    'description': '初始资金，用于计算收益率'
                }
            },
            'advantages': [
                '平衡收益与风险的复合奖励结构',
                '基于Calmar比率的理论基础',
                '自适应权重调整机制',
                '分级回撤惩罚系统',
                '实时风险监控'
            ],
            'use_cases': [
                '平衡型投资策略',
                '风险控制型交易',
                '长期投资组合管理',
                '机构资金管理',
                '回撤敏感型策略'
            ],
            'mathematical_foundation': 'Calmar比率理论，结合动态回撤控制和收益优化',
            'complexity': 'intermediate'
        }