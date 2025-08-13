"""
动态时间尺度索提诺比率（DTS）奖励函数

基于索提诺比率理论的创新扩展，引入动态时间尺度适应机制，
为强化学习交易系统提供时变风险调整奖励信号。

核心创新：
1. 自适应窗口大小：根据市场状态调整计算窗口
2. 时间衰减加权：近期数据权重更高
3. 波动性感知：高波动期间增加稳定性要求
4. 趋势识别：区分上升、下降和横盘市场
5. 多尺度融合：结合短期反应性和长期稳定性

数学基础：
- 传统索提诺比率：Sortino = (R̄ - Rf) / σ⁻
- 动态窗口：window_size = f(volatility, trend, performance)
- 时间加权：weight_i = decay_factor^(n-i)
- 多尺度融合：DTS = α₁×short_term + α₂×medium_term + α₃×long_term
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import deque
from .base_reward import BaseRewardScheme


class DynamicSortinoReward(BaseRewardScheme):
    """
    动态时间尺度索提诺比率（DTS）奖励函数
    
    该奖励函数基于索提诺比率的动态扩展，通过自适应时间窗口和
    多尺度分析提供更精确的风险调整奖励信号。
    
    特别适用于需要适应不同市场状态和时间尺度的智能交易策略。
    """
    
    def __init__(self, 
                 base_window_size: int = 50,
                 min_window_size: int = 20,
                 max_window_size: int = 200,
                 risk_free_rate: float = 0.02,
                 volatility_threshold: float = 0.02,
                 trend_sensitivity: float = 0.1,
                 time_decay_factor: float = 0.95,
                 multi_scale_weights: Optional[Dict[str, float]] = None,
                 scale_factor: float = 100.0,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化动态索提诺奖励函数
        
        Args:
            base_window_size: 基础计算窗口大小，作为动态调整的基准
            min_window_size: 最小窗口大小，保证计算的统计意义
            max_window_size: 最大窗口大小，避免过度滞后
            risk_free_rate: 年化无风险收益率，用于索提诺比率计算
            volatility_threshold: 波动性阈值，触发窗口调整的临界值
            trend_sensitivity: 趋势敏感度，控制趋势对窗口大小的影响
            time_decay_factor: 时间衰减因子，控制历史数据的权重衰减
            multi_scale_weights: 多时间尺度权重配置
            scale_factor: 奖励缩放因子，调整奖励信号的量级
            initial_balance: 初始资金，用于收益率计算
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 核心参数
        self.base_window_size = base_window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.risk_free_rate = risk_free_rate / 252  # 转换为日收益率
        self.volatility_threshold = volatility_threshold
        self.trend_sensitivity = trend_sensitivity
        self.time_decay_factor = time_decay_factor
        self.scale_factor = scale_factor
        
        # 多时间尺度权重 (短期、中期、长期)
        self.multi_scale_weights = multi_scale_weights or {
            'short': 0.5,   # 短期权重（快速响应）
            'medium': 0.3,  # 中期权重（平衡）
            'long': 0.2     # 长期权重（稳定性）
        }
        
        # 数据存储
        self.returns_history = deque(maxlen=self.max_window_size * 2)  # 收益率历史
        self.value_history = deque(maxlen=self.max_window_size * 2)    # 价值历史
        self.volatility_history = deque(maxlen=100)                   # 波动率历史
        
        # 动态状态
        self.current_window_size = base_window_size
        self.market_state = 'neutral'  # 'bullish', 'bearish', 'neutral', 'volatile'
        self.trend_direction = 0.0     # -1 to 1, 负数表示下跌趋势
        self.current_volatility = 0.0
        
        # 多尺度索提诺比率
        self.short_term_sortino = 0.0
        self.medium_term_sortino = 0.0
        self.long_term_sortino = 0.0
        self.dynamic_sortino = 0.0
        
        # 性能跟踪
        self.sortino_history = deque(maxlen=1000)
        self.window_size_history = deque(maxlen=1000)
        self.market_state_changes = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 动态时间尺度索提诺比率
        
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
                self.value_history.append(portfolio_value)
                return 0.0
            
            # 计算收益率
            prev_value = self.value_history[-1]
            if prev_value > 0:
                period_return = (portfolio_value - prev_value) / prev_value
            else:
                period_return = 0.0
            
            # 更新历史数据
            self.value_history.append(portfolio_value)
            self.returns_history.append(period_return)
            
            # 需要足够的数据进行计算
            if len(self.returns_history) < self.min_window_size:
                return 0.0
            
            # 更新市场状态和动态参数
            self._update_market_state()
            self._update_dynamic_window_size()
            
            # 计算多时间尺度索提诺比率
            self._calculate_multi_scale_sortino()
            
            # 计算最终奖励
            final_reward = self.dynamic_sortino * self.scale_factor
            
            # 更新跟踪变量
            self._update_tracking_variables(final_reward)
            self.step_count += 1
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"DynamicSortino奖励计算异常: {e}")
            return 0.0
        
    def reward(self, env) -> float:
        """
        计算动态时间尺度索提诺比率奖励
        
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
                self.value_history.append(current_value)
                return 0.0
            
            # 计算收益率
            prev_value = self.value_history[-1]
            if prev_value > 0:
                period_return = (current_value - prev_value) / prev_value
            else:
                period_return = 0.0
            
            # 更新历史数据
            self.value_history.append(current_value)
            self.returns_history.append(period_return)
            
            # 需要足够的数据进行计算
            if len(self.returns_history) < self.min_window_size:
                return 0.0
            
            # 更新市场状态和动态参数
            self._update_market_state()
            self._update_dynamic_window_size()
            
            # 计算多时间尺度索提诺比率
            self._calculate_multi_scale_sortino()
            
            # 计算最终奖励
            final_reward = self.dynamic_sortino * self.scale_factor
            
            # 更新跟踪变量
            self._update_tracking_variables(final_reward)
            self.step_count += 1
            
            # 记录重要信息
            if self.step_count % 100 == 0 or abs(final_reward) > 5:
                self.logger.info(
                    f"[DynamicSortino] 步骤{self.step_count}: "
                    f"动态索提诺={self.dynamic_sortino:.6f}, "
                    f"窗口大小={self.current_window_size}, "
                    f"市场状态={self.market_state}, "
                    f"奖励={final_reward:.4f}"
                )
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"DynamicSortino奖励计算异常: {e}")
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
                self.value_history.append(current_value)
                return 0.0
            
            # 计算收益率
            prev_value = self.value_history[-1]
            if prev_value > 0:
                period_return = (current_value - prev_value) / prev_value
            else:
                period_return = 0.0
            
            # 更新历史
            self.value_history.append(current_value)
            self.returns_history.append(period_return)
            
            if len(self.returns_history) < self.min_window_size:
                return 0.0
            
            # 简化的计算（适用于portfolio接口）
            returns_array = np.array(list(self.returns_history)[-self.current_window_size:])
            excess_returns = returns_array - self.risk_free_rate
            
            # 计算下行标准差
            negative_returns = excess_returns[excess_returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-6
            
            # 计算索提诺比率
            mean_excess_return = np.mean(excess_returns)
            sortino = mean_excess_return / downside_std if downside_std > 0 else 0.0
            
            self.step_count += 1
            return float(sortino * self.scale_factor)
            
        except Exception as e:
            self.logger.error(f"Portfolio奖励计算异常: {e}")
            return 0.0
    
    def _update_market_state(self) -> None:
        """
        更新市场状态分析
        
        分析当前市场的趋势、波动性等特征，用于指导动态参数调整
        """
        if len(self.returns_history) < self.min_window_size:
            return
        
        try:
            # 获取最近的收益率数据
            recent_returns = np.array(list(self.returns_history)[-self.current_window_size:])
            
            # 计算波动率
            self.current_volatility = np.std(recent_returns)
            self.volatility_history.append(self.current_volatility)
            
            # 计算趋势方向
            if len(recent_returns) >= 10:
                # 使用线性回归斜率估计趋势
                x = np.arange(len(recent_returns))
                slope = np.polyfit(x, recent_returns, 1)[0]
                self.trend_direction = np.clip(slope * 1000, -1.0, 1.0)  # 归一化到[-1, 1]
            
            # 确定市场状态
            prev_state = self.market_state
            
            if self.current_volatility > self.volatility_threshold * 2:
                self.market_state = 'volatile'
            elif self.trend_direction > self.trend_sensitivity:
                self.market_state = 'bullish'
            elif self.trend_direction < -self.trend_sensitivity:
                self.market_state = 'bearish'
            else:
                self.market_state = 'neutral'
            
            # 记录状态变化
            if prev_state != self.market_state:
                self.market_state_changes += 1
                self.logger.info(f"市场状态变化: {prev_state} → {self.market_state}")
            
        except Exception as e:
            self.logger.warning(f"市场状态更新异常: {e}")
    
    def _update_dynamic_window_size(self) -> None:
        """
        根据市场状态动态调整计算窗口大小
        
        不同市场状态需要不同的时间尺度：
        - 高波动市场：较小窗口，快速响应
        - 趋势市场：中等窗口，捕捉趋势
        - 平静市场：较大窗口，提高稳定性
        """
        try:
            # 基于市场状态的窗口调整
            if self.market_state == 'volatile':
                # 高波动时缩小窗口，提高响应性
                target_size = int(self.base_window_size * 0.7)
            elif self.market_state in ['bullish', 'bearish']:
                # 趋势市场使用中等窗口
                target_size = int(self.base_window_size * 0.9)
            else:  # neutral
                # 平静市场使用较大窗口，提高稳定性
                target_size = int(self.base_window_size * 1.2)
            
            # 基于波动率的进一步调整
            volatility_factor = 1.0 - min(self.current_volatility / (self.volatility_threshold * 3), 0.5)
            target_size = int(target_size * volatility_factor)
            
            # 限制在合理范围内
            self.current_window_size = np.clip(target_size, self.min_window_size, self.max_window_size)
            
            # 记录窗口大小变化
            self.window_size_history.append(self.current_window_size)
            
        except Exception as e:
            self.logger.warning(f"动态窗口大小更新异常: {e}")
    
    def _calculate_multi_scale_sortino(self) -> None:
        """
        计算多时间尺度索提诺比率
        
        同时计算短期、中期、长期的索提诺比率，并进行加权融合
        """
        try:
            if len(self.returns_history) < self.min_window_size:
                return
            
            returns_array = np.array(list(self.returns_history))
            
            # 定义多时间尺度窗口
            short_window = max(self.min_window_size, self.current_window_size // 2)
            medium_window = self.current_window_size
            long_window = min(len(returns_array), self.current_window_size * 2)
            
            # 计算各时间尺度的索提诺比率
            self.short_term_sortino = self._calculate_weighted_sortino(
                returns_array[-short_window:], 
                window_type='short'
            )
            
            self.medium_term_sortino = self._calculate_weighted_sortino(
                returns_array[-medium_window:], 
                window_type='medium'
            )
            
            self.long_term_sortino = self._calculate_weighted_sortino(
                returns_array[-long_window:], 
                window_type='long'
            )
            
            # 加权融合得到动态索提诺比率
            self.dynamic_sortino = (
                self.multi_scale_weights['short'] * self.short_term_sortino +
                self.multi_scale_weights['medium'] * self.medium_term_sortino +
                self.multi_scale_weights['long'] * self.long_term_sortino
            )
            
        except Exception as e:
            self.logger.warning(f"多尺度索提诺计算异常: {e}")
            self.dynamic_sortino = 0.0
    
    def _calculate_weighted_sortino(self, returns: np.ndarray, window_type: str = 'medium') -> float:
        """
        计算时间加权索提诺比率
        
        Args:
            returns: 收益率数组
            window_type: 窗口类型 ('short', 'medium', 'long')
            
        Returns:
            float: 加权索提诺比率
        """
        try:
            if len(returns) < 2:
                return 0.0
            
            # 计算时间衰减权重
            n = len(returns)
            if window_type == 'short':
                # 短期：权重衰减更快，更关注近期
                decay = self.time_decay_factor ** 0.5
            elif window_type == 'long':
                # 长期：权重衰减更慢，更平滑
                decay = self.time_decay_factor ** 2
            else:
                # 中期：标准衰减
                decay = self.time_decay_factor
            
            weights = np.array([decay ** (n - 1 - i) for i in range(n)])
            weights = weights / np.sum(weights)  # 归一化
            
            # 计算超额收益
            excess_returns = returns - self.risk_free_rate
            
            # 加权平均超额收益
            weighted_mean_return = np.sum(weights * excess_returns)
            
            # 计算加权下行标准差
            negative_excess = excess_returns[excess_returns < 0]
            negative_weights = weights[excess_returns < 0]
            
            if len(negative_excess) > 0 and np.sum(negative_weights) > 0:
                # 重新归一化负收益的权重
                negative_weights = negative_weights / np.sum(negative_weights)
                weighted_downside_var = np.sum(negative_weights * (negative_excess ** 2))
                downside_std = np.sqrt(weighted_downside_var)
            else:
                # 没有负收益时使用小的正值避免除零
                downside_std = 1e-6
            
            # 计算索提诺比率
            sortino = weighted_mean_return / downside_std if downside_std > 0 else 0.0
            
            # 异常值处理
            if not np.isfinite(sortino):
                sortino = 0.0
            else:
                sortino = np.clip(sortino, -10.0, 10.0)
            
            return float(sortino)
            
        except Exception as e:
            self.logger.warning(f"加权索提诺计算异常: {e}")
            return 0.0
    
    def _update_tracking_variables(self, reward: float) -> None:
        """
        更新跟踪变量和历史记录
        
        Args:
            reward: 当前奖励值
        """
        # 更新索提诺比率历史
        self.sortino_history.append(self.dynamic_sortino)
        
        # 更新奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
    
    def reset(self) -> 'DynamicSortinoReward':
        """
        重置奖励函数状态
        
        Returns:
            DynamicSortinoReward: 返回self以支持链式调用
        """
        # 记录回合性能
        if len(self.value_history) > 0:
            final_value = self.value_history[-1]
            final_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
            avg_sortino = np.mean(list(self.sortino_history)) if self.sortino_history else 0.0
            avg_window_size = np.mean(list(self.window_size_history)) if self.window_size_history else self.base_window_size
            
            self.logger.info(
                f"[DynamicSortino回合{self.episode_count}结束] "
                f"最终收益率: {final_return:.4f}, "
                f"平均索提诺: {avg_sortino:.6f}, "
                f"平均窗口: {avg_window_size:.1f}, "
                f"市场状态变化: {self.market_state_changes}, "
                f"步数: {self.step_count}"
            )
        
        # 调用父类reset
        super().reset()
        
        # 重置状态但保留部分历史用于连续学习
        if len(self.returns_history) > 50:
            # 保留最近的一些数据作为下一回合的起点
            recent_returns = list(self.returns_history)[-20:]
            recent_values = list(self.value_history)[-20:]
            self.returns_history.clear()
            self.value_history.clear()
            self.returns_history.extend(recent_returns)
            self.value_history.extend(recent_values)
        else:
            self.returns_history.clear()
            self.value_history.clear()
        
        # 重置动态状态
        self.current_window_size = self.base_window_size
        self.market_state = 'neutral'
        self.trend_direction = 0.0
        self.current_volatility = 0.0
        self.market_state_changes = 0
        
        # 重置多尺度索提诺
        self.short_term_sortino = 0.0
        self.medium_term_sortino = 0.0
        self.long_term_sortino = 0.0
        self.dynamic_sortino = 0.0
        
        return self
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要，包含DynamicSortino特有指标
        
        Returns:
            Dict[str, Any]: 性能摘要信息
        """
        base_summary = super().get_performance_summary()
        
        # 计算平均指标
        avg_sortino = np.mean(list(self.sortino_history)) if self.sortino_history else 0.0
        avg_window_size = np.mean(list(self.window_size_history)) if self.window_size_history else self.base_window_size
        avg_volatility = np.mean(list(self.volatility_history)) if self.volatility_history else 0.0
        
        # 添加DynamicSortino特有指标
        dynamic_sortino_metrics = {
            'dynamic_sortino_ratio': self.dynamic_sortino,
            'short_term_sortino': self.short_term_sortino,
            'medium_term_sortino': self.medium_term_sortino,
            'long_term_sortino': self.long_term_sortino,
            'avg_sortino_ratio': avg_sortino,
            'current_window_size': self.current_window_size,
            'avg_window_size': avg_window_size,
            'market_state': self.market_state,
            'trend_direction': self.trend_direction,
            'current_volatility': self.current_volatility,
            'avg_volatility': avg_volatility,
            'market_state_changes': self.market_state_changes,
            'window_adaptability': np.std(list(self.window_size_history)) if len(self.window_size_history) > 1 else 0.0
        }
        
        base_summary.update(dynamic_sortino_metrics)
        return base_summary
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'DynamicSortinoReward',
            'description': '动态时间尺度索提诺比率奖励函数，基于自适应窗口和多时间尺度分析的风险调整奖励',
            'category': 'adaptive_risk_adjusted',
            'parameters': {
                'base_window_size': {
                    'type': 'int',
                    'default': 50,
                    'description': '基础计算窗口大小，作为动态调整的基准'
                },
                'min_window_size': {
                    'type': 'int',
                    'default': 20,
                    'description': '最小窗口大小，保证计算的统计意义'
                },
                'max_window_size': {
                    'type': 'int',
                    'default': 200,
                    'description': '最大窗口大小，避免过度滞后'
                },
                'risk_free_rate': {
                    'type': 'float',
                    'default': 0.02,
                    'description': '年化无风险收益率，用于索提诺比率计算'
                },
                'volatility_threshold': {
                    'type': 'float',
                    'default': 0.02,
                    'description': '波动性阈值，触发窗口调整的临界值'
                },
                'trend_sensitivity': {
                    'type': 'float',
                    'default': 0.1,
                    'description': '趋势敏感度，控制趋势对窗口大小的影响'
                },
                'time_decay_factor': {
                    'type': 'float',
                    'default': 0.95,
                    'description': '时间衰减因子，控制历史数据的权重衰减'
                },
                'scale_factor': {
                    'type': 'float',
                    'default': 100.0,
                    'description': '奖励缩放因子，调整奖励信号的量级'
                },
                'initial_balance': {
                    'type': 'float',
                    'default': 10000.0,
                    'description': '初始资金，用于收益率计算'
                }
            },
            'advantages': [
                '自适应时间窗口调整',
                '多时间尺度分析',
                '市场状态感知',
                '时间衰减加权',
                '下行风险专门优化',
                '动态参数调整'
            ],
            'use_cases': [
                '多变市场环境的自适应交易',
                '需要不同时间尺度响应的策略',
                '风险敏感的投资管理',
                '波动性自适应的交易系统',
                '趋势识别与风险控制结合'
            ],
            'mathematical_foundation': '基于索提诺比率的动态扩展，结合自适应窗口和多时间尺度分析',
            'complexity': 'advanced'
        }