"""
多指标综合奖励函数

综合考虑收益率、风险、稳定性、交易成本等多个维度的奖励函数。
适合需要全面评估交易策略表现的场景。
"""

import numpy as np
import logging
from typing import Dict, Any, List
from .base_reward import BaseRewardScheme


class DiversifiedReward(BaseRewardScheme):
    """
    多指标综合奖励函数
    
    这个奖励函数综合考虑多个维度：
    1. 收益率（Return）
    2. 风险控制（Risk）
    3. 稳定性（Stability）
    4. 交易效率（Efficiency）
    5. 回撤控制（Drawdown）
    """
    
    def __init__(self,
                 initial_balance: float = 10000.0,
                 weights: Dict[str, float] = None,
                 risk_free_rate: float = 0.02,
                 volatility_window: int = 20,
                 drawdown_threshold: float = 0.1,
                 trading_cost: float = 0.001,
                 **kwargs):
        """
        初始化多指标综合奖励函数
        
        Args:
            initial_balance: 初始资金
            weights: 各指标权重字典
            risk_free_rate: 无风险收益率
            volatility_window: 波动率计算窗口
            drawdown_threshold: 回撤阈值
            trading_cost: 交易成本
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 默认权重配置
        default_weights = {
            'return': 0.4,      # 收益率权重
            'risk': 0.2,        # 风险控制权重
            'stability': 0.15,   # 稳定性权重
            'efficiency': 0.15,  # 交易效率权重
            'drawdown': 0.1     # 回撤控制权重
        }
        
        self.weights = weights or default_weights
        self.risk_free_rate = risk_free_rate / 252  # 转换为日收益率
        self.volatility_window = volatility_window
        self.drawdown_threshold = drawdown_threshold
        self.trading_cost = trading_cost
        
        # 性能跟踪变量
        self.returns_history: List[float] = []
        self.actions_history: List[float] = []
        self.max_portfolio_value = initial_balance
        self.volatility_cache = []
        self.sharpe_ratios = []
        
        # 稳定性指标
        self.consecutive_positive_days = 0
        self.consecutive_negative_days = 0
        self.stability_score = 0.0
        
    def reward(self, env) -> float:
        """
        计算多指标综合奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 综合奖励值
        """
        try:
            # 更新状态并获取关键指标
            state = self.update_state(env)
            
            # 初始化处理 - 第一步返回0
            if state['step_count'] == 1:
                self.max_portfolio_value = state['current_value']
                logging.info(f"[多指标综合奖励初始化] 初始资产: ${state['current_value']:.2f}")
                return 0.0
            
            # 更新历史数据
            self._update_histories(state)
            
            # 计算各个指标的奖励
            return_reward = self._calculate_return_reward(state)
            risk_reward = self._calculate_risk_reward(state)
            stability_reward = self._calculate_stability_reward(state)
            efficiency_reward = self._calculate_efficiency_reward(state)
            drawdown_reward = self._calculate_drawdown_reward(state)
            
            # 综合奖励计算
            total_reward = (
                self.weights['return'] * return_reward +
                self.weights['risk'] * risk_reward +
                self.weights['stability'] * stability_reward +
                self.weights['efficiency'] * efficiency_reward +
                self.weights['drawdown'] * drawdown_reward
            )
            
            # 记录奖励历史
            self.reward_history.append(total_reward)
            
            # 日志记录（每30步或重要事件）
            if (state['step_count'] % 30 == 0 or abs(total_reward) > 5.0 or
                state['step_count'] % 100 == 0):
                
                self._log_detailed_metrics(state, {
                    'return': return_reward,
                    'risk': risk_reward,
                    'stability': stability_reward,
                    'efficiency': efficiency_reward,
                    'drawdown': drawdown_reward,
                    'total': total_reward
                })
            
            return float(total_reward)
            
        except Exception as e:
            logging.error(f"多指标综合奖励计算异常: {e}")
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
            # 从portfolio获取当前价值
            current_value = float(portfolio.net_worth)
            
            # 如果是第一步，初始化
            if self.previous_value is None:
                self.previous_value = current_value
                self.initial_value = current_value
                self.max_portfolio_value = current_value
                return 0.0
            
            # 计算收益率
            step_return_pct = (current_value - self.previous_value) / self.previous_value
            total_return_pct = (current_value - self.initial_value) / self.initial_value
            
            # 更新历史记录
            self.returns_history.append(step_return_pct)
            if len(self.returns_history) > 100:  # 保持最近100步的记录
                self.returns_history.pop(0)
            
            # 更新最大投资组合价值
            if current_value > self.max_portfolio_value:
                self.max_portfolio_value = current_value
            
            # 简化的多指标奖励计算
            # 1. 收益率奖励
            return_reward = total_return_pct * 10
            
            # 2. 风险调整（简化的夏普比率）
            risk_reward = 0.0
            if len(self.returns_history) >= 5:
                volatility = np.std(self.returns_history[-5:])
                if volatility > 0:
                    sharpe = np.mean(self.returns_history[-5:]) / volatility
                    risk_reward = sharpe * 2.0
                else:
                    risk_reward = 1.0  # 低波动奖励
            
            # 3. 回撤控制
            drawdown_reward = 0.0
            current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
            if current_drawdown < 0.05:  # 回撤小于5%
                drawdown_reward = 2.0
            elif current_drawdown > 0.2:  # 回撤大于20%
                drawdown_reward = -5.0
            
            # 综合奖励
            total_reward = (
                self.weights['return'] * return_reward +
                self.weights['risk'] * risk_reward +
                self.weights['drawdown'] * drawdown_reward
            )
            
            # 更新状态
            self.previous_value = current_value
            self.step_count += 1
            
            return float(total_reward)
            
        except Exception as e:
            return 0.0
    
    def _update_histories(self, state: Dict[str, Any]) -> None:
        """更新历史数据"""
        # 更新收益率历史
        if state['step_count'] > 1:
            self.returns_history.append(state['step_return_pct'] / 100)  # 转换为小数
            
        # 更新动作历史
        self.actions_history.append(state['current_action'])
        
        # 保持窗口大小
        max_history = max(100, self.volatility_window * 2)
        if len(self.returns_history) > max_history:
            self.returns_history.pop(0)
        if len(self.actions_history) > max_history:
            self.actions_history.pop(0)
        
        # 更新最大资产价值
        if state['current_value'] > self.max_portfolio_value:
            self.max_portfolio_value = state['current_value']
    
    def _calculate_return_reward(self, state: Dict[str, Any]) -> float:
        """计算收益率奖励"""
        step_return = state['step_return_pct']
        total_return = state['total_return_pct']
        
        # 基础收益奖励
        return_reward = step_return * 2.0  # 步骤收益率权重
        
        # 总收益率奖励（较小权重）
        return_reward += total_return * 0.1
        
        # 高收益奖励
        if step_return > 1.0:  # 超过1%
            return_reward *= 1.5
        elif step_return > 2.0:  # 超过2%
            return_reward *= 2.0
        
        return return_reward
    
    def _calculate_risk_reward(self, state: Dict[str, Any]) -> float:
        """计算风险控制奖励"""
        if len(self.returns_history) < 2:
            return 0.0
        
        # 计算波动率
        if len(self.returns_history) >= self.volatility_window:
            recent_returns = self.returns_history[-self.volatility_window:]
            volatility = np.std(recent_returns) * np.sqrt(252)  # 年化波动率
            
            # 计算夏普比率
            avg_return = np.mean(recent_returns) * 252  # 年化收益率
            excess_return = avg_return - self.risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # 夏普比率奖励
            risk_reward = sharpe_ratio * 2.0
            
            # 低波动率奖励
            if volatility < 0.15:  # 年化波动率低于15%
                risk_reward += 1.0
            elif volatility > 0.3:  # 高波动率惩罚
                risk_reward -= 2.0
            
            self.sharpe_ratios.append(sharpe_ratio)
            if len(self.sharpe_ratios) > 50:
                self.sharpe_ratios.pop(0)
                
        else:
            # 数据不足时的简单风险评估
            recent_returns = self.returns_history[-min(len(self.returns_history), 5):]
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
            risk_reward = -volatility * 10  # 早期阶段惩罚高波动
        
        return risk_reward
    
    def _calculate_stability_reward(self, state: Dict[str, Any]) -> float:
        """计算稳定性奖励"""
        step_return = state['step_return_pct']
        
        # 更新连续正负天数
        if step_return > 0:
            self.consecutive_positive_days += 1
            self.consecutive_negative_days = 0
        elif step_return < 0:
            self.consecutive_negative_days += 1
            self.consecutive_positive_days = 0
        
        # 连续盈利奖励
        stability_reward = 0.0
        if self.consecutive_positive_days >= 3:
            stability_reward += min(self.consecutive_positive_days * 0.2, 2.0)
        
        # 连续亏损惩罚
        if self.consecutive_negative_days >= 3:
            stability_reward -= min(self.consecutive_negative_days * 0.3, 3.0)
        
        # 收益率稳定性奖励
        if len(self.returns_history) >= 10:
            recent_returns = self.returns_history[-10:]
            returns_std = np.std(recent_returns)
            returns_mean = np.mean(recent_returns)
            
            # 稳定性评分：平均收益为正且波动小
            if returns_mean > 0 and returns_std < 0.01:  # 1%以内的波动
                stability_reward += 1.0
        
        return stability_reward
    
    def _calculate_efficiency_reward(self, state: Dict[str, Any]) -> float:
        """计算交易效率奖励"""
        current_action = state['current_action']
        
        # 交易成本惩罚
        trading_penalty = abs(current_action) * self.trading_cost * 100  # 转换为奖励规模
        
        # 过度交易惩罚
        if len(self.actions_history) >= 5:
            recent_actions = self.actions_history[-5:]
            action_volatility = np.std(recent_actions)
            
            if action_volatility > 0.5:  # 动作变化过于频繁
                trading_penalty += action_volatility * 0.5
        
        # 有效交易奖励
        efficiency_reward = -trading_penalty
        
        # 如果当前步骤有正收益且交易合理，给予效率奖励
        if state['step_return_pct'] > 0 and abs(current_action) > 0.1:
            efficiency_reward += state['step_return_pct'] * 0.1
        
        return efficiency_reward
    
    def _calculate_drawdown_reward(self, state: Dict[str, Any]) -> float:
        """计算回撤控制奖励"""
        current_value = state['current_value']
        
        # 计算当前回撤
        if current_value >= self.max_portfolio_value:
            return 1.0  # 创新高奖励
        
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        # 回撤惩罚
        if drawdown <= self.drawdown_threshold:
            # 可接受回撤范围内
            drawdown_reward = 1.0 - drawdown * 5  # 轻微惩罚
        else:
            # 超过回撤阈值，严重惩罚
            excess_drawdown = drawdown - self.drawdown_threshold
            drawdown_reward = -10 * excess_drawdown
        
        return drawdown_reward
    
    def _log_detailed_metrics(self, state: Dict[str, Any], rewards: Dict[str, float]) -> None:
        """记录详细指标"""
        current_drawdown = 0.0
        if self.max_portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - state['current_value']) / self.max_portfolio_value
        
        current_volatility = 0.0
        current_sharpe = 0.0
        if len(self.returns_history) >= self.volatility_window:
            recent_returns = self.returns_history[-self.volatility_window:]
            current_volatility = np.std(recent_returns) * np.sqrt(252)
            avg_return = np.mean(recent_returns) * 252
            current_sharpe = (avg_return - self.risk_free_rate) / current_volatility if current_volatility > 0 else 0
        
        logging.info(f"[多指标综合] 步骤{state['step_count']}: "
                   f"总奖励:{rewards['total']:.3f} = "
                   f"收益:{rewards['return']:.2f} + "
                   f"风险:{rewards['risk']:.2f} + "
                   f"稳定:{rewards['stability']:.2f} + "
                   f"效率:{rewards['efficiency']:.2f} + "
                   f"回撤:{rewards['drawdown']:.2f} | "
                   f"当前回撤:{current_drawdown:.2%} "
                   f"波动率:{current_volatility:.2%} "
                   f"夏普:{current_sharpe:.2f}")
    
    def reset(self) -> 'DiversifiedReward':
        """重置奖励函数状态"""
        # 记录回合统计
        if self.returns_history:
            total_return = (self.previous_value - self.initial_value) / self.initial_value if self.previous_value else 0
            avg_return = np.mean(self.returns_history) * 252 if self.returns_history else 0
            volatility = np.std(self.returns_history) * np.sqrt(252) if len(self.returns_history) > 1 else 0
            sharpe = (avg_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            max_drawdown = (self.max_portfolio_value - self.previous_value) / self.max_portfolio_value if self.previous_value else 0
            
            logging.info(f"[多指标回合{self.episode_count}结束] "
                        f"总收益:{total_return:.2%} "
                        f"年化收益:{avg_return:.2%} "
                        f"波动率:{volatility:.2%} "
                        f"夏普比率:{sharpe:.2f} "
                        f"最大回撤:{max_drawdown:.2%}")
        
        # 调用父类reset方法
        super().reset()
        
        # 重置特定状态（保留部分经验）
        if len(self.returns_history) > 50:
            # 保留最近的统计信息作为经验
            self.returns_history = self.returns_history[-20:]
            self.actions_history = self.actions_history[-20:]
        else:
            self.returns_history = []
            self.actions_history = []
        
        self.max_portfolio_value = self.initial_balance
        self.consecutive_positive_days = 0
        self.consecutive_negative_days = 0
        self.volatility_cache = []
        
        return self
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            'name': 'DiversifiedReward',
            'description': '多指标综合奖励函数，综合考虑收益、风险、稳定性、效率和回撤',
            'category': 'comprehensive',
            'parameters': {
                'initial_balance': {
                    'type': 'float',
                    'default': 10000.0,
                    'description': '初始资金'
                },
                'weights': {
                    'type': 'dict',
                    'default': {
                        'return': 0.4,
                        'risk': 0.2,
                        'stability': 0.15,
                        'efficiency': 0.15,
                        'drawdown': 0.1
                    },
                    'description': '各指标权重配置'
                },
                'risk_free_rate': {
                    'type': 'float',
                    'default': 0.02,
                    'description': '无风险收益率'
                },
                'volatility_window': {
                    'type': 'int',
                    'default': 20,
                    'description': '波动率计算窗口'
                },
                'drawdown_threshold': {
                    'type': 'float',
                    'default': 0.1,
                    'description': '回撤阈值'
                },
                'trading_cost': {
                    'type': 'float',
                    'default': 0.001,
                    'description': '交易成本'
                }
            },
            'metrics': [
                '收益率奖励',
                '风险控制（夏普比率）',
                '稳定性评估',
                '交易效率',
                '回撤控制'
            ],
            'features': [
                '多维度评估',
                '权重可配置',
                '经验保留',
                '全面风险控制',
                '适应性强'
            ],
            'use_cases': [
                '机构级策略',
                '风险平价策略',
                '全市场策略',
                '综合性能评估'
            ]
        }
    
    def calculate_reward(self, current_step, current_price, current_portfolio_value, action, **kwargs):
        """
        计算奖励 - BaseRewardScheme抽象方法的实现
        
        这是为了兼容抽象基类要求的方法，实际奖励计算逻辑在get_reward方法中
        """
        # 构建状态字典用于get_reward方法
        state = {
            'step_count': current_step,
            'current_price': current_price,
            'current_value': current_portfolio_value,
            'current_action': action,
            'total_return_pct': ((current_portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'step_return_pct': 0.0  # 将在get_reward中计算
        }
        
        # 调用实际的奖励计算方法
        return self.get_reward(state)