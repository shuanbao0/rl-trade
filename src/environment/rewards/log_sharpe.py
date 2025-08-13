"""
对数收益与夏普比率结合的奖励函数

基于Moody & Saffell (2001)的差分夏普比率理论，结合对数收益的统计优势，
为强化学习交易系统提供更稳健的风险调整奖励信号。

核心特性：
1. 对数收益计算：利用对数的统计优势处理复合收益
2. 差分夏普比率：支持在线学习的夏普比率变体
3. 指数移动平均：动态跟踪收益的一阶和二阶矩
4. 风险调整：平衡收益与波动性的关系
5. 自适应学习率：根据市场条件动态调整学习参数

数学基础：
- 对数收益：r_t = ln(P_t / P_{t-1})
- 差分夏普比率：DSR_t = [B_{t-1} ΔA_t - (1/2) A_{t-1} ΔB_t] / (B_{t-1} - A_{t-1}²)^(3/2)
- 其中A_t和B_t分别是收益的一阶和二阶矩的指数移动平均
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from .base_reward import BaseRewardScheme


class LogSharpeReward(BaseRewardScheme):
    """
    对数收益与差分夏普比率结合的奖励函数
    
    实现了Moody & Saffell的差分夏普比率算法，结合对数收益的统计优势，
    为强化学习提供更加稳健的风险调整奖励信号。
    
    该方法特别适用于需要考虑复合收益和风险调整的量化交易策略。
    """
    
    def __init__(self, 
                 eta: float = 0.01,
                 risk_free_rate: float = 0.02,
                 min_variance: float = 1e-6,
                 scale_factor: float = 100.0,
                 adaptive_eta: bool = True,
                 initial_balance: float = 10000.0,
                 **kwargs):
        """
        初始化对数夏普奖励函数
        
        Args:
            eta: 指数移动平均的衰减率，控制历史数据的影响程度
            risk_free_rate: 年化无风险收益率，用于风险调整计算
            min_variance: 最小方差阈值，避免除零错误
            scale_factor: 奖励缩放因子，调整奖励信号的量级
            adaptive_eta: 是否启用自适应学习率调整
            initial_balance: 初始资金，用于计算对数收益
            **kwargs: 其他参数
        """
        super().__init__(initial_balance=initial_balance, **kwargs)
        
        # 核心参数
        self.eta = eta
        self.risk_free_rate = risk_free_rate / 252  # 转换为日收益率
        self.min_variance = min_variance
        self.scale_factor = scale_factor
        self.adaptive_eta = adaptive_eta
        
        # 差分夏普比率计算状态
        self.A_t = 0.0  # 收益的一阶矩（期望）
        self.B_t = 0.0  # 收益的二阶矩（方差+期望²）
        self.prev_A_t = 0.0
        self.prev_B_t = 0.0
        
        # 对数收益计算状态
        self.prev_portfolio_value = None
        self.log_returns_history = []
        
        # 自适应参数
        self.volatility_estimate = 0.0
        self.base_eta = eta
        self.eta_adjustment_window = 50
        
        # 性能跟踪
        self.sharpe_ratios = []
        self.differential_sharpe_ratios = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        奖励计算接口 - 基于对数收益和差分夏普比率
        
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
            if self.prev_portfolio_value is None:
                self.prev_portfolio_value = portfolio_value
                self.A_t = 0.0
                self.B_t = self.min_variance
                return 0.0
            
            # 计算对数收益
            log_return = self._calculate_log_return(portfolio_value)
            
            # 更新指数移动平均
            self._update_exponential_moving_averages(log_return)
            
            # 计算差分夏普比率
            differential_sharpe = self._calculate_differential_sharpe_ratio(log_return)
            
            # 自适应调整学习率
            if self.adaptive_eta:
                self._adjust_eta()
            
            # 计算最终奖励
            reward = differential_sharpe * self.scale_factor
            
            # 更新历史和状态
            self._update_tracking_variables(log_return, differential_sharpe)
            self.prev_portfolio_value = portfolio_value
            self.step_count += 1
            
            return float(reward)
            
        except Exception as e:
            self.logger.error(f"LogSharpe奖励计算异常: {e}")
            return 0.0
        
    def reward(self, env) -> float:
        """
        计算基于对数收益和差分夏普比率的奖励
        
        Args:
            env: TensorTrade环境实例
            
        Returns:
            float: 计算得到的奖励值
        """
        try:
            # 获取当前投资组合价值
            current_value = self.get_portfolio_value(env)
            
            # 第一步初始化
            if self.prev_portfolio_value is None:
                self.prev_portfolio_value = current_value
                self.A_t = 0.0
                self.B_t = self.min_variance  # 初始化为最小方差避免除零
                return 0.0
            
            # 计算对数收益
            log_return = self._calculate_log_return(current_value)
            
            # 更新指数移动平均
            self._update_exponential_moving_averages(log_return)
            
            # 计算差分夏普比率
            differential_sharpe = self._calculate_differential_sharpe_ratio(log_return)
            
            # 自适应调整学习率
            if self.adaptive_eta:
                self._adjust_eta()
            
            # 计算最终奖励
            reward = differential_sharpe * self.scale_factor
            
            # 更新历史和状态
            self._update_tracking_variables(log_return, differential_sharpe)
            self.prev_portfolio_value = current_value
            self.step_count += 1
            
            # 记录重要信息
            if self.step_count % 100 == 0 or abs(reward) > 10:
                self.logger.info(
                    f"[LogSharpe] 步骤{self.step_count}: "
                    f"对数收益={log_return:.6f}, DSR={differential_sharpe:.6f}, "
                    f"奖励={reward:.4f}, A_t={self.A_t:.6f}, B_t={self.B_t:.6f}"
                )
            
            return float(reward)
            
        except Exception as e:
            self.logger.error(f"LogSharpe奖励计算异常: {e}")
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
            
            if self.prev_portfolio_value is None:
                self.prev_portfolio_value = current_value
                self.A_t = 0.0
                self.B_t = self.min_variance
                return 0.0
            
            # 计算对数收益
            log_return = self._calculate_log_return(current_value)
            
            # 更新移动平均
            self._update_exponential_moving_averages(log_return)
            
            # 计算差分夏普比率
            differential_sharpe = self._calculate_differential_sharpe_ratio(log_return)
            
            # 更新状态
            self.prev_portfolio_value = current_value
            self.step_count += 1
            
            return float(differential_sharpe * self.scale_factor)
            
        except Exception as e:
            self.logger.error(f"Portfolio奖励计算异常: {e}")
            return 0.0
    
    def _calculate_log_return(self, current_value: float) -> float:
        """
        计算对数收益率
        
        对数收益的优势：
        1. 时间可加性：多期对数收益可以直接相加
        2. 正态分布假设：更符合金融时间序列的统计特性
        3. 复合效应：自然处理复利计算
        
        Args:
            current_value: 当前投资组合价值
            
        Returns:
            float: 对数收益率
        """
        if self.prev_portfolio_value is None or self.prev_portfolio_value <= 0:
            return 0.0
        
        try:
            # 确保价值为正，避免log(0)或log(负数)
            prev_value = max(self.prev_portfolio_value, 1e-8)
            curr_value = max(current_value, 1e-8)
            
            # 计算对数收益：ln(P_t / P_{t-1})
            log_return = np.log(curr_value / prev_value)
            
            # 异常值处理：限制极端收益
            log_return = np.clip(log_return, -0.5, 0.5)  # 限制在±50%内
            
            return float(log_return)
            
        except Exception as e:
            self.logger.warning(f"对数收益计算异常: {e}")
            return 0.0
    
    def _update_exponential_moving_averages(self, log_return: float) -> None:
        """
        更新对数收益的指数移动平均
        
        计算公式：
        A_t = A_{t-1} + η * (r_t - A_{t-1})  # 一阶矩（期望）
        B_t = B_{t-1} + η * (r_t² - B_{t-1})  # 二阶矩（期望的平方）
        
        Args:
            log_return: 当前对数收益
        """
        # 保存上一步的值用于差分计算
        self.prev_A_t = self.A_t
        self.prev_B_t = self.B_t
        
        # 风险调整：减去无风险收益率
        excess_return = log_return - self.risk_free_rate
        
        # 更新一阶矩（期望）
        self.A_t = self.A_t + self.eta * (excess_return - self.A_t)
        
        # 更新二阶矩（方差+期望²）
        self.B_t = self.B_t + self.eta * (excess_return**2 - self.B_t)
        
        # 确保二阶矩不低于最小方差阈值
        self.B_t = max(self.B_t, self.min_variance)
    
    def _calculate_differential_sharpe_ratio(self, log_return: float) -> float:
        """
        计算差分夏普比率
        
        基于Moody & Saffell (2001)的公式：
        DSR_t = [B_{t-1} * ΔA_t - (1/2) * A_{t-1} * ΔB_t] / (B_{t-1} - A_{t-1}²)^(3/2)
        
        其中：
        ΔA_t = A_t - A_{t-1} = η * (r_t - A_{t-1})
        ΔB_t = B_t - B_{t-1} = η * (r_t² - B_{t-1})
        
        Args:
            log_return: 当前对数收益
            
        Returns:
            float: 差分夏普比率
        """
        try:
            # 计算差分项
            excess_return = log_return - self.risk_free_rate
            delta_A = self.eta * (excess_return - self.prev_A_t)
            delta_B = self.eta * (excess_return**2 - self.prev_B_t)
            
            # 计算分母：方差 = B_{t-1} - A_{t-1}²
            variance = self.prev_B_t - self.prev_A_t**2
            variance = max(variance, self.min_variance)  # 避免负方差或零方差
            
            # 计算分子：B_{t-1} * ΔA_t - (1/2) * A_{t-1} * ΔB_t
            numerator = self.prev_B_t * delta_A - 0.5 * self.prev_A_t * delta_B
            
            # 计算差分夏普比率
            differential_sharpe = numerator / (variance**(3/2))
            
            # 异常值处理
            if not np.isfinite(differential_sharpe):
                differential_sharpe = 0.0
            else:
                # 限制极端值
                differential_sharpe = np.clip(differential_sharpe, -10.0, 10.0)
            
            return float(differential_sharpe)
            
        except Exception as e:
            self.logger.warning(f"差分夏普比率计算异常: {e}")
            return 0.0
    
    def _adjust_eta(self) -> None:
        """
        根据市场波动性自适应调整学习率
        
        在高波动期间降低学习率以提高稳定性，
        在低波动期间提高学习率以增强响应性。
        """
        if len(self.log_returns_history) < self.eta_adjustment_window:
            return
        
        try:
            # 计算最近的收益波动性
            recent_returns = self.log_returns_history[-self.eta_adjustment_window:]
            current_volatility = np.std(recent_returns)
            
            # 平滑波动性估计
            if self.volatility_estimate == 0:
                self.volatility_estimate = current_volatility
            else:
                self.volatility_estimate = 0.9 * self.volatility_estimate + 0.1 * current_volatility
            
            # 基于波动性调整学习率
            # 高波动时降低eta，低波动时提高eta
            volatility_factor = 1.0 / (1.0 + 10.0 * self.volatility_estimate)
            self.eta = self.base_eta * (0.5 + 0.5 * volatility_factor)
            
            # 限制eta的范围
            self.eta = np.clip(self.eta, 0.001, 0.1)
            
        except Exception as e:
            self.logger.warning(f"自适应eta调整异常: {e}")
    
    def _update_tracking_variables(self, log_return: float, differential_sharpe: float) -> None:
        """
        更新跟踪变量和历史记录
        
        Args:
            log_return: 对数收益
            differential_sharpe: 差分夏普比率
        """
        # 更新对数收益历史
        self.log_returns_history.append(log_return)
        if len(self.log_returns_history) > 1000:  # 保持最近1000步
            self.log_returns_history.pop(0)
        
        # 更新差分夏普比率历史
        self.differential_sharpe_ratios.append(differential_sharpe)
        if len(self.differential_sharpe_ratios) > 1000:
            self.differential_sharpe_ratios.pop(0)
        
        # 计算传统夏普比率用于比较
        if len(self.log_returns_history) > 10:
            try:
                returns_array = np.array(self.log_returns_history[-50:])  # 最近50步
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if std_return > 0:
                    sharpe = (mean_return - self.risk_free_rate) / std_return
                    self.sharpe_ratios.append(sharpe)
                    if len(self.sharpe_ratios) > 100:
                        self.sharpe_ratios.pop(0)
            except Exception:
                pass
    
    def reset(self) -> 'LogSharpeReward':
        """
        重置奖励函数状态
        
        Returns:
            LogSharpeReward: 返回self以支持链式调用
        """
        # 记录回合性能
        if self.prev_portfolio_value is not None and self.initial_balance > 0:
            final_return = (self.prev_portfolio_value - self.initial_balance) / self.initial_balance
            avg_dsr = np.mean(self.differential_sharpe_ratios) if self.differential_sharpe_ratios else 0.0
            avg_sharpe = np.mean(self.sharpe_ratios) if self.sharpe_ratios else 0.0
            
            self.logger.info(
                f"[LogSharpe回合{self.episode_count}结束] "
                f"最终收益率: {final_return:.4f}, "
                f"平均DSR: {avg_dsr:.6f}, "
                f"平均Sharpe: {avg_sharpe:.4f}, "
                f"步数: {self.step_count}"
            )
        
        # 调用父类reset
        super().reset()
        
        # 重置状态但保留学习参数
        self.prev_portfolio_value = None
        self.A_t = 0.0
        self.B_t = self.min_variance
        self.prev_A_t = 0.0
        self.prev_B_t = 0.0
        
        # 保留部分历史用于连续学习
        if len(self.log_returns_history) > 100:
            self.log_returns_history = self.log_returns_history[-50:]
        if len(self.differential_sharpe_ratios) > 100:
            self.differential_sharpe_ratios = self.differential_sharpe_ratios[-50:]
        if len(self.sharpe_ratios) > 50:
            self.sharpe_ratios = self.sharpe_ratios[-25:]
        
        return self
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要，包含LogSharpe特有指标
        
        Returns:
            Dict[str, Any]: 性能摘要信息
        """
        base_summary = super().get_performance_summary()
        
        # 添加LogSharpe特有指标
        log_sharpe_metrics = {
            'avg_log_return': np.mean(self.log_returns_history) if self.log_returns_history else 0.0,
            'log_return_volatility': np.std(self.log_returns_history) if self.log_returns_history else 0.0,
            'avg_differential_sharpe': np.mean(self.differential_sharpe_ratios) if self.differential_sharpe_ratios else 0.0,
            'avg_traditional_sharpe': np.mean(self.sharpe_ratios) if self.sharpe_ratios else 0.0,
            'current_eta': self.eta,
            'current_A_t': self.A_t,
            'current_B_t': self.B_t,
            'estimated_volatility': self.volatility_estimate
        }
        
        base_summary.update(log_sharpe_metrics)
        return base_summary
    
    @classmethod
    def get_reward_info(cls) -> Dict[str, Any]:
        """
        获取奖励函数信息
        
        Returns:
            Dict[str, Any]: 奖励函数的描述信息
        """
        return {
            'name': 'LogSharpeReward',
            'description': '基于对数收益和差分夏普比率的奖励函数，适用于风险调整的量化交易策略',
            'category': 'risk_adjusted_advanced',
            'parameters': {
                'eta': {
                    'type': 'float',
                    'default': 0.01,
                    'description': '指数移动平均的衰减率，控制历史数据影响程度'
                },
                'risk_free_rate': {
                    'type': 'float',
                    'default': 0.02,
                    'description': '年化无风险收益率，用于风险调整计算'
                },
                'min_variance': {
                    'type': 'float',
                    'default': 1e-6,
                    'description': '最小方差阈值，避免数值计算问题'
                },
                'scale_factor': {
                    'type': 'float',
                    'default': 100.0,
                    'description': '奖励缩放因子，调整奖励信号量级'
                },
                'adaptive_eta': {
                    'type': 'bool',
                    'default': True,
                    'description': '是否启用自适应学习率调整'
                },
                'initial_balance': {
                    'type': 'float',
                    'default': 10000.0,
                    'description': '初始资金，用于计算对数收益'
                }
            },
            'advantages': [
                '基于对数收益的统计优势',
                '差分夏普比率支持在线学习',
                '自适应学习率调整',
                '风险调整的收益优化',
                '适合复合收益计算'
            ],
            'use_cases': [
                '量化交易策略',
                '投资组合优化',
                '风险管理',
                '长期投资策略',
                '机构级交易系统'
            ],
            'mathematical_foundation': 'Moody & Saffell (2001) 差分夏普比率理论',
            'complexity': 'advanced'
        }