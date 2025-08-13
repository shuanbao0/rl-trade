"""
风险管理模块
实现多层次风险控制和实时性能监控

主要功能:
1. 仓位限制控制
2. 回撤限制监控
3. 日损失限制检查
4. 实时性能监控
5. 风险报告生成
6. 异常检测和告警
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import deque
import warnings

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


@dataclass
class RiskLimits:
    """风险限制配置"""
    max_position_ratio: float = 0.95  # 最大仓位比例
    max_single_position: float = 0.3   # 单个资产最大仓位
    max_drawdown: float = 0.15         # 最大回撤
    daily_loss_limit: float = 0.05     # 日损失限制
    stop_loss: float = 0.1             # 止损限制
    max_leverage: float = 1.0          # 最大杠杆
    min_cash_ratio: float = 0.05       # 最小现金比例
    
    def validate(self) -> None:
        """验证风险参数的合理性"""
        if not 0 < self.max_position_ratio <= 1.0:
            raise ValueError("max_position_ratio must be between 0 and 1")
        if not 0 < self.max_single_position <= 1.0:
            raise ValueError("max_single_position must be between 0 and 1")
        if not 0 < self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be between 0 and 1")
        if not 0 < self.daily_loss_limit <= 1.0:
            raise ValueError("daily_loss_limit must be between 0 and 1")


@dataclass
class PortfolioState:
    """投资组合状态"""
    total_value: float = 0.0
    cash: float = 0.0
    positions: Dict[str, float] = None
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    drawdown: float = 0.0
    max_value: float = 0.0
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}


@dataclass
class RiskEvent:
    """风险事件"""
    timestamp: datetime
    event_type: str
    severity: str  # 'warning', 'critical'
    message: str
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class PerformanceMonitor:
    """
    性能监控组件
    
    实时监控交易性能和风险指标:
    - 收益率跟踪
    - 波动率计算
    - 回撤监控
    - 夏普比率计算
    - 风险调整收益
    """
    
    def __init__(self, window_size: int = 252):
        """
        初始化性能监控器
        
        Args:
            window_size: 滚动窗口大小（默认252个交易日）
        """
        self.window_size = window_size
        
        # 历史数据存储
        self.returns_history = deque(maxlen=window_size)
        self.values_history = deque(maxlen=window_size)
        self.drawdown_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # 性能指标
        self.peak_value = 0.0
        self.max_drawdown = 0.0
        self.cumulative_return = 0.0
        
        # 初始化日志
        self.logger = setup_logger(
            name="PerformanceMonitor",
            level="INFO",
            log_file=get_default_log_file("performance_monitor")
        )
    
    def update(self, portfolio_value: float, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        更新性能数据
        
        Args:
            portfolio_value: 当前投资组合价值
            timestamp: 时间戳
            
        Returns:
            Dict[str, float]: 当前性能指标
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 计算收益率
        if len(self.values_history) > 0:
            returns = (portfolio_value - self.values_history[-1]) / self.values_history[-1]
        else:
            returns = 0.0
        
        # 更新历史数据
        self.returns_history.append(returns)
        self.values_history.append(portfolio_value)
        self.timestamps.append(timestamp)
        
        # 更新峰值和回撤
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        self.drawdown_history.append(current_drawdown)
        
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # 计算累计收益
        if len(self.values_history) >= 2:
            initial_value = self.values_history[0]
            self.cumulative_return = (portfolio_value - initial_value) / initial_value if initial_value > 0 else 0
        
        # 计算当前性能指标
        metrics = self.calculate_current_metrics()
        
        return metrics
    
    def calculate_current_metrics(self) -> Dict[str, float]:
        """
        计算当前性能指标
        
        Returns:
            Dict[str, float]: 性能指标字典
        """
        if len(self.returns_history) == 0:
            return {}
        
        returns_array = np.array(list(self.returns_history))
        
        # 基础指标
        metrics = {
            'current_value': self.values_history[-1] if self.values_history else 0,
            'current_drawdown': self.drawdown_history[-1] if self.drawdown_history else 0,
            'max_drawdown': self.max_drawdown,
            'cumulative_return': self.cumulative_return,
            'peak_value': self.peak_value
        }
        
        if len(returns_array) > 1:
            # 收益率统计
            metrics.update({
                'mean_return': float(np.mean(returns_array)),
                'volatility': float(np.std(returns_array)),
                'skewness': float(self._calculate_skewness(returns_array)),
                'kurtosis': float(self._calculate_kurtosis(returns_array))
            })
            
            # 年化指标（假设252个交易日）
            if len(returns_array) >= 10:  # 至少10个数据点
                annualized_return = float(np.mean(returns_array) * 252)
                annualized_volatility = float(np.std(returns_array) * np.sqrt(252))
                
                metrics.update({
                    'annualized_return': annualized_return,
                    'annualized_volatility': annualized_volatility
                })
                
                # 夏普比率（假设无风险利率为0）
                if annualized_volatility > 0:
                    metrics['sharpe_ratio'] = annualized_return / annualized_volatility
                else:
                    metrics['sharpe_ratio'] = 0.0
            
            # 最近表现
            if len(returns_array) >= 5:
                recent_returns = returns_array[-5:]  # 最近5期
                metrics['recent_mean_return'] = float(np.mean(recent_returns))
                metrics['recent_volatility'] = float(np.std(recent_returns))
        
        return metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """计算偏度"""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """计算峰度"""
        if len(returns) < 4:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return kurtosis
    
    def get_performance_summary(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Args:
            days: 回看天数，None为全部历史
            
        Returns:
            Dict[str, Any]: 性能摘要
        """
        if not self.returns_history:
            return {"message": "No performance data available"}
        
        # 确定数据范围
        if days is None or days >= len(self.returns_history):
            returns_data = list(self.returns_history)
            values_data = list(self.values_history)
            timestamps_data = list(self.timestamps)
        else:
            returns_data = list(self.returns_history)[-days:]
            values_data = list(self.values_history)[-days:]
            timestamps_data = list(self.timestamps)[-days:]
        
        if not returns_data:
            return {"message": "No data in specified period"}
        
        summary = {
            'period': {
                'start_date': str(timestamps_data[0]),
                'end_date': str(timestamps_data[-1]),
                'days': len(returns_data)
            },
            'return_metrics': {
                'total_return': float((values_data[-1] - values_data[0]) / values_data[0]) if values_data[0] != 0 else 0,
                'mean_daily_return': float(np.mean(returns_data)),
                'volatility': float(np.std(returns_data)),
                'best_day': float(np.max(returns_data)),
                'worst_day': float(np.min(returns_data))
            },
            'risk_metrics': {
                'max_drawdown': float(np.max(list(self.drawdown_history)[-len(returns_data):])),
                'current_drawdown': float(self.drawdown_history[-1]) if self.drawdown_history else 0,
                'positive_days': int(np.sum(np.array(returns_data) > 0)),
                'negative_days': int(np.sum(np.array(returns_data) < 0)),
                'win_rate': float(np.mean(np.array(returns_data) > 0))
            }
        }
        
        # 年化指标
        if len(returns_data) >= 30:  # 至少30天数据
            annualized_return = np.mean(returns_data) * 252
            annualized_volatility = np.std(returns_data) * np.sqrt(252)
            
            summary['annualized_metrics'] = {
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_volatility),
                'sharpe_ratio': float(annualized_return / annualized_volatility) if annualized_volatility > 0 else 0
            }
        
        return summary
    
    def reset(self) -> None:
        """重置监控器"""
        self.returns_history.clear()
        self.values_history.clear()
        self.drawdown_history.clear()
        self.timestamps.clear()
        
        self.peak_value = 0.0
        self.max_drawdown = 0.0
        self.cumulative_return = 0.0
        
        self.logger.info("性能监控器已重置")


class RiskManager:
    """
    风险管理器
    
    提供多层次风险控制:
    - 仓位管理和限制
    - 回撤控制
    - 日损失监控
    - 实时风险检查
    - 风险事件记录
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        risk_limits: Optional[RiskLimits] = None
    ):
        """
        初始化风险管理器
        
        Args:
            config: 配置对象
            risk_limits: 风险限制配置
        """
        self.config = config or Config()
        self.risk_limits = risk_limits or RiskLimits()
        self.risk_limits.validate()
        
        # 初始化日志
        self.logger = setup_logger(
            name="RiskManager",
            level="INFO",
            log_file=get_default_log_file("risk_manager")
        )
        
        # 投资组合状态
        self.portfolio_state = PortfolioState()
        
        # 性能监控器
        self.performance_monitor = PerformanceMonitor()
        
        # 风险事件记录
        self.risk_events = []
        
        # 日期跟踪
        self.current_date = None
        self.daily_start_value = 0.0
        
        # 风险状态
        self.risk_breaches = {
            'position_limit': False,
            'drawdown_limit': False,
            'daily_loss_limit': False,
            'stop_loss': False
        }
        
        self.logger.info("RiskManager 初始化完成")
        self.logger.info(f"风险限制: {self.risk_limits}")
    
    def update_portfolio_state(
        self,
        total_value: float,
        cash: float,
        positions: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        更新投资组合状态
        
        Args:
            total_value: 总价值
            cash: 现金
            positions: 持仓字典 {asset: value}
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 检查是否是新的一天
        current_date = timestamp.date()
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_value = total_value
            self.logger.debug(f"新交易日开始: {current_date}, 起始价值: {total_value}")
        
        # 计算当日PnL
        daily_pnl = total_value - self.daily_start_value
        daily_pnl_ratio = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
        
        # 更新投资组合状态
        previous_value = self.portfolio_state.total_value
        self.portfolio_state.total_value = total_value
        self.portfolio_state.cash = cash
        self.portfolio_state.positions = positions.copy()
        self.portfolio_state.daily_pnl = daily_pnl_ratio
        
        # 计算累计PnL
        if previous_value > 0:
            period_return = (total_value - previous_value) / previous_value
            self.portfolio_state.cumulative_pnl += period_return
        
        # 更新最大价值和回撤
        if total_value > self.portfolio_state.max_value:
            self.portfolio_state.max_value = total_value
        
        if self.portfolio_state.max_value > 0:
            self.portfolio_state.drawdown = (self.portfolio_state.max_value - total_value) / self.portfolio_state.max_value
        
        # 更新性能监控
        self.performance_monitor.update(total_value, timestamp)
        
        self.logger.debug(f"组合状态更新: 总价值={total_value:.2f}, 现金={cash:.2f}, 回撤={self.portfolio_state.drawdown:.4f}")
    
    def apply_risk_controls(self, proposed_action: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        应用风险控制，调整建议动作
        
        Args:
            proposed_action: 建议的交易动作
            
        Returns:
            Tuple[Dict[str, Any], List[str]]: (调整后的动作, 风险警告列表)
        """
        warnings_list = []
        adjusted_action = proposed_action.copy()
        
        # 1. 检查仓位限制
        position_warnings = self.check_position_limit(adjusted_action)
        if position_warnings:
            warnings_list.extend(position_warnings)
            adjusted_action = self._adjust_for_position_limit(adjusted_action)
        
        # 2. 检查回撤限制
        drawdown_warnings = self.check_drawdown_limit()
        if drawdown_warnings:
            warnings_list.extend(drawdown_warnings)
            adjusted_action = self._adjust_for_drawdown_limit(adjusted_action)
        
        # 3. 检查日损失限制
        daily_loss_warnings = self.check_daily_loss_limit()
        if daily_loss_warnings:
            warnings_list.extend(daily_loss_warnings)
            adjusted_action = self._adjust_for_daily_loss_limit(adjusted_action)
        
        # 4. 检查现金比例
        cash_warnings = self.check_cash_ratio(adjusted_action)
        if cash_warnings:
            warnings_list.extend(cash_warnings)
            adjusted_action = self._adjust_for_cash_ratio(adjusted_action)
        
        # 记录风险事件
        if warnings_list:
            self._record_risk_event("risk_control_applied", "warning", 
                                   f"应用了风险控制: {'; '.join(warnings_list)}")
        
        return adjusted_action, warnings_list
    
    def check_position_limit(self, action: Dict[str, Any]) -> List[str]:
        """
        检查仓位限制
        
        Args:
            action: 交易动作
            
        Returns:
            List[str]: 违规警告列表
        """
        warnings_list = []
        
        if self.portfolio_state.total_value <= 0:
            return warnings_list
        
        # 检查总仓位比例
        total_position_value = sum(self.portfolio_state.positions.values())
        current_position_ratio = total_position_value / self.portfolio_state.total_value
        
        if current_position_ratio > self.risk_limits.max_position_ratio:
            warnings_list.append(f"总仓位比例超限: {current_position_ratio:.2%} > {self.risk_limits.max_position_ratio:.2%}")
            self.risk_breaches['position_limit'] = True
        
        # 检查单个资产仓位
        for asset, value in self.portfolio_state.positions.items():
            asset_ratio = value / self.portfolio_state.total_value
            if asset_ratio > self.risk_limits.max_single_position:
                warnings_list.append(f"资产 {asset} 仓位超限: {asset_ratio:.2%} > {self.risk_limits.max_single_position:.2%}")
                self.risk_breaches['position_limit'] = True
        
        if not warnings_list:
            self.risk_breaches['position_limit'] = False
        
        return warnings_list
    
    def check_drawdown_limit(self) -> List[str]:
        """
        检查回撤限制
        
        Returns:
            List[str]: 违规警告列表
        """
        warnings_list = []
        
        if self.portfolio_state.drawdown > self.risk_limits.max_drawdown:
            warnings_list.append(f"回撤超限: {self.portfolio_state.drawdown:.2%} > {self.risk_limits.max_drawdown:.2%}")
            self.risk_breaches['drawdown_limit'] = True
        else:
            self.risk_breaches['drawdown_limit'] = False
        
        return warnings_list
    
    def check_daily_loss_limit(self) -> List[str]:
        """
        检查日损失限制
        
        Returns:
            List[str]: 违规警告列表
        """
        warnings_list = []
        
        # 检查当日损失
        if self.portfolio_state.daily_pnl < -self.risk_limits.daily_loss_limit:
            warnings_list.append(f"日损失超限: {self.portfolio_state.daily_pnl:.2%} < -{self.risk_limits.daily_loss_limit:.2%}")
            self.risk_breaches['daily_loss_limit'] = True
        else:
            self.risk_breaches['daily_loss_limit'] = False
        
        return warnings_list
    
    def check_cash_ratio(self, action: Dict[str, Any]) -> List[str]:
        """
        检查现金比例
        
        Args:
            action: 交易动作
            
        Returns:
            List[str]: 违规警告列表
        """
        warnings_list = []
        
        if self.portfolio_state.total_value <= 0:
            return warnings_list
        
        cash_ratio = self.portfolio_state.cash / self.portfolio_state.total_value
        
        if cash_ratio < self.risk_limits.min_cash_ratio:
            warnings_list.append(f"现金比例过低: {cash_ratio:.2%} < {self.risk_limits.min_cash_ratio:.2%}")
        
        return warnings_list
    
    def _adjust_for_position_limit(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """调整动作以符合仓位限制"""
        # 简化实现：减少买入规模
        if 'buy_ratio' in action and action['buy_ratio'] > 0:
            action['buy_ratio'] *= 0.5  # 减半买入
            self.logger.warning("由于仓位限制，减少买入规模")
        
        return action
    
    def _adjust_for_drawdown_limit(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """调整动作以应对回撤超限"""
        # 回撤超限时，停止买入，考虑减仓
        if 'buy_ratio' in action:
            action['buy_ratio'] = 0
        if 'sell_ratio' in action:
            action['sell_ratio'] = min(action.get('sell_ratio', 0) + 0.2, 1.0)
        
        self.logger.warning("由于回撤超限，调整为减仓策略")
        return action
    
    def _adjust_for_daily_loss_limit(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """调整动作以应对日损失超限"""
        # 日损失超限时，停止所有买入
        if 'buy_ratio' in action:
            action['buy_ratio'] = 0
        
        self.logger.warning("由于日损失超限，停止买入")
        return action
    
    def _adjust_for_cash_ratio(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """调整动作以维持现金比例"""
        # 现金不足时，减少买入或增加卖出
        if 'buy_ratio' in action and action['buy_ratio'] > 0:
            action['buy_ratio'] *= 0.8
        
        return action
    
    def _record_risk_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录风险事件
        
        Args:
            event_type: 事件类型
            severity: 严重程度
            message: 事件消息
            data: 附加数据
        """
        event = RiskEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            message=message,
            data=data or {}
        )
        
        self.risk_events.append(event)
        
        # 限制事件历史长度
        if len(self.risk_events) > 1000:
            self.risk_events = self.risk_events[-500:]  # 保留最近500个事件
        
        # 记录日志
        if severity == 'critical':
            self.logger.error(f"风险事件: {message}")
        else:
            self.logger.warning(f"风险事件: {message}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """
        获取当前风险状态
        
        Returns:
            Dict[str, Any]: 风险状态报告
        """
        performance_metrics = self.performance_monitor.calculate_current_metrics()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_state': {
                'total_value': self.portfolio_state.total_value,
                'cash': self.portfolio_state.cash,
                'cash_ratio': self.portfolio_state.cash / self.portfolio_state.total_value if self.portfolio_state.total_value > 0 else 0,
                'positions_count': len(self.portfolio_state.positions),
                'daily_pnl': self.portfolio_state.daily_pnl,
                'cumulative_pnl': self.portfolio_state.cumulative_pnl,
                'current_drawdown': self.portfolio_state.drawdown,
                'max_value': self.portfolio_state.max_value
            },
            'risk_limits': {
                'max_position_ratio': self.risk_limits.max_position_ratio,
                'max_single_position': self.risk_limits.max_single_position,
                'max_drawdown': self.risk_limits.max_drawdown,
                'daily_loss_limit': self.risk_limits.daily_loss_limit,
                'min_cash_ratio': self.risk_limits.min_cash_ratio
            },
            'risk_breaches': self.risk_breaches.copy(),
            'performance_metrics': performance_metrics,
            'recent_events_count': len([e for e in self.risk_events if (datetime.now() - e.timestamp).days <= 1])
        }
        
        # 风险评分 (0-100, 越高越危险)
        risk_score = self._calculate_risk_score()
        status['risk_score'] = risk_score
        
        return status
    
    def _calculate_risk_score(self) -> float:
        """
        计算综合风险评分
        
        Returns:
            float: 风险评分 (0-100)
        """
        score = 0.0
        
        # 回撤风险 (0-30分)
        drawdown_score = min(30, (self.portfolio_state.drawdown / self.risk_limits.max_drawdown) * 30)
        score += drawdown_score
        
        # 仓位风险 (0-25分)
        if self.portfolio_state.total_value > 0:
            total_position_value = sum(self.portfolio_state.positions.values())
            position_ratio = total_position_value / self.portfolio_state.total_value
            position_score = min(25, (position_ratio / self.risk_limits.max_position_ratio) * 25)
            score += position_score
        
        # 日损失风险 (0-25分)
        if self.portfolio_state.daily_pnl < 0:
            daily_loss_score = min(25, (abs(self.portfolio_state.daily_pnl) / self.risk_limits.daily_loss_limit) * 25)
            score += daily_loss_score
        
        # 波动率风险 (0-20分)
        performance_metrics = self.performance_monitor.calculate_current_metrics()
        if 'volatility' in performance_metrics:
            volatility = performance_metrics['volatility']
            # 日波动率超过2%认为高风险
            volatility_score = min(20, (volatility / 0.02) * 20)
            score += volatility_score
        
        return min(100, score)
    
    def get_risk_report(self, days: int = 7) -> Dict[str, Any]:
        """
        生成风险报告
        
        Args:
            days: 报告天数
            
        Returns:
            Dict[str, Any]: 风险报告
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 筛选时间范围内的事件
        period_events = [
            e for e in self.risk_events 
            if start_time <= e.timestamp <= end_time
        ]
        
        # 事件统计
        event_stats = {}
        for event in period_events:
            event_type = event.event_type
            if event_type not in event_stats:
                event_stats[event_type] = {'warning': 0, 'critical': 0}
            event_stats[event_type][event.severity] += 1
        
        # 性能摘要
        performance_summary = self.performance_monitor.get_performance_summary(days)
        
        report = {
            'report_period': {
                'start_date': start_time.isoformat(),
                'end_date': end_time.isoformat(),
                'days': days
            },
            'current_status': self.get_risk_status(),
            'event_summary': {
                'total_events': len(period_events),
                'by_type': event_stats,
                'critical_events': len([e for e in period_events if e.severity == 'critical']),
                'warning_events': len([e for e in period_events if e.severity == 'warning'])
            },
            'performance_summary': performance_summary,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        生成风险管理建议
        
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 基于当前风险状态给出建议
        if self.risk_breaches['drawdown_limit']:
            recommendations.append("当前回撤超限，建议减仓或停止新开仓")
        
        if self.risk_breaches['daily_loss_limit']:
            recommendations.append("当日损失较大，建议暂停交易或降低仓位")
        
        if self.risk_breaches['position_limit']:
            recommendations.append("仓位过重，建议适当减仓以控制风险")
        
        # 基于性能指标的建议
        performance_metrics = self.performance_monitor.calculate_current_metrics()
        
        if 'volatility' in performance_metrics and performance_metrics['volatility'] > 0.03:
            recommendations.append("市场波动较大，建议降低仓位规模")
        
        if 'sharpe_ratio' in performance_metrics and performance_metrics['sharpe_ratio'] < 0:
            recommendations.append("风险调整收益为负，建议重新评估交易策略")
        
        # 现金比例建议
        if self.portfolio_state.total_value > 0:
            cash_ratio = self.portfolio_state.cash / self.portfolio_state.total_value
            if cash_ratio < 0.05:
                recommendations.append("现金比例较低，建议适当增加现金储备")
            elif cash_ratio > 0.3:
                recommendations.append("现金比例较高，可考虑适当增加投资")
        
        if not recommendations:
            recommendations.append("当前风险状况良好，继续保持谨慎的投资策略")
        
        return recommendations
    
    def save_risk_report(self, filepath: str, days: int = 30) -> None:
        """
        保存风险报告到文件
        
        Args:
            filepath: 文件路径
            days: 报告天数
        """
        try:
            report = self.get_risk_report(days)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"风险报告已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存风险报告失败: {e}")
            raise
    
    def reset(self) -> None:
        """重置风险管理器"""
        self.portfolio_state = PortfolioState()
        self.performance_monitor.reset()
        self.risk_events.clear()
        self.current_date = None
        self.daily_start_value = 0.0
        
        for key in self.risk_breaches:
            self.risk_breaches[key] = False
        
        self.logger.info("风险管理器已重置")