"""
Direct PnL Reward Function
直接基于盈亏的奖励函数

Purpose: 解决实验003A-005中奖励-回报完全脱钩的致命问题
Design: 确保奖励与实际交易盈亏100%相关

Key Features:
- 直接基于投资组合价值变化计算奖励
- 考虑EURUSD外汇交易特有的成本结构
- 数值稳定性控制，避免异常奖励值
- 实时相关性验证机制
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base_reward import BaseRewardScheme


class DirectPnLReward(BaseRewardScheme):
    """
    直接基于盈亏的奖励函数
    
    解决历史实验中奖励与实际回报完全脱钩的问题
    确保奖励函数与实际交易表现强相关 (correlation > 0.9)
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # EURUSD外汇交易参数
        self.initial_balance = getattr(config, 'initial_balance', 10000.0)
        self.pip_value = 0.0001  # EURUSD点值
        self.typical_spread = 1.2  # 典型点差（点）
        self.transaction_cost_rate = 0.0002  # 0.2%交易成本
        
        # 奖励函数参数
        self.reward_scale = 100  # 将回报率转换为合理数值范围
        self.clip_range = (-10.0, 10.0)  # 奖励值限制在±10范围内
        self.position_penalty_rate = 0.001  # 轻微的仓位持有成本
        
        # 状态跟踪
        self.prev_portfolio_value = self.initial_balance
        self.prev_action = 0.0
        self.episode_rewards = []
        self.episode_returns = []
        
        # 相关性监控
        self.correlation_history = []
        self.min_correlation_threshold = 0.8
        
        from ...utils.logger import setup_logger
        self.logger = setup_logger("DirectPnLReward")
        self.logger.info(f"DirectPnLReward初始化 - 初始资金: ${self.initial_balance:,.0f}")
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: Dict, trade_info: Dict, step: int, **kwargs) -> float:
        """
        直接基于投资组合价值变化计算奖励
        
        Args:
            portfolio_value: 当前投资组合价值
            action: 当前动作 (position in [-1, 1])
            price: 当前价格
            portfolio_info: 投资组合信息
            trade_info: 交易信息
            step: 当前步数
            
        Returns:
            reward: 奖励值，与实际盈亏高度相关
        """
        
        # 1. 计算实际收益率
        current_portfolio = portfolio_value
        prev_portfolio = self.prev_portfolio_value
        
        if prev_portfolio <= 0:
            self.logger.warning(f"异常的投资组合价值: prev={prev_portfolio}")
            return 0.0
            
        actual_return_rate = (current_portfolio - prev_portfolio) / prev_portfolio
        
        # 2. 计算交易成本
        position_change = abs(action - self.prev_action) if hasattr(self, 'prev_action') else 0.0
        transaction_cost_rate = position_change * self.transaction_cost_rate
        
        # 3. 计算持仓成本 (鼓励适度交易)
        position_cost_rate = abs(action) * self.position_penalty_rate
        
        # 4. 净收益率 = 实际收益率 - 交易成本 - 持仓成本
        net_return_rate = actual_return_rate - transaction_cost_rate - position_cost_rate
        
        # 5. 转换为奖励值 (放大到合理范围)
        raw_reward = net_return_rate * self.reward_scale
        
        # 6. 数值稳定性控制
        final_reward = np.clip(raw_reward, self.clip_range[0], self.clip_range[1])
        
        # 7. 记录用于相关性监控
        self.episode_rewards.append(final_reward)
        self.episode_returns.append(actual_return_rate * 100)  # 转换为百分比
        
        # 8. 状态更新
        self.prev_action = action
        self.prev_portfolio_value = current_portfolio
        
        # 9. 异常值检查
        if not np.isfinite(final_reward):
            self.logger.error(f"检测到异常奖励值: {final_reward}")
            return 0.0
            
        return final_reward
    
    def reset(self):
        """重置奖励函数状态"""
        # 基类没有reset方法，直接处理本类的重置逻辑
        
        # 计算并记录本episode的相关性
        if len(self.episode_rewards) > 5 and len(self.episode_returns) > 5:
            correlation = self._calculate_correlation()
            self.correlation_history.append(correlation)
            
            # 相关性监控报告
            if len(self.correlation_history) % 10 == 0:  # 每10个episode报告一次
                self._report_correlation_status()
        
        # 重置episode状态
        self.episode_rewards = []
        self.episode_returns = []
        self.prev_portfolio_value = self.initial_balance
        self.prev_action = 0.0
    
    def _calculate_correlation(self) -> float:
        """计算奖励与回报的相关性"""
        try:
            if len(self.episode_rewards) < 2 or len(self.episode_returns) < 2:
                return 0.0
                
            correlation = np.corrcoef(self.episode_rewards, self.episode_returns)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.warning(f"计算相关性失败: {e}")
            return 0.0
    
    def _report_correlation_status(self):
        """报告相关性监控状态"""
        if not self.correlation_history:
            return
            
        recent_correlations = self.correlation_history[-10:]  # 最近10次
        mean_correlation = np.mean(recent_correlations)
        
        status = "✅ GOOD" if mean_correlation >= self.min_correlation_threshold else "⚠️ WARNING"
        
        self.logger.info(f"奖励-回报相关性监控 {status}")
        self.logger.info(f"  最近10次平均相关性: {mean_correlation:.3f}")
        self.logger.info(f"  目标阈值: {self.min_correlation_threshold}")
        self.logger.info(f"  总计算次数: {len(self.correlation_history)}")
        
        # 相关性过低警告
        if mean_correlation < self.min_correlation_threshold:
            self.logger.warning("⚠️ 奖励-回报相关性过低！可能存在奖励函数设计问题")
    
    def get_reward_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        current_correlation = self._calculate_correlation() if self.episode_rewards else 0.0
        avg_correlation = np.mean(self.correlation_history) if self.correlation_history else 0.0
        
        return {
            "name": "DirectPnLReward",
            "description": "直接基于投资组合盈亏的奖励函数",
            "version": "1.0.0",
            "designed_for": "EURUSD外汇交易",
            
            # 设计参数
            "initial_balance": self.initial_balance,
            "transaction_cost_rate": self.transaction_cost_rate,
            "reward_scale": self.reward_scale,
            "reward_range": f"[{self.clip_range[0]}, {self.clip_range[1]}]",
            
            # 性能指标
            "current_episode_correlation": current_correlation,
            "average_correlation": avg_correlation,
            "correlation_threshold": self.min_correlation_threshold,
            "correlation_status": "GOOD" if avg_correlation >= self.min_correlation_threshold else "WARNING",
            
            # 计数统计
            "episodes_calculated": len(self.correlation_history),
            "current_episode_length": len(self.episode_rewards),
            
            # 预期表现
            "expected_correlation": "> 0.9",
            "expected_reward_range": "[-10, +10]",
            "design_objective": "解决实验003A-005的奖励-回报脱钩问题"
        }
    
    def validate_reward_function(self) -> Dict[str, Any]:
        """验证奖励函数的有效性"""
        validation_result = {
            "overall_status": "UNKNOWN",
            "issues_found": [],
            "recommendations": []
        }
        
        # 1. 检查相关性
        if self.correlation_history:
            avg_correlation = np.mean(self.correlation_history)
            
            if avg_correlation >= 0.8:
                validation_result["correlation_status"] = "EXCELLENT"
            elif avg_correlation >= 0.5:
                validation_result["correlation_status"] = "ACCEPTABLE"
                validation_result["issues_found"].append("相关性偏低，建议检查奖励函数参数")
            else:
                validation_result["correlation_status"] = "POOR"
                validation_result["issues_found"].append("相关性严重偏低，奖励函数设计存在问题")
        
        # 2. 检查数值稳定性
        if self.episode_rewards:
            reward_range = (min(self.episode_rewards), max(self.episode_rewards))
            
            if reward_range[0] >= self.clip_range[0] and reward_range[1] <= self.clip_range[1]:
                validation_result["numerical_stability"] = "GOOD"
            else:
                validation_result["numerical_stability"] = "WARNING" 
                validation_result["issues_found"].append("奖励值超出预期范围")
        
        # 3. 整体状态评估
        if len(validation_result["issues_found"]) == 0:
            validation_result["overall_status"] = "HEALTHY"
        elif len(validation_result["issues_found"]) <= 2:
            validation_result["overall_status"] = "WARNING"
        else:
            validation_result["overall_status"] = "CRITICAL"
        
        return validation_result
    
    def __str__(self) -> str:
        info = self.get_reward_info()
        return f"DirectPnLReward(correlation={info['average_correlation']:.3f}, episodes={info['episodes_calculated']})"
    
    def __repr__(self) -> str:
        return self.__str__()


# 注册奖励函数
def create_direct_pnl_reward(config=None, **kwargs):
    """创建DirectPnLReward实例的工厂函数"""
    if config is None:
        config = type('Config', (), {})()
    
    # 从kwargs更新配置
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return DirectPnLReward(config)


# 向系统注册
if __name__ == "__main__":
    # 测试奖励函数
    config = type('Config', (), {})()
    config.initial_balance = 10000
    
    reward_fn = DirectPnLReward(config)
    
    # 模拟测试
    print("测试DirectPnLReward:")
    print(reward_fn.get_reward_info())
    
    # 模拟一个episode
    prev_portfolio = 10000
    for i in range(10):
        current_portfolio = prev_portfolio * (1 + np.random.normal(0, 0.01))  # ±1%随机变化
        action = np.random.uniform(-1, 1)
        
        reward = reward_fn.calculate_reward(prev_portfolio, current_portfolio, action)
        actual_return = (current_portfolio - prev_portfolio) / prev_portfolio * 100
        
        print(f"Step {i+1}: 回报={actual_return:+6.2f}%, 奖励={reward:+6.2f}")
        prev_portfolio = current_portfolio
    
    # 重置并检查相关性
    reward_fn.reset()
    validation = reward_fn.validate_reward_function()
    print(f"\n验证结果: {validation}")