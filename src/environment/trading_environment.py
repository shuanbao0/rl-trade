"""
Modern Trading Environment
现代化交易环境 - 已迁移至Gymnasium框架

主要功能:
1. 标准Gymnasium接口 (替代TensorTrade)
2. 简化的投资组合管理系统
3. 动态仓位调整动作空间
4. 现代化数据处理和观察空间
5. 兼容17种高级奖励函数
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


@dataclass
class PortfolioState:
    """投资组合状态 - 替代TensorTrade Portfolio"""
    cash: float = 10000.0
    shares: float = 0.0
    total_value: float = 10000.0
    net_worth_history: List[float] = None
    transaction_costs: float = 0.001  # 0.1% 交易费用
    
    def __post_init__(self):
        if self.net_worth_history is None:
            self.net_worth_history = [self.total_value]



class SimplePortfolioManager:
    """
    简化投资组合管理器 - 替代复杂的TensorTrade OMS系统
    """
    
    def __init__(self, initial_balance: float = 10000.0, transaction_costs: float = 0.001):
        """
        初始化投资组合管理器
        
        Args:
            initial_balance: 初始资金
            transaction_costs: 交易费用率 (默认0.1%)
        """
        self.initial_balance = initial_balance
        self.transaction_costs = transaction_costs
        self.reset()
        
    def reset(self):
        """重置投资组合状态"""
        self.state = PortfolioState(
            cash=self.initial_balance,
            shares=0.0,
            total_value=self.initial_balance,
            transaction_costs=self.transaction_costs
        )
        
    def execute_action(self, action: float, current_price: float) -> Dict[str, Any]:
        """
        执行交易动作 - 替代TensorTrade的OrderManagement
        
        Args:
            action: 动作值 [-1, 1]，正值买入，负值卖出
            current_price: 当前价格
            
        Returns:
            Dict: 交易信息 {executed: bool, amount: float, cost: float}
        """
        action = np.clip(action, -1.0, 1.0)
        
        # 最小交易阈值
        if abs(action) < 0.01:
            return {"executed": False, "amount": 0.0, "cost": 0.0}
            
        if action > 0:  # 买入
            return self._execute_buy(action, current_price)
        elif action < 0:  # 卖出
            return self._execute_sell(abs(action), current_price)
        else:
            return {"executed": False, "amount": 0.0, "cost": 0.0}
    
    def _execute_buy(self, action: float, price: float) -> Dict[str, Any]:
        """执行买入操作"""
        available_cash = self.state.cash
        max_purchase_amount = available_cash / (1 + self.transaction_costs)
        purchase_amount = max_purchase_amount * action
        
        if purchase_amount < 1.0:  # 最小交易金额
            return {"executed": False, "amount": 0.0, "cost": 0.0}
        
        shares_to_buy = purchase_amount / price
        total_cost = shares_to_buy * price * (1 + self.transaction_costs)
        
        if total_cost <= self.state.cash:
            self.state.cash -= total_cost
            self.state.shares += shares_to_buy
            return {"executed": True, "amount": shares_to_buy, "cost": total_cost}
        
        return {"executed": False, "amount": 0.0, "cost": 0.0}
    
    def _execute_sell(self, action: float, price: float) -> Dict[str, Any]:
        """执行卖出操作"""
        shares_to_sell = self.state.shares * action
        
        if shares_to_sell < 0.01:  # 最小交易股数
            return {"executed": False, "amount": 0.0, "cost": 0.0}
        
        gross_revenue = shares_to_sell * price
        net_revenue = gross_revenue * (1 - self.transaction_costs)
        
        self.state.cash += net_revenue
        self.state.shares -= shares_to_sell
        
        return {"executed": True, "amount": -shares_to_sell, "cost": -net_revenue}
    
    def update_value(self, current_price: float):
        """更新投资组合总价值"""
        self.state.total_value = self.state.cash + self.state.shares * current_price
        self.state.net_worth_history.append(self.state.total_value)
    
    def get_portfolio_info(self, current_price: float) -> Dict[str, float]:
        """获取投资组合信息"""
        return {
            "cash": self.state.cash,
            "shares": self.state.shares,
            "share_value": self.state.shares * current_price,
            "total_value": self.state.total_value,
            "cash_ratio": self.state.cash / self.state.total_value if self.state.total_value > 0 else 1.0,
            "return_pct": (self.state.total_value - self.initial_balance) / self.initial_balance * 100
        }


class TradingEnvironment(gym.Env):
    """
    现代化交易环境 - 基于Gymnasium框架
    
    提供标准化的强化学习交易环境接口，完全替代TensorTrade依赖
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[Config] = None,
        initial_balance: float = 10000.0,
        window_size: int = 50,
        feature_columns: List[str] = None,
        reward_function=None,
        transaction_costs: float = 0.001,
        max_episode_steps: Optional[int] = None
    ):
        """
        初始化现代化交易环境
        
        Args:
            df: 包含价格和特征的数据DataFrame
            config: 配置对象
            initial_balance: 初始资金
            window_size: 观察窗口大小
            feature_columns: 特征列名列表
            reward_function: 奖励函数对象
            transaction_costs: 交易费用率
            max_episode_steps: 最大步数
        """
        super().__init__()
        
        # 基础配置
        self.df = df.reset_index(drop=True)
        self.config = config or Config()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_costs = transaction_costs
        self.max_episode_steps = max_episode_steps or len(df) - window_size
        
        # 数据列配置 - 自动检测价格列
        if 'Close' in self.df.columns:
            self.price_column = 'Close'
        elif 'close' in self.df.columns:
            self.price_column = 'close'
        else:
            # 尝试其他可能的价格列名
            price_candidates = ['price', 'Price', 'close_price', 'Close_Price', 'adj_close', 'Adj Close']
            self.price_column = None
            for candidate in price_candidates:
                if candidate in self.df.columns:
                    self.price_column = candidate
                    break
            
            if self.price_column is None:
                # 如果都找不到，使用第一个数值列
                numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    self.price_column = numeric_columns[0]
                else:
                    raise ValueError("无法找到合适的价格列")
        
        if feature_columns is None:
            # 默认使用数值列作为特征（排除价格列）
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numeric_columns if col != self.price_column]
        else:
            self.feature_columns = feature_columns
        
        self.n_features = len(self.feature_columns)
        
        # 投资组合管理器 - 替代TensorTrade Portfolio
        self.portfolio_manager = SimplePortfolioManager(initial_balance, transaction_costs)
        
        # 奖励函数
        self.reward_function = reward_function
        
        # 环境状态
        self.current_step = 0
        self.done = False
        self.info = {}
        
        # Gymnasium空间定义
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 观察空间：窗口大小 × 特征数量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), 
            dtype=np.float32
        )
        
        # 初始化日志
        self.logger = setup_logger(
            name="TradingEnvironment",
            level="INFO",
            log_file=get_default_log_file("trading_environment")
        )
        
        # 数据预处理
        self._prepare_data()
        
        self.logger.info(f"现代化交易环境初始化完成: {len(self.df)} 个时间步，{self.n_features} 个特征")
        
        # 为了保持向后兼容性，添加一些原有属性
        self.env = self  # 环境引用自身
        self.features_data = df
    
    def _prepare_data(self):
        """数据预处理和标准化"""
        # 检查必需的列
        if self.price_column not in self.df.columns:
            raise ValueError(f"价格列 '{self.price_column}' 不存在于数据中")
        
        for col in self.feature_columns:
            if col not in self.df.columns:
                raise ValueError(f"特征列 '{col}' 不存在于数据中")
        
        # 计算标准化参数
        feature_data = self.df[self.feature_columns].values
        self.feature_mean = np.nanmean(feature_data, axis=0)
        self.feature_std = np.nanstd(feature_data, axis=0) + 1e-8  # 避免除零
        
        # 标准化特征数据
        self.normalized_features = (feature_data - self.feature_mean) / self.feature_std
        
        # 处理NaN值
        self.normalized_features = np.nan_to_num(self.normalized_features, 0.0)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境 - Gymnasium标准接口"""
        super().reset(seed=seed)
        
        # 重置状态
        self.current_step = min(self.window_size, len(self.df) - 1)
        self.done = False
        self.portfolio_manager.reset()
        
        # 获取初始观察
        observation = self._get_observation()
        
        # 更新投资组合价值 - 确保索引不越界
        if self.current_step < len(self.df):
            current_price = self.df[self.price_column].iloc[self.current_step]
        else:
            current_price = self.df[self.price_column].iloc[-1]
        self.portfolio_manager.update_value(current_price)
        
        # 环境信息
        self.info = {
            "step": self.current_step,
            "portfolio": self.portfolio_manager.get_portfolio_info(current_price),
            "price": current_price,
            "total_return": 0.0
        }
        
        return observation, self.info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步交易 - Gymnasium标准接口"""
        if self.done:
            raise RuntimeError("Environment已经结束，请调用reset()")
        
        # 执行动作
        current_price = self.df[self.price_column].iloc[self.current_step]
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # 执行交易
        trade_info = self.portfolio_manager.execute_action(action_value, current_price)
        
        # 更新投资组合价值
        self.portfolio_manager.update_value(current_price)
        portfolio_info = self.portfolio_manager.get_portfolio_info(current_price)
        
        # 计算奖励
        reward = self._calculate_reward(action_value, current_price, portfolio_info, trade_info)
        
        # 更新步数
        self.current_step += 1
        
        # 检查终止条件
        terminated = self._check_terminated()
        truncated = self.current_step >= (len(self.df) - 1) or \
                   (self.max_episode_steps and self.current_step >= self.max_episode_steps)
        
        self.done = terminated or truncated
        
        # 获取新观察
        observation = self._get_observation()
        
        # 更新环境信息
        self.info = {
            "step": self.current_step,
            "portfolio": portfolio_info,
            "price": current_price,
            "action": action_value,
            "trade_info": trade_info,
            "total_return": (portfolio_info["total_value"] - self.initial_balance) / self.initial_balance * 100,
            "terminated": terminated,
            "truncated": truncated
        }
        
        return observation, reward, terminated, truncated, self.info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(self.normalized_features), end_idx)
        
        observation = self.normalized_features[start_idx:end_idx]
        
        # 如果观察窗口不足，使用零填充
        if observation.shape[0] < self.window_size:
            padding_size = self.window_size - observation.shape[0]
            padding = np.zeros((padding_size, self.n_features))
            observation = np.vstack([padding, observation])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, action: float, price: float, portfolio_info: Dict, trade_info: Dict) -> float:
        """计算奖励 - 兼容17种奖励函数"""
        if self.reward_function is not None:
            try:
                # 为奖励函数提供必要信息
                reward_data = {
                    "portfolio_value": portfolio_info["total_value"],
                    "action": action,
                    "price": price,
                    "portfolio_info": portfolio_info,
                    "trade_info": trade_info,
                    "step": self.current_step
                }
                return self.reward_function.calculate_reward(**reward_data)
            except Exception as e:
                self.logger.warning(f"奖励函数计算错误: {e}，使用默认奖励")
                return self._default_reward(portfolio_info)
        else:
            return self._default_reward(portfolio_info)
    
    def _default_reward(self, portfolio_info: Dict) -> float:
        """默认奖励函数：简单回报"""
        if len(self.portfolio_manager.state.net_worth_history) < 2:
            return 0.0
        
        # 计算相对收益
        previous_value = self.portfolio_manager.state.net_worth_history[-2]
        current_value = portfolio_info["total_value"]
        
        if previous_value > 0:
            return (current_value - previous_value) / previous_value
        return 0.0
    
    def _check_terminated(self) -> bool:
        """检查终止条件"""
        # 资金耗尽
        if self.portfolio_manager.state.total_value <= 0.1 * self.initial_balance:
            return True
        return False
    
    def render(self, mode="human"):
        """渲染环境状态"""
        if mode == "human":
            current_price = self.df[self.price_column].iloc[self.current_step]
            portfolio_info = self.portfolio_manager.get_portfolio_info(current_price)
            
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Cash: ${portfolio_info['cash']:.2f}")
            print(f"Shares: {portfolio_info['shares']:.2f}")
            print(f"Total Value: ${portfolio_info['total_value']:.2f}")
            print(f"Return: {portfolio_info['return_pct']:.2f}%")
            print("-" * 40)
    
    def create_environment(self, features_data: pd.DataFrame) -> 'TradingEnvironment':
        """
        创建现代化交易环境 - 保持原API兼容性
        
        Args:
            features_data: 特征数据，包含价格和其他特征
            
        Returns:
            现代化环境实例 (自身)
        """
        try:
            # 更新数据并重新预处理 - 简化的现代化实现
            self.df = features_data.copy()
            self.features_data = features_data.copy()
            
            # 更新特征列配置 - 现代化简化处理
            feature_columns = [col for col in features_data.columns if col != 'Close']
            
            # 特征选择逻辑保持不变，但简化实现
            if hasattr(self, 'max_features') and self.max_features is not None:
                important_features = [
                    'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Stochastic_K', 'Stochastic_D',
                    'BB_Upper_20', 'BB_Lower_20', 'BB_Width_20', 'BB_Position_20', 'ATR_14',
                    'OBV', 'Volume_Ratio_5d', 'Return_5d', 'Return_10d', 'Return_20d',
                    'LogReturn_5d', 'LogReturn_10d', 'LogReturn_20d', 'Volatility_5d', 'Volatility_10d', 'Volatility_20d',
                    'Mean_5d', 'Mean_10d', 'Mean_20d', 'Std_5d', 'Std_10d', 'Std_20d',
                    'Price_Position_5d', 'Price_Position_20d'
                ]
                selected_columns = [col for col in feature_columns if any(feat in col for feat in important_features)][:self.max_features]
            else:
                selected_columns = feature_columns
            
            # 更新特征列配置
            self.feature_columns = selected_columns
            self.n_features = len(selected_columns)
            
            # 重新预处理数据
            self._prepare_data()
            
            self.logger.info(f"环境数据更新完成: 使用 {len(selected_columns)} 个特征")
            
            # 返回自身作为环境实例 - 保持API兼容性
            return self
            
        except Exception as e:
            self.logger.error(f"环境数据更新失败: {e}")
            raise
    
    # 保持向后兼容性的旧方法名
    def get_env(self):
        """获取环境实例 - 向后兼容"""
        return self
    
    def get_portfolio_info(self) -> Dict[str, Any]:
        """获取投资组合信息 - 兼容性方法"""
        if self.current_step == 0:
            current_price = self.df[self.price_column].iloc[self.window_size]
        else:
            current_price = self.df[self.price_column].iloc[self.current_step]
        
        return self.portfolio_manager.get_portfolio_info(current_price)


# 创建现代化交易环境的工厂函数 - 保持向后兼容性
def create_trading_environment(
    df: pd.DataFrame,
    config: Optional[Config] = None,
    initial_balance: float = 10000.0,
    window_size: int = 50,
    feature_columns: List[str] = None,
    reward_function=None,
    transaction_costs: float = 0.001,
    max_episode_steps: Optional[int] = None
) -> TradingEnvironment:
    """
    创建现代化交易环境的工厂函数
    
    Args:
        df: 数据DataFrame
        config: 配置对象
        initial_balance: 初始资金
        window_size: 观察窗口大小
        feature_columns: 特征列名列表
        reward_function: 奖励函数
        transaction_costs: 交易费用率
        max_episode_steps: 最大步数
        
    Returns:
        TradingEnvironment: 配置好的交易环境
    """
    return TradingEnvironment(
        df=df,
        config=config,
        initial_balance=initial_balance,
        window_size=window_size,
        feature_columns=feature_columns,
        reward_function=reward_function,
        transaction_costs=transaction_costs,
        max_episode_steps=max_episode_steps
    )
