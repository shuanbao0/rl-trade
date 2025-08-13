"""
测试交易环境模块
"""

import pytest
import numpy as np
import pandas as pd
from src.environment.trading_environment import TradingEnvironment
from src.utils.config import Config


class TestTradingEnvironment:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105, 
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        self.env = TradingEnvironment(
            df=self.test_data,
            config=self.config
        )
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        assert self.env.config is not None
        assert self.env.df is not None
        assert hasattr(self.env, 'action_space')
        assert hasattr(self.env, 'observation_space')
    
    def test_reset(self):
        """测试环境重置"""
        observation, info = self.env.reset()
        
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert info is not None
        assert isinstance(info, dict)
    
    def test_step(self):
        """测试环境步进"""
        self.env.reset()
        
        # 执行一个动作
        action = 0.5
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        assert observation is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_action_space_bounds(self):
        """测试动作空间边界"""
        # 测试边界动作
        self.env.reset()
        
        # 最大动作
        obs, reward, terminated, truncated, info = self.env.step(1.0)
        assert obs is not None
        
        # 最小动作  
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(-1.0)
        assert obs is not None