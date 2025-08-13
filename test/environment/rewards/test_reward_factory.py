"""
测试奖励函数工厂模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.reward_factory import create_reward_function, list_available_reward_types


class TestRewardFactory:
    def test_list_available_reward_types(self):
        """测试列出可用奖励类型"""
        available_types = list_available_reward_types()
        
        assert isinstance(available_types, list)
        assert len(available_types) > 0
        
        # 验证包含基本奖励类型
        expected_types = ['risk_adjusted', 'simple_return', 'profit_loss']
        for expected_type in expected_types:
            assert expected_type in available_types
    
    def test_create_basic_reward_functions(self):
        """测试创建基本奖励函数"""
        basic_types = ['risk_adjusted', 'simple_return', 'profit_loss']
        
        for reward_type in basic_types:
            reward_func = create_reward_function(reward_type, initial_balance=10000.0)
            
            assert reward_func is not None
            assert hasattr(reward_func, 'get_reward')
            assert hasattr(reward_func, 'get_reward_info')
            
            # 测试基本功能
            info = reward_func.get_reward_info()
            assert isinstance(info, dict)
            assert 'name' in info
    
    def test_create_advanced_reward_functions(self):
        """测试创建高级奖励函数"""
        advanced_types = ['diversified', 'dynamic_sortino', 'curriculum_reward']
        
        for reward_type in advanced_types:
            try:
                reward_func = create_reward_function(reward_type, initial_balance=10000.0)
                
                assert reward_func is not None
                info = reward_func.get_reward_info()
                assert isinstance(info, dict)
                
            except (ImportError, NotImplementedError):
                # 某些高级奖励函数可能需要额外依赖
                pass
    
    def test_invalid_reward_type(self):
        """测试无效奖励类型"""
        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            create_reward_function('invalid_reward_type')
    
    def test_reward_function_parameters(self):
        """测试奖励函数参数传递"""
        params = {
            'initial_balance': 20000.0,
            'risk_free_rate': 0.03,
            'window_size': 30
        }
        
        reward_func = create_reward_function('risk_adjusted', **params)
        
        assert reward_func.initial_balance == 20000.0
        # 其他参数的验证取决于具体实现
    
    def test_reward_function_with_mock_env(self):
        """测试奖励函数与模拟环境"""
        reward_func = create_reward_function('simple_return', initial_balance=10000.0)
        
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = reward_func.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)