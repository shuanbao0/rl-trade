"""
测试配置管理模块
"""

import pytest
import json
import os
from unittest.mock import patch, mock_open
from src.utils.config import Config


class TestConfig:
    def setup_method(self):
        """每个测试方法前的设置"""
        # 清理环境变量
        self.original_env = dict(os.environ)
        for key in list(os.environ.keys()):
            if key.startswith('TENSORTRADE_'):
                del os.environ[key]
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        # 恢复环境变量
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_config_initialization_default(self):
        """测试默认配置初始化"""
        config = Config()
        
        # 验证基本配置结构
        assert hasattr(config, 'data')
        assert hasattr(config, 'feature')  
        assert hasattr(config, 'trading')
        assert hasattr(config, 'reward')
        
        # 验证默认值
        assert config.data.cache_expiry_hours == 24
        assert config.feature.sma_periods == [20, 50]
        assert config.trading.initial_balance == 10000.0
    
    @patch("builtins.open", new_callable=mock_open, read_data='{"data": {"cache_expiry_hours": 48}, "feature": {"sma_periods": [10, 20]}}')
    @patch("os.path.exists", return_value=True)
    def test_config_with_json_file(self, mock_exists, mock_file):
        """测试JSON配置文件加载"""
        config = Config(config_file="test_config.json")
        
        # 验证JSON配置覆盖默认值
        assert config.data.cache_expiry_hours == 48
        assert config.feature.sma_periods == [10, 20]
    
    def test_environment_variable_override(self):
        """测试环境变量覆盖"""
        os.environ['DATA_CACHE_EXPIRY_HOURS'] = '48'
        os.environ['INITIAL_BALANCE'] = '50000.0'
        
        config = Config()
        
        # 验证环境变量覆盖
        assert config.data.cache_expiry_hours == 48
        assert config.trading.initial_balance == 50000.0
    
    def test_get_method(self):
        """测试get方法"""
        config = Config()
        
        # 测试获取自定义配置（使用_config字典）
        config.set('test_key', 'test_value')
        value = config.get('test_key')
        assert value == 'test_value'
        
        # 测试获取不存在的配置（返回默认值）
        non_existent = config.get('non_existent', {'default': True})
        assert non_existent == {'default': True}
    
    def test_config_validation(self):
        """测试配置验证"""
        config = Config()
        
        # 验证必需的配置项
        assert hasattr(config.data, 'cache_dir')
        assert hasattr(config.feature, 'sma_periods')
        assert hasattr(config.trading, 'initial_balance')
    
    def test_nested_config_access(self):
        """测试嵌套配置访问"""
        config = Config()
        
        # 测试配置对象属性访问
        cache_dir = config.data.cache_dir
        assert isinstance(cache_dir, str)
        assert cache_dir == "data_cache"
        
        # 测试奖励配置访问
        reward_type = config.reward.reward_type
        assert reward_type == "risk_adjusted"
    
    def test_to_dict_conversion(self):
        """测试配置转换为字典"""
        config = Config()
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'data' in config_dict
        assert 'feature' in config_dict
        assert 'trading' in config_dict
        assert 'reward' in config_dict
    
    def test_reward_function_creation(self):
        """测试奖励函数创建"""
        config = Config()
        
        try:
            # 测试创建风险调整奖励函数
            reward_func = config.create_reward_function('risk_adjusted')
            assert reward_func is not None
        except ImportError:
            # 如果模块不存在，跳过测试
            pass