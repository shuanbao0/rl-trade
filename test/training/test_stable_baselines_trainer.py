"""
测试Stable-Baselines3训练器模块
"""

import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from src.training.stable_baselines_trainer import StableBaselinesTrainer
from src.environment.trading_environment import TradingEnvironment
from src.utils.config import Config


class TestStableBaselinesTrainer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟数据
        self.mock_data = {
            'Close': np.random.randn(1000).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 1000),
            'Open': np.random.randn(1000).cumsum() + 99,
            'High': np.random.randn(1000).cumsum() + 101,
            'Low': np.random.randn(1000).cumsum() + 98
        }
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        trainer = StableBaselinesTrainer(config=self.config)
        
        assert trainer.config is not None
        assert trainer.model is None  # 初始时模型为空
        assert trainer.env is None    # 初始时环境为空
    
    def test_setup_environment(self):
        """测试环境设置"""
        trainer = StableBaselinesTrainer(config=self.config)
        
        env = trainer.setup_environment(
            data=self.mock_data,
            reward_function="simple_return"
        )
        
        assert env is not None
        assert isinstance(env, TradingEnvironment)
    
    @patch('stable_baselines3.PPO')
    def test_setup_training(self, mock_ppo):
        """测试训练设置"""
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        trainer = StableBaselinesTrainer(config=self.config)
        env = trainer.setup_environment(self.mock_data, "simple_return")
        
        model = trainer.setup_training(
            env=env,
            algorithm="PPO",
            total_timesteps=1000
        )
        
        assert model is not None
        mock_ppo.assert_called_once()
    
    @patch('stable_baselines3.PPO')
    def test_train(self, mock_ppo):
        """测试训练过程"""
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        trainer = StableBaselinesTrainer(config=self.config)
        env = trainer.setup_environment(self.mock_data, "simple_return")
        model = trainer.setup_training(env, "PPO", 1000)
        
        # 执行训练
        trainer.train()
        
        # 验证模型的learn方法被调用
        mock_model.learn.assert_called_once()
    
    @patch('stable_baselines3.PPO')
    def test_evaluate(self, mock_ppo):
        """测试模型评估"""
        mock_model = MagicMock()
        mock_model.predict.return_value = ([0.5], None)
        mock_ppo.return_value = mock_model
        
        trainer = StableBaselinesTrainer(config=self.config)
        env = trainer.setup_environment(self.mock_data, "simple_return")
        trainer.setup_training(env, "PPO", 1000)
        
        # 执行评估
        results = trainer.evaluate(n_episodes=1)
        
        assert 'total_reward' in results
        assert 'episode_length' in results
        assert 'final_portfolio_value' in results
    
    @patch('stable_baselines3.PPO')
    def test_save_load_model(self, mock_ppo):
        """测试模型保存和加载"""
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        trainer = StableBaselinesTrainer(config=self.config)
        env = trainer.setup_environment(self.mock_data, "simple_return")
        trainer.setup_training(env, "PPO", 1000)
        
        # 测试保存
        model_path = f"{self.temp_dir}/test_model"
        trainer.save_model(model_path)
        mock_model.save.assert_called_once_with(model_path)
        
        # 测试加载
        trainer.load_model(model_path)
        mock_ppo.load.assert_called_once_with(model_path)
    
    def test_get_training_metrics(self):
        """测试获取训练指标"""
        trainer = StableBaselinesTrainer(config=self.config)
        
        metrics = trainer.get_training_metrics()
        
        assert isinstance(metrics, dict)
        assert 'training_progress' in metrics
        assert 'evaluation_results' in metrics
        assert 'model_info' in metrics
    
    @patch('stable_baselines3.PPO')
    def test_hyperparameter_tuning(self, mock_ppo):
        """测试超参数调优"""
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        
        trainer = StableBaselinesTrainer(config=self.config)
        env = trainer.setup_environment(self.mock_data, "simple_return")
        
        # 定义超参数搜索空间
        param_space = {
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128]
        }
        
        # 执行超参数调优（模拟）
        best_params = trainer.tune_hyperparameters(
            env=env,
            param_space=param_space,
            n_trials=2
        )
        
        assert isinstance(best_params, dict)
        assert 'learning_rate' in best_params
        assert 'batch_size' in best_params
    
    def test_training_configuration_validation(self):
        """测试训练配置验证"""
        trainer = StableBaselinesTrainer(config=self.config)
        
        # 测试有效配置
        valid_config = {
            'algorithm': 'PPO',
            'total_timesteps': 1000,
            'learning_rate': 0.001
        }
        
        is_valid = trainer.validate_training_config(valid_config)
        assert is_valid is True
        
        # 测试无效配置
        invalid_config = {
            'algorithm': 'INVALID',
            'total_timesteps': -1
        }
        
        is_valid = trainer.validate_training_config(invalid_config)
        assert is_valid is False
    
    def test_training_callbacks(self):
        """测试训练回调函数"""
        trainer = StableBaselinesTrainer(config=self.config)
        
        callback_called = []
        
        def test_callback(epoch, metrics):
            callback_called.append((epoch, metrics))
        
        trainer.add_training_callback(test_callback)
        
        # 模拟触发回调
        trainer._trigger_callbacks(1, {'loss': 0.5})
        
        assert len(callback_called) == 1
        assert callback_called[0][0] == 1
        assert callback_called[0][1]['loss'] == 0.5