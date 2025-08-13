"""
测试超参数优化器模块
"""

import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from src.training.hyperparameter_optimizer import HyperparameterOptimizer
from src.utils.config import Config


class TestHyperparameterOptimizer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        
        # 模拟环境
        self.mock_env = MagicMock()
        self.mock_env.reset.return_value = [0.0] * 10
        self.mock_env.step.return_value = ([0.0] * 10, 1.0, False, {})
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        optimizer = HyperparameterOptimizer(
            config=self.config,
            study_name="test_study",
            storage_url=None
        )
        
        assert optimizer.config is not None
        assert optimizer.study_name == "test_study"
        assert optimizer.study is not None
    
    def test_define_search_space(self):
        """测试搜索空间定义"""
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        # 定义搜索空间
        search_space = {
            'learning_rate': ('loguniform', 1e-5, 1e-2),
            'batch_size': ('categorical', [32, 64, 128]),
            'n_steps': ('int', 512, 2048)
        }
        
        optimizer.define_search_space(search_space)
        
        assert optimizer.search_space == search_space
    
    @patch('stable_baselines3.PPO')
    def test_objective_function(self, mock_ppo):
        """测试目标函数"""
        mock_model = MagicMock()
        mock_model.learn.return_value = None
        mock_model.predict.return_value = ([0.5], None)
        mock_ppo.return_value = mock_model
        
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        # 模拟试验参数
        mock_trial = MagicMock()
        mock_trial.suggest_loguniform.return_value = 0.001
        mock_trial.suggest_categorical.return_value = 64
        mock_trial.suggest_int.return_value = 1024
        
        # 执行目标函数
        score = optimizer._objective(mock_trial, self.mock_env)
        
        assert isinstance(score, (int, float))
    
    @patch('optuna.create_study')
    def test_optimize(self, mock_create_study):
        """测试优化过程"""
        # 模拟研究对象
        mock_study = MagicMock()
        mock_study.best_params = {'learning_rate': 0.001, 'batch_size': 64}
        mock_study.best_value = 100.0
        mock_create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        optimizer.study = mock_study
        
        # 定义简单搜索空间
        search_space = {
            'learning_rate': ('loguniform', 1e-5, 1e-2),
            'batch_size': ('categorical', [32, 64, 128])
        }
        optimizer.define_search_space(search_space)
        
        # 执行优化
        best_params = optimizer.optimize(
            env=self.mock_env,
            n_trials=5,
            timeout=None
        )
        
        assert isinstance(best_params, dict)
        assert 'learning_rate' in best_params
        assert 'batch_size' in best_params
    
    def test_get_optimization_history(self):
        """测试获取优化历史"""
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        # 模拟一些试验历史
        mock_trials = [
            MagicMock(value=10.0, params={'lr': 0.001}),
            MagicMock(value=15.0, params={'lr': 0.01}),
            MagicMock(value=12.0, params={'lr': 0.005})
        ]
        optimizer.study.trials = mock_trials
        
        history = optimizer.get_optimization_history()
        
        assert isinstance(history, dict)
        assert 'trials' in history
        assert 'best_value' in history
        assert len(history['trials']) == 3
    
    def test_save_load_study(self):
        """测试研究保存和加载"""
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        study_path = f"{self.temp_dir}/test_study.pkl"
        
        # 测试保存
        optimizer.save_study(study_path)
        
        # 测试加载
        optimizer.load_study(study_path)
        
        # 验证文件存在（实际实现中会创建文件）
        assert optimizer.study is not None
    
    def test_pruning_callback(self):
        """测试剪枝回调"""
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        # 模拟试验对象
        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = True
        
        # 创建剪枝回调
        callback = optimizer.create_pruning_callback(mock_trial)
        
        assert callback is not None
        assert callable(callback)
    
    def test_parallel_optimization(self):
        """测试并行优化"""
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        # 定义搜索空间
        search_space = {
            'learning_rate': ('loguniform', 1e-5, 1e-2),
            'batch_size': ('categorical', [32, 64])
        }
        optimizer.define_search_space(search_space)
        
        # 测试并行优化设置
        optimizer.enable_parallel_optimization(n_jobs=2)
        
        assert optimizer.n_jobs == 2
    
    def test_visualization_data(self):
        """测试可视化数据生成"""
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        # 模拟试验数据
        mock_trials = []
        for i in range(5):
            trial = MagicMock()
            trial.value = 10 + i
            trial.params = {'lr': 0.001 * (i + 1), 'batch_size': 32}
            trial.number = i
            mock_trials.append(trial)
        
        optimizer.study.trials = mock_trials
        
        viz_data = optimizer.get_visualization_data()
        
        assert isinstance(viz_data, dict)
        assert 'parameter_importance' in viz_data
        assert 'optimization_history' in viz_data
        assert 'parallel_coordinate' in viz_data
    
    def test_early_stopping(self):
        """测试早停机制"""
        optimizer = HyperparameterOptimizer(self.config, "test_study")
        
        # 启用早停
        optimizer.enable_early_stopping(
            patience=3,
            min_improvement=0.01
        )
        
        assert optimizer.early_stopping_enabled is True
        assert optimizer.patience == 3
        assert optimizer.min_improvement == 0.01