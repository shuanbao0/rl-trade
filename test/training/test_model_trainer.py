"""
测试模型训练器模块
"""

import pytest
import numpy as np
import tempfile
import shutil
import pickle
from unittest.mock import patch, MagicMock, mock_open
from src.training.model_trainer import ModelTrainer
from src.utils.config import Config


class TestModelTrainer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟训练数据
        self.mock_data = {
            'features': np.random.randn(1000, 10),
            'targets': np.random.randn(1000, 1),
            'timestamps': np.arange(1000)
        }
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        trainer = ModelTrainer(
            config=self.config,
            model_type="neural_network"
        )
        
        assert trainer.config is not None
        assert trainer.model_type == "neural_network"
        assert trainer.model is None
        assert trainer.training_history == []
    
    def test_prepare_data(self):
        """测试数据准备"""
        trainer = ModelTrainer(self.config, "neural_network")
        
        prepared_data = trainer.prepare_data(
            features=self.mock_data['features'],
            targets=self.mock_data['targets'],
            validation_split=0.2
        )
        
        assert 'X_train' in prepared_data
        assert 'X_val' in prepared_data
        assert 'y_train' in prepared_data
        assert 'y_val' in prepared_data
        
        # 验证数据分割比例
        total_samples = len(self.mock_data['features'])
        expected_train_samples = int(total_samples * 0.8)
        assert len(prepared_data['X_train']) == expected_train_samples
    
    @patch('tensorflow.keras.Sequential')
    def test_build_neural_network_model(self, mock_sequential):
        """测试神经网络模型构建"""
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model
        
        trainer = ModelTrainer(self.config, "neural_network")
        
        model_config = {
            'layers': [64, 32, 1],
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mse'
        }
        
        model = trainer.build_model(
            input_shape=(10,),
            model_config=model_config
        )
        
        assert model is not None
        mock_model.add.assert_called()
        mock_model.compile.assert_called_once()
    
    @patch('sklearn.ensemble.RandomForestRegressor')
    def test_build_random_forest_model(self, mock_rf):
        """测试随机森林模型构建"""
        mock_model = MagicMock()
        mock_rf.return_value = mock_model
        
        trainer = ModelTrainer(self.config, "random_forest")
        
        model_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        model = trainer.build_model(
            input_shape=(10,),
            model_config=model_config
        )
        
        assert model is not None
        mock_rf.assert_called_once_with(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def test_train_neural_network(self):
        """测试神经网络训练"""
        trainer = ModelTrainer(self.config, "neural_network")
        
        # 模拟模型
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {'loss': [0.5, 0.3, 0.2], 'val_loss': [0.6, 0.4, 0.25]}
        mock_model.fit.return_value = mock_history
        
        trainer.model = mock_model
        
        # 准备数据
        prepared_data = trainer.prepare_data(
            self.mock_data['features'],
            self.mock_data['targets']
        )
        
        # 执行训练
        history = trainer.train(
            X_train=prepared_data['X_train'],
            y_train=prepared_data['y_train'],
            X_val=prepared_data['X_val'],
            y_val=prepared_data['y_val'],
            epochs=3,
            batch_size=32
        )
        
        assert history is not None
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        mock_model.fit.assert_called_once()
    
    def test_train_sklearn_model(self):
        """测试sklearn模型训练"""
        trainer = ModelTrainer(self.config, "random_forest")
        
        # 模拟sklearn模型
        mock_model = MagicMock()
        trainer.model = mock_model
        
        # 准备数据
        prepared_data = trainer.prepare_data(
            self.mock_data['features'],
            self.mock_data['targets']
        )
        
        # 执行训练
        trainer.train_sklearn(
            X_train=prepared_data['X_train'],
            y_train=prepared_data['y_train']
        )
        
        mock_model.fit.assert_called_once()
    
    def test_evaluate_model(self):
        """测试模型评估"""
        trainer = ModelTrainer(self.config, "neural_network")
        
        # 模拟模型预测
        mock_model = MagicMock()
        mock_predictions = np.random.randn(100, 1)
        mock_model.predict.return_value = mock_predictions
        
        trainer.model = mock_model
        
        # 执行评估
        X_test = np.random.randn(100, 10)
        y_test = np.random.randn(100, 1)
        
        evaluation_results = trainer.evaluate(X_test, y_test)
        
        assert isinstance(evaluation_results, dict)
        assert 'mae' in evaluation_results
        assert 'mse' in evaluation_results
        assert 'rmse' in evaluation_results
        assert 'r2_score' in evaluation_results
    
    def test_save_model(self):
        """测试模型保存"""
        trainer = ModelTrainer(self.config, "neural_network")
        
        # 模拟TensorFlow模型
        mock_model = MagicMock()
        trainer.model = mock_model
        
        model_path = f"{self.temp_dir}/test_model"
        
        # 执行保存
        trainer.save_model(model_path)
        
        mock_model.save.assert_called_once_with(model_path)
    
    def test_load_model_tensorflow(self):
        """测试TensorFlow模型加载"""
        trainer = ModelTrainer(self.config, "neural_network")
        
        model_path = f"{self.temp_dir}/test_model"
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            trainer.load_model(model_path)
            
            assert trainer.model == mock_model
            mock_load.assert_called_once_with(model_path)
    
    def test_load_model_sklearn(self):
        """测试sklearn模型加载"""
        trainer = ModelTrainer(self.config, "random_forest")
        
        model_path = f"{self.temp_dir}/test_model.pkl"
        
        # 创建模拟的pickle文件
        mock_model = MagicMock()
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        
        trainer.load_model(model_path)
        
        assert trainer.model is not None
    
    def test_cross_validation(self):
        """测试交叉验证"""
        trainer = ModelTrainer(self.config, "random_forest")
        
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.85, 0.82, 0.88, 0.84])
            
            # 模拟模型
            mock_model = MagicMock()
            trainer.model = mock_model
            
            cv_scores = trainer.cross_validate(
                X=self.mock_data['features'],
                y=self.mock_data['targets'].ravel(),
                cv=5,
                scoring='r2'
            )
            
            assert len(cv_scores) == 5
            assert all(score >= 0.8 for score in cv_scores)
    
    def test_feature_importance(self):
        """测试特征重要性"""
        trainer = ModelTrainer(self.config, "random_forest")
        
        # 模拟具有feature_importances_属性的模型
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.random.rand(10)
        trainer.model = mock_model
        
        importance = trainer.get_feature_importance()
        
        assert len(importance) == 10
        assert all(0 <= imp <= 1 for imp in importance)
    
    def test_training_callbacks(self):
        """测试训练回调"""
        trainer = ModelTrainer(self.config, "neural_network")
        
        callback_data = []
        
        def test_callback(epoch, logs):
            callback_data.append((epoch, logs))
        
        trainer.add_callback(test_callback)
        
        # 模拟触发回调
        trainer._trigger_callbacks(1, {'loss': 0.5})
        
        assert len(callback_data) == 1
        assert callback_data[0][0] == 1
        assert callback_data[0][1]['loss'] == 0.5
    
    def test_get_training_summary(self):
        """测试获取训练摘要"""
        trainer = ModelTrainer(self.config, "neural_network")
        
        # 添加一些训练历史
        trainer.training_history = [
            {'epoch': 1, 'loss': 0.5, 'val_loss': 0.6},
            {'epoch': 2, 'loss': 0.3, 'val_loss': 0.4}
        ]
        
        summary = trainer.get_training_summary()
        
        assert isinstance(summary, dict)
        assert 'model_type' in summary
        assert 'total_epochs' in summary
        assert 'best_loss' in summary
        assert 'training_time' in summary