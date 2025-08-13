"""
完整的训练流水线管理器

整合数据处理、超参数优化、模型训练、评估和部署的完整工作流。
支持自动化训练流程和实验管理。
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from .stable_baselines_trainer import StableBaselinesTrainer
from .hyperparameter_optimizer import HyperparameterOptimizer
from ..data.data_manager import DataManager
from ..features.feature_engineer import FeatureEngineer
from ..environment.rewards import create_reward_function
from ..utils.logger import setup_logger
from ..utils.config import Config


class TrainingPipeline:
    """
    完整的训练流水线管理器
    
    负责管理从数据获取到模型部署的整个训练流程
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 experiment_name: Optional[str] = None,
                 base_path: str = "experiments"):
        """
        初始化训练流水线
        
        Args:
            config: 配置对象
            experiment_name: 实验名称
            base_path: 实验基础路径
        """
        self.config = config or Config()
        self.logger = setup_logger('TrainingPipeline')
        
        # 实验管理
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_path = Path(base_path) / self.experiment_name
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        # 组件初始化
        self.data_manager = DataManager(config)
        self.feature_engineer = FeatureEngineer(config)
        self.optimizer = None
        self.trainer = None
        
        # 实验状态
        self.experiment_config = {}
        self.results = {}
        self.models = {}
        
        self.logger.info(f"初始化训练流水线: {self.experiment_name}")
        self.logger.info(f"实验路径: {self.experiment_path}")
    
    def prepare_data(self, 
                    symbol: str,
                    period: str = '2y',
                    features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        准备训练数据
        
        Args:
            symbol: 股票代码
            period: 数据周期
            features: 特征列表
            
        Returns:
            pd.DataFrame: 准备好的数据
        """
        self.logger.info(f"准备数据: {symbol}, 周期: {period}")
        
        # 获取原始数据
        raw_data = self.data_manager.get_stock_data(symbol, period=period)
        
        # 特征工程
        if features is None:
            features = ['Volume', 'SMA_20', 'RSI_14', 'MACD', 'ATR']
        
        processed_data = self.feature_engineer.prepare_features(raw_data)
        
        # 保存数据信息
        self.experiment_config.update({
            'symbol': symbol,
            'period': period,
            'features': features,
            'data_shape': processed_data.shape,
            'data_start': str(processed_data.index[0]),
            'data_end': str(processed_data.index[-1])
        })
        
        # 保存数据
        data_path = self.experiment_path / "data.csv"
        processed_data.to_csv(data_path)
        
        self.logger.info(f"数据准备完成: {processed_data.shape}")
        return processed_data
    
    def run_optimization(self,
                        data: pd.DataFrame,
                        algorithm: str = 'ppo',
                        reward_type: str = 'risk_adjusted',
                        n_trials: int = 100,
                        total_timesteps: int = 50000,
                        val_ratio: float = 0.2) -> Dict[str, Any]:
        """
        运行超参数优化
        
        Args:
            data: 训练数据
            algorithm: 算法类型
            reward_type: 奖励函数类型
            n_trials: 优化试验次数
            total_timesteps: 每次试验的训练步数
            val_ratio: 验证集比例
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        self.logger.info(f"开始超参数优化: {algorithm} + {reward_type}")
        
        # 创建优化器
        self.optimizer = HyperparameterOptimizer(
            config=self.config,
            n_trials=n_trials,
            n_jobs=1  # 单线程以避免资源冲突
        )
        
        # 运行优化
        opt_result = self.optimizer.optimize(
            df=data,
            algorithm=algorithm,
            reward_type=reward_type,
            total_timesteps=total_timesteps,
            val_ratio=val_ratio,
            study_name=f"{self.experiment_name}_optimization"
        )
        
        # 保存优化结果
        self.results['optimization'] = opt_result
        
        # 保存优化研究
        study_path = self.experiment_path / "optimization_study.pkl"
        self.optimizer.save_study(str(study_path))
        
        # 保存最佳参数
        params_path = self.experiment_path / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(opt_result['best_params'], f, indent=2)
        
        # 参数重要性分析
        importance = self.optimizer.get_feature_importance()
        if importance:
            importance_path = self.experiment_path / "param_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(importance, f, indent=2)
        
        self.experiment_config.update({
            'optimization': {
                'algorithm': algorithm,
                'reward_type': reward_type,
                'n_trials': n_trials,
                'total_timesteps': total_timesteps,
                'val_ratio': val_ratio,
                'best_score': opt_result['best_score']
            }
        })
        
        self.logger.info(f"超参数优化完成，最佳分数: {opt_result['best_score']:.4f}")
        return opt_result
    
    def train_final_model(self,
                         data: pd.DataFrame,
                         total_timesteps: int = 200000,
                         use_best_params: bool = True,
                         save_checkpoints: bool = True,
                         checkpoint_freq: int = 10000,
                         reward_type: str = 'risk_adjusted',
                         reward_kwargs: Optional[Dict] = None,
                         model_kwargs: Optional[Dict] = None,
                         env_kwargs: Optional[Dict] = None,
                         feature_columns: Optional[List[str]] = None,
                         n_envs: int = 1) -> Dict[str, Any]:
        """
        训练最终模型
        
        Args:
            data: 训练数据
            total_timesteps: 训练步数
            use_best_params: 是否使用优化后的参数
            save_checkpoints: 是否保存检查点
            checkpoint_freq: 检查点保存频率（步数）
            reward_type: 奖励函数类型
            reward_kwargs: 奖励函数参数
            model_kwargs: 模型超参数 (learning_rate, clip_range, n_steps等)
            env_kwargs: 环境参数 (transaction_costs等)
            feature_columns: 要使用的特征列列表，如果为None则使用所有特征
            n_envs: 并行环境数量，>1时启用多进程并行训练
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        self.logger.info(f"开始训练最终模型: {total_timesteps} steps")
        
        if use_best_params and self.optimizer is not None:
            # 使用优化后的参数创建训练器
            self.trainer = self.optimizer.create_optimized_trainer(data)
        else:
            # 使用指定的参数创建训练器
            reward_kwargs = reward_kwargs or {'initial_balance': 10000}
            reward_fn = create_reward_function(reward_type, **reward_kwargs)
            self.trainer = StableBaselinesTrainer(self.config)
            
            # 传递模型参数和环境参数
            self.trainer.setup_training(
                data, 
                reward_function=reward_fn,
                feature_columns=feature_columns,
                model_kwargs=model_kwargs,
                env_kwargs=env_kwargs,
                n_envs=n_envs
            )
            
            self.logger.info(f"使用奖励函数: {reward_type}, 参数: {reward_kwargs}")
            if model_kwargs:
                self.logger.info(f"模型参数: {model_kwargs}")
            if env_kwargs:
                self.logger.info(f"环境参数: {env_kwargs}")
        
        # 设置保存路径
        model_path = self.experiment_path / "final_model"
        
        # 训练模型
        training_result = self.trainer.train(
            total_timesteps=total_timesteps,
            eval_freq=10000 if save_checkpoints else 0,
            save_path=str(model_path),
            checkpoint_freq=checkpoint_freq if save_checkpoints else 0,
            save_checkpoints=save_checkpoints
        )
        
        # 保存训练结果
        self.results['training'] = training_result
        self.models['final_model'] = str(model_path) + ".zip"
        
        self.experiment_config.update({
            'final_training': {
                'total_timesteps': total_timesteps,
                'use_best_params': use_best_params,
                'training_time': training_result['training_time']
            }
        })
        
        self.logger.info(f"模型训练完成，用时: {training_result['training_time']:.2f}s")
        return training_result
    
    def evaluate_model(self,
                      test_data: Optional[pd.DataFrame] = None,
                      n_episodes: int = 20,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        评估训练好的模型
        
        Args:
            test_data: 测试数据，如果为None则使用最后20%的训练数据
            n_episodes: 评估轮数
            save_results: 是否保存评估结果
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"开始模型评估: {n_episodes} episodes")
        
        if self.trainer is None:
            raise RuntimeError("请先训练模型")
        
        # 准备测试数据
        if test_data is None:
            # 使用训练数据的最后20%作为测试数据
            data_path = self.experiment_path / "data.csv"
            if data_path.exists():
                full_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
                split_point = int(len(full_data) * 0.8)
                test_data = full_data.iloc[split_point:].copy()
            else:
                raise RuntimeError("找不到测试数据")
        
        # 评估模型
        model_path = self.models.get('final_model')
        if not model_path:
            raise RuntimeError("找不到训练好的模型")
        
        eval_result = self.trainer.evaluate(
            model_path=model_path.replace('.zip', ''),
            test_df=test_data,
            n_episodes=n_episodes,
            render=False
        )
        
        # 保存评估结果
        self.results['evaluation'] = eval_result
        
        if save_results:
            eval_path = self.experiment_path / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_result, f, indent=2)
        
        self.experiment_config.update({
            'evaluation': {
                'n_episodes': n_episodes,
                'test_data_shape': test_data.shape,
                'mean_return': eval_result['mean_return'],
                'win_rate': eval_result['win_rate']
            }
        })
        
        self.logger.info(f"模型评估完成:")
        self.logger.info(f"  - 平均回报: {eval_result['mean_return']:.2f}%")
        self.logger.info(f"  - 胜率: {eval_result['win_rate']:.2%}")
        self.logger.info(f"  - 夏普比率: {eval_result['mean_return'] / max(eval_result['std_return'], 0.01):.4f}")
        
        return eval_result
    
    def run_complete_pipeline(self,
                             symbol: str,
                             period: str = '2y',
                             features: Optional[List[str]] = None,
                             algorithm: str = 'ppo',
                             reward_type: str = 'risk_adjusted',
                             optimization_trials: int = 50,
                             final_timesteps: int = 200000) -> Dict[str, Any]:
        """
        运行完整的训练流水线
        
        Args:
            symbol: 股票代码
            period: 数据周期
            features: 特征列表
            algorithm: RL算法
            reward_type: 奖励函数类型
            optimization_trials: 优化试验次数
            final_timesteps: 最终训练步数
            
        Returns:
            Dict[str, Any]: 完整的实验结果
        """
        self.logger.info(f"开始完整训练流水线: {symbol}")
        
        pipeline_start = datetime.now()
        
        try:
            # Step 1: 准备数据
            self.logger.info("=" * 50)
            self.logger.info("Step 1: 数据准备")
            data = self.prepare_data(symbol, period, features)
            
            # Step 2: 超参数优化
            self.logger.info("=" * 50)
            self.logger.info("Step 2: 超参数优化")
            opt_result = self.run_optimization(
                data, algorithm, reward_type, 
                optimization_trials, final_timesteps // 10
            )
            
            # Step 3: 训练最终模型
            self.logger.info("=" * 50)
            self.logger.info("Step 3: 最终模型训练")
            train_result = self.train_final_model(
                data, final_timesteps, use_best_params=True
            )
            
            # Step 4: 模型评估
            self.logger.info("=" * 50)
            self.logger.info("Step 4: 模型评估")
            eval_result = self.evaluate_model(n_episodes=30)
            
            # 完成流水线
            pipeline_time = datetime.now() - pipeline_start
            
            # 整合最终结果
            final_results = {
                'experiment_name': self.experiment_name,
                'experiment_path': str(self.experiment_path),
                'pipeline_time': pipeline_time.total_seconds(),
                'config': self.experiment_config,
                'optimization': opt_result,
                'training': train_result,
                'evaluation': eval_result,
                'models': self.models
            }
            
            # 保存实验配置和结果
            self.save_experiment()
            
            self.logger.info("=" * 50)
            self.logger.info(f"完整训练流水线完成! 用时: {pipeline_time}")
            self.logger.info(f"实验结果已保存到: {self.experiment_path}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"训练流水线失败: {e}")
            raise
    
    def save_experiment(self) -> None:
        """保存实验配置和结果"""
        # 保存实验配置
        config_path = self.experiment_path / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.experiment_config, f, indent=2, default=str)
        
        # 保存完整结果
        results_path = self.experiment_path / "complete_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # 创建实验报告
        self.create_experiment_report()
    
    def create_experiment_report(self) -> None:
        """创建实验报告"""
        report_path = self.experiment_path / "experiment_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 实验报告: {self.experiment_name}\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据信息
            if 'symbol' in self.experiment_config:
                f.write("## 数据信息\n")
                f.write(f"- 股票代码: {self.experiment_config['symbol']}\n")
                f.write(f"- 数据周期: {self.experiment_config['period']}\n")
                f.write(f"- 数据形状: {self.experiment_config['data_shape']}\n")
                f.write(f"- 特征列表: {', '.join(self.experiment_config['features'])}\n\n")
            
            # 优化结果
            if 'optimization' in self.experiment_config:
                opt_config = self.experiment_config['optimization']
                f.write("## 超参数优化\n")
                f.write(f"- 算法: {opt_config['algorithm']}\n")
                f.write(f"- 奖励函数: {opt_config['reward_type']}\n")
                f.write(f"- 试验次数: {opt_config['n_trials']}\n")
                f.write(f"- 最佳分数: {opt_config['best_score']:.4f}\n\n")
            
            # 训练结果
            if 'final_training' in self.experiment_config:
                train_config = self.experiment_config['final_training']
                f.write("## 模型训练\n")
                f.write(f"- 训练步数: {train_config['total_timesteps']:,}\n")
                f.write(f"- 训练用时: {train_config['training_time']:.2f}秒\n")
                f.write(f"- 使用优化参数: {'是' if train_config['use_best_params'] else '否'}\n\n")
            
            # 评估结果
            if 'evaluation' in self.experiment_config:
                eval_config = self.experiment_config['evaluation']
                f.write("## 模型评估\n")
                f.write(f"- 评估轮数: {eval_config['n_episodes']}\n")
                f.write(f"- 平均回报: {eval_config['mean_return']:.2f}%\n")
                f.write(f"- 胜率: {eval_config['win_rate']:.2%}\n\n")
            
            # 文件列表
            f.write("## 生成文件\n")
            for file_path in self.experiment_path.iterdir():
                if file_path.is_file():
                    f.write(f"- {file_path.name}\n")
    
    def load_experiment(self, experiment_path: str) -> None:
        """
        加载已有实验
        
        Args:
            experiment_path: 实验路径
        """
        exp_path = Path(experiment_path)
        if not exp_path.exists():
            raise FileNotFoundError(f"实验路径不存在: {experiment_path}")
        
        self.experiment_path = exp_path
        self.experiment_name = exp_path.name
        
        # 加载配置
        config_path = exp_path / "experiment_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.experiment_config = json.load(f)
        
        # 加载结果
        results_path = exp_path / "complete_results.pkl"
        if results_path.exists():
            with open(results_path, 'rb') as f:
                self.results = pickle.load(f)
        
        # 加载优化研究
        study_path = exp_path / "optimization_study.pkl"
        if study_path.exists():
            self.optimizer = HyperparameterOptimizer()
            self.optimizer.load_study(str(study_path))
        
        self.logger.info(f"实验已加载: {self.experiment_name}")


def run_quick_experiment(symbol: str,
                        algorithm: str = 'ppo',
                        reward_type: str = 'risk_adjusted',
                        optimization_trials: int = 20,
                        final_timesteps: int = 50000) -> Dict[str, Any]:
    """
    快速实验函数
    
    Args:
        symbol: 股票代码
        algorithm: 算法类型
        reward_type: 奖励函数类型
        optimization_trials: 优化试验次数
        final_timesteps: 最终训练步数
        
    Returns:
        Dict[str, Any]: 实验结果
    """
    pipeline = TrainingPipeline()
    return pipeline.run_complete_pipeline(
        symbol=symbol,
        algorithm=algorithm,
        reward_type=reward_type,
        optimization_trials=optimization_trials,
        final_timesteps=final_timesteps
    )


if __name__ == "__main__":
    # 测试代码
    print("训练流水线测试")
    print("=" * 50)
    
    # 快速实验
    result = run_quick_experiment(
        symbol='AAPL',
        optimization_trials=5,
        final_timesteps=10000
    )
    
    print(f"实验完成:")
    print(f"  - 实验名称: {result['experiment_name']}")
    print(f"  - 最佳分数: {result['optimization']['best_score']:.4f}")
    print(f"  - 平均回报: {result['evaluation']['mean_return']:.2f}%")
    print(f"  - 胜率: {result['evaluation']['win_rate']:.2%}")