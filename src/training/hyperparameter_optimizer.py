"""
超参数优化器 - 基于Optuna的智能调参系统

使用Optuna自动搜索最佳超参数配置，支持多种RL算法和奖励函数的组合优化。
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .stable_baselines_trainer import StableBaselinesTrainer
from ..environment.rewards import create_reward_function
from ..utils.logger import setup_logger
from ..utils.config import Config


class HyperparameterOptimizer:
    """
    超参数优化器
    
    使用Optuna进行贝叶斯优化，自动搜索最佳超参数配置
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 n_jobs: int = 1):
        """
        初始化超参数优化器
        
        Args:
            config: 配置对象
            n_trials: 优化试验次数
            timeout: 优化超时时间（秒）
            n_jobs: 并行任务数
        """
        self.config = config or Config()
        self.logger = setup_logger('HyperparameterOptimizer')
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        
        # 优化结果
        self.study = None
        self.best_params = None
        self.best_score = None
        
        # 数据分割
        self.train_data = None
        self.val_data = None
        
    def split_data(self, df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分割训练和验证数据
        
        Args:
            df: 完整数据
            val_ratio: 验证集比例
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练数据, 验证数据)
        """
        split_point = int(len(df) * (1 - val_ratio))
        train_data = df.iloc[:split_point].copy()
        val_data = df.iloc[split_point:].copy()
        
        self.logger.info(f"数据分割完成 - 训练集: {len(train_data)}, 验证集: {len(val_data)}")
        
        return train_data, val_data
    
    def _suggest_ppo_params(self, trial: Trial) -> Dict[str, Any]:
        """建议PPO算法参数"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 2.0),
        }
    
    def _suggest_sac_params(self, trial: Trial) -> Dict[str, Any]:
        """建议SAC算法参数"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'tau': trial.suggest_float('tau', 0.001, 0.02),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
            'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4]),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
            'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.01, 0.1, 0.5]),
        }
    
    def _suggest_dqn_params(self, trial: Trial) -> Dict[str, Any]:
        """建议DQN算法参数"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
            'target_update_interval': trial.suggest_int('target_update_interval', 1000, 10000),
            'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8, 16]),
            'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4]),
            'exploration_fraction': trial.suggest_float('exploration_fraction', 0.05, 0.5),
            'exploration_initial_eps': trial.suggest_float('exploration_initial_eps', 0.5, 1.0),
            'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.2),
        }
    
    def _suggest_reward_params(self, trial: Trial, reward_type: str) -> Dict[str, Any]:
        """建议奖励函数参数"""
        base_params = {
            'initial_balance': trial.suggest_float('reward_initial_balance', 5000, 20000)
        }
        
        if reward_type == 'risk_adjusted':
            base_params.update({
                'reward_risk_free_rate': trial.suggest_float('risk_free_rate', 0.01, 0.05),
                'reward_window_size': trial.suggest_int('risk_window_size', 20, 100),
            })
        elif reward_type == 'simple_return':
            base_params.update({
                'step_weight': trial.suggest_float('step_weight', 0.5, 2.0),
                'total_weight': trial.suggest_float('total_weight', 0.1, 1.0),
            })
        elif reward_type == 'profit_loss':
            base_params.update({
                'profit_bonus': trial.suggest_float('profit_bonus', 1.5, 3.0),
                'loss_penalty': trial.suggest_float('loss_penalty', 1.0, 2.5),
                'consecutive_loss_penalty': trial.suggest_float('consecutive_loss_penalty', 0.1, 1.0),
            })
        
        return base_params
    
    def _suggest_env_params(self, trial: Trial) -> Dict[str, Any]:
        """建议环境参数"""
        return {
            'window_size': trial.suggest_int('env_window_size', 20, 100),
            'transaction_costs': trial.suggest_float('transaction_costs', 0.0001, 0.01, log=True),
            'initial_balance': trial.suggest_float('env_initial_balance', 5000, 20000),
        }
    
    def objective(self, trial: Trial, 
                  algorithm: str, 
                  reward_type: str,
                  total_timesteps: int = 20000) -> float:
        """
        优化目标函数
        
        Args:
            trial: Optuna trial对象
            algorithm: 算法类型
            reward_type: 奖励函数类型
            total_timesteps: 训练步数
            
        Returns:
            float: 优化目标值（越高越好）
        """
        try:
            # 建议模型参数
            if algorithm == 'ppo':
                model_params = self._suggest_ppo_params(trial)
            elif algorithm == 'sac':
                model_params = self._suggest_sac_params(trial)
            elif algorithm == 'dqn':
                model_params = self._suggest_dqn_params(trial)
            else:
                raise ValueError(f"不支持的算法: {algorithm}")
            
            # 建议奖励函数参数
            reward_params = self._suggest_reward_params(trial, reward_type)
            
            # 建议环境参数
            env_params = self._suggest_env_params(trial)
            
            # 创建奖励函数
            reward_function = create_reward_function(reward_type, **reward_params)
            
            # 创建训练器
            trainer = StableBaselinesTrainer(self.config)
            trainer.setup_training(
                df=self.train_data,
                algorithm=algorithm,
                reward_function=reward_function,
                model_kwargs=model_params,
                env_kwargs=env_params
            )
            
            # 训练模型
            training_result = trainer.train(
                total_timesteps=total_timesteps,
                eval_freq=0,  # 禁用评估以加速
                save_path=f"models/optuna_trial_{trial.number}"
            )
            
            # 在验证集上评估
            eval_result = trainer.evaluate(
                model_path=f"models/optuna_trial_{trial.number}",
                test_df=self.val_data,
                n_episodes=5  # 少量评估以加速
            )
            
            # 清理临时模型文件
            try:
                os.remove(f"models/optuna_trial_{trial.number}.zip")
            except:
                pass
            
            # 返回平均回报作为优化目标
            objective_value = eval_result['mean_return']
            
            self.logger.info(f"Trial {trial.number}: 平均回报 = {objective_value:.4f}")
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} 失败: {e}")
            return -np.inf  # 返回最差分数
    
    def optimize(self, 
                 df: pd.DataFrame,
                 algorithm: str = 'ppo',
                 reward_type: str = 'risk_adjusted',
                 total_timesteps: int = 20000,
                 val_ratio: float = 0.2,
                 study_name: Optional[str] = None) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            df: 训练数据
            algorithm: RL算法类型
            reward_type: 奖励函数类型
            total_timesteps: 每次试验的训练步数
            val_ratio: 验证集比例
            study_name: 研究名称
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        self.logger.info(f"开始超参数优化 - 算法: {algorithm}, 奖励: {reward_type}")
        
        # 分割数据
        self.train_data, self.val_data = self.split_data(df, val_ratio)
        
        # 创建研究
        study_name = study_name or f"{algorithm}_{reward_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # 创建目标函数
        def objective_wrapper(trial):
            return self.objective(trial, algorithm, reward_type, total_timesteps)
        
        # 开始优化
        start_time = datetime.now()
        self.study.optimize(
            objective_wrapper, 
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs
        )
        optimization_time = datetime.now() - start_time
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # 整理结果
        result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'optimization_time': optimization_time.total_seconds(),
            'study': self.study,
            'algorithm': algorithm,
            'reward_type': reward_type
        }
        
        self.logger.info(f"超参数优化完成:")
        self.logger.info(f"  - 最佳分数: {self.best_score:.4f}")
        self.logger.info(f"  - 试验次数: {len(self.study.trials)}")
        self.logger.info(f"  - 优化用时: {optimization_time}")
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取参数重要性排名"""
        if self.study is None:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            self.logger.warning(f"无法计算参数重要性: {e}")
            return {}
    
    def save_study(self, file_path: str) -> None:
        """保存优化研究结果"""
        if self.study is None:
            self.logger.warning("没有可保存的研究结果")
            return
        
        try:
            # 保存为pickle文件
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self.study, f)
            
            self.logger.info(f"研究结果已保存到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存研究结果失败: {e}")
    
    def load_study(self, file_path: str) -> None:
        """加载优化研究结果"""
        try:
            import pickle
            with open(file_path, 'rb') as f:
                self.study = pickle.load(f)
            
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            self.logger.info(f"研究结果已加载: {file_path}")
            
        except Exception as e:
            self.logger.error(f"加载研究结果失败: {e}")
    
    def create_optimized_trainer(self, df: pd.DataFrame) -> StableBaselinesTrainer:
        """
        基于最佳参数创建优化后的训练器
        
        Args:
            df: 训练数据
            
        Returns:
            StableBaselinesTrainer: 配置好的训练器
        """
        if self.best_params is None:
            raise RuntimeError("请先执行optimize()获取最佳参数")
        
        # 分离参数
        model_params = {}
        reward_params = {}
        env_params = {}
        
        for key, value in self.best_params.items():
            if key.startswith('reward_') or key in ['risk_free_rate', 'step_weight', 'total_weight', 'profit_bonus', 'loss_penalty', 'consecutive_loss_penalty']:
                clean_key = key.replace('reward_', '') if key.startswith('reward_') else key
                reward_params[clean_key] = value
            elif key.startswith('env_') or key in ['window_size', 'transaction_costs']:
                clean_key = key.replace('env_', '') if key.startswith('env_') else key
                env_params[clean_key] = value
            else:
                model_params[key] = value
        
        # 获取算法和奖励类型
        algorithm = getattr(self, '_last_algorithm', 'ppo')
        reward_type = getattr(self, '_last_reward_type', 'risk_adjusted')
        
        # 创建奖励函数
        reward_function = create_reward_function(reward_type, **reward_params)
        
        # 创建训练器
        trainer = StableBaselinesTrainer(self.config)
        trainer.setup_training(
            df=df,
            algorithm=algorithm,
            reward_function=reward_function,
            model_kwargs=model_params,
            env_kwargs=env_params
        )
        
        return trainer


def quick_optimize(df: pd.DataFrame,
                  algorithm: str = 'ppo',
                  reward_type: str = 'risk_adjusted',
                  n_trials: int = 50,
                  total_timesteps: int = 20000) -> Dict[str, Any]:
    """
    快速超参数优化函数
    
    Args:
        df: 训练数据
        algorithm: 算法类型
        reward_type: 奖励函数类型
        n_trials: 试验次数
        total_timesteps: 训练步数
        
    Returns:
        Dict[str, Any]: 优化结果
    """
    optimizer = HyperparameterOptimizer(n_trials=n_trials)
    return optimizer.optimize(df, algorithm, reward_type, total_timesteps)


if __name__ == "__main__":
    # 测试代码
    print("超参数优化器测试")
    print("=" * 50)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=500)
    data = {
        'Close': np.random.randn(500).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 500),
        'SMA_20': np.random.randn(500).cumsum() + 100,
        'RSI_14': np.random.uniform(20, 80, 500)
    }
    df = pd.DataFrame(data, index=dates)
    
    print(f"测试数据创建完成: {df.shape}")
    
    # 快速优化测试
    print("\n开始快速优化测试...")
    result = quick_optimize(df, n_trials=10, total_timesteps=5000)
    
    print(f"\n优化结果:")
    print(f"  - 最佳分数: {result['best_score']:.4f}")
    print(f"  - 试验次数: {result['n_trials']}")
    print(f"  - 最佳参数: {result['best_params']}")
    
    print("\n超参数优化器测试完成")