"""
Stable-Baselines3 Trading Agent Trainer
交易智能体训练器 - 基于Stable-Baselines3

替代Ray RLlib，提供更简单直接的训练接口
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ..environment.trading_environment import TradingEnvironment
from ..environment.rewards.simple_return import SimpleReturnReward
from ..utils.logger import setup_logger
from ..utils.config import Config


class TradingCallback(BaseCallback):
    """增强的自定义训练回调函数 - 提供详细的训练进度输出"""
    
    def __init__(self, verbose=0, detailed_output=True, enable_visualization=False, 
                 visualization_frequency=50000, experiment_name=""):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.actions_history = []
        self.detailed_output = detailed_output
        
        # 可视化配置
        self.enable_visualization = enable_visualization
        self.visualization_frequency = visualization_frequency  # 每多少步生成一次可视化
        self.experiment_name = experiment_name
        self.last_visualization_step = 0
        
        # 统计信息
        self.total_steps = 0
        self.current_episode = 0
        self.episode_start_time = None
        self.training_start_time = None
        
        # 设置日志记录器
        from ..utils.logger import setup_logger
        self._logger = setup_logger('TradingCallback')
        
        # 初始化可视化管理器（如果启用）
        self.viz_manager = None
        if self.enable_visualization:
            try:
                from ..visualization import VisualizationManager
                self.viz_manager = VisualizationManager(
                    output_dir=f"visualizations/training_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self._logger.info(f"可视化已启用 - 频率: 每{visualization_frequency}步")
            except Exception as e:
                self._logger.warning(f"可视化初始化失败: {e}")
                self.enable_visualization = False
        
    def _on_training_start(self) -> None:
        """训练开始时的回调"""
        self.training_start_time = datetime.now()
        if self.detailed_output:
            self._logger.info("=" * 80)
            self._logger.info("开始强化学习训练过程...")
            self._logger.info("=" * 80)
        
    def _on_rollout_start(self) -> None:
        """每次rollout开始时的回调"""
        if self.detailed_output:
            self._logger.info(f"开始新的数据收集阶段 (Rollout) - 步数: {self.total_steps:,}")
    
    def _on_step(self) -> bool:
        """每个训练步骤的回调"""
        self.total_steps += 1
        
        # 获取当前状态信息
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0])
        actions = self.locals.get('actions', [0])
        dones = self.locals.get('dones', [False])
        
        # 收集所有环境信息用于汇总统计
        env_balances = []
        env_rewards = []
        env_actions = []
        
        # 处理每个环境的信息
        for i, (info, reward, action, done) in enumerate(zip(infos, rewards, actions, dones)):
            if self.detailed_output:
                # 获取投资组合信息
                portfolio_info = info.get('portfolio', {})
                current_balance = portfolio_info.get('total_value', 10000.0)
                action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
                
                # 收集统计数据
                env_balances.append(current_balance)
                env_rewards.append(reward)
                env_actions.append(action_val)
                
                # 每1000步输出一次详细信息
                if self.total_steps % 1000 == 0:
                    initial_balance = 10000.0  # 假设初始资金
                    profit_pct = ((current_balance - initial_balance) / initial_balance) * 100
                    
                    # 添加环境ID标识
                    self._logger.info(f"    [ENV-{i+1}] 训练步数 {self.total_steps:6d} | 资金: ${current_balance:8,.0f} | "
                                   f"盈亏: {profit_pct:+6.2f}% | 奖励: {reward:+7.3f} | "
                                   f"动作: {action_val:+6.3f}")
                    
                    # 强制刷新日志缓冲区
                    if hasattr(self._logger, 'flush_all'):
                        self._logger.flush_all()
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()
        
        # 每1000步输出多环境汇总统计
        if self.detailed_output and self.total_steps % 1000 == 0 and len(env_balances) > 1:
            initial_balance = 10000.0
            avg_balance = sum(env_balances) / len(env_balances)
            avg_profit = ((avg_balance - initial_balance) / initial_balance) * 100
            avg_reward = sum(env_rewards) / len(env_rewards)
            best_env_idx = env_balances.index(max(env_balances))
            worst_env_idx = env_balances.index(min(env_balances))
            
            self._logger.info("    " + "="*80)
            self._logger.info(f"    [汇总] 平均资金: ${avg_balance:8,.0f} | 平均盈亏: {avg_profit:+6.2f}% | 平均奖励: {avg_reward:+7.3f}")
            self._logger.info(f"    [汇总] 最佳环境: ENV-{best_env_idx+1} (${env_balances[best_env_idx]:,.0f}) | "
                           f"最差环境: ENV-{worst_env_idx+1} (${env_balances[worst_env_idx]:,.0f})")
            self._logger.info("    " + "="*80)
            
            # 强制刷新日志缓冲区
            if hasattr(self._logger, 'flush_all'):
                self._logger.flush_all()
        
        # 继续处理每个环境的Episode结束事件
        for i, (info, reward, action, done) in enumerate(zip(infos, rewards, actions, dones)):
            if self.detailed_output:
                # 获取投资组合信息
                portfolio_info = info.get('portfolio', {})
                current_balance = portfolio_info.get('total_value', 10000.0)
                
                # Episode结束处理
                if done:
                    self.current_episode += 1
                    episode_reward = self.locals.get('episode_rewards', [0])[i] if 'episode_rewards' in self.locals else reward
                    episode_length = self.locals.get('episode_lengths', [0])[i] if 'episode_lengths' in self.locals else 1
                    
                    # 记录数据
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.portfolio_values.append(current_balance)
                    
                    # 计算统计信息
                    initial_balance = 10000.0
                    final_return = ((current_balance - initial_balance) / initial_balance) * 100
                    
                    if self.detailed_output:
                        self._logger.info("-" * 60)
                        self._logger.info(f"  [ENV-{i+1}] Episode {self.current_episode} 完成:")
                        self._logger.info(f"    步数: {episode_length:,}")
                        self._logger.info(f"    最终资金: ${current_balance:,.2f}")
                        self._logger.info(f"    收益率: {final_return:+.2f}%")
                        self._logger.info(f"    累计奖励: {episode_reward:+.3f}")
                        
                        # 计算最近100个episode的平均统计
                        if len(self.episode_rewards) >= 10:
                            recent_episodes = min(100, len(self.episode_rewards))
                            avg_reward = np.mean(self.episode_rewards[-recent_episodes:])
                            avg_length = np.mean(self.episode_lengths[-recent_episodes:])
                            avg_return = np.mean([((pv - initial_balance) / initial_balance) * 100 
                                               for pv in self.portfolio_values[-recent_episodes:]])
                            win_rate = len([r for r in self.portfolio_values[-recent_episodes:] if r > initial_balance]) / recent_episodes
                            
                            self._logger.info(f"    最近{recent_episodes}个Episode统计:")
                            self._logger.info(f"      平均奖励: {avg_reward:+.3f}")
                            self._logger.info(f"      平均步数: {avg_length:.0f}")
                            self._logger.info(f"      平均收益: {avg_return:+.2f}%")
                            self._logger.info(f"      胜率: {win_rate:.2%}")
                        self._logger.info("=" * 80)
        
        # 检查是否需要输出训练进度摘要
        if self.total_steps % 10000 == 0:
            self._log_training_summary()
        
        # 检查是否需要生成可视化
        if (self.enable_visualization and self.viz_manager and 
            self.total_steps - self.last_visualization_step >= self.visualization_frequency and
            len(self.episode_rewards) >= 10):  # 至少有10个episode数据
            self._generate_training_visualizations()
            self.last_visualization_step = self.total_steps
        
        return True
    
    def _log_training_summary(self):
        """输出训练进度摘要"""
        if not self.detailed_output or not self.episode_rewards:
            return
            
        elapsed_time = datetime.now() - self.training_start_time if self.training_start_time else 0
        
        self._logger.info("=" * 80)
        self._logger.info(f"训练进度摘要 - 步数: {self.total_steps:,}")
        self._logger.info(f"已完成Episodes: {len(self.episode_rewards)}")
        self._logger.info(f"训练用时: {elapsed_time}")
        
        if len(self.episode_rewards) > 0:
            # 计算整体统计
            total_avg_reward = np.mean(self.episode_rewards)
            total_avg_length = np.mean(self.episode_lengths)
            
            # 最近的性能
            recent_count = min(50, len(self.episode_rewards))
            recent_avg_reward = np.mean(self.episode_rewards[-recent_count:])
            recent_avg_length = np.mean(self.episode_lengths[-recent_count:])
            
            # 收益统计
            if self.portfolio_values:
                initial_balance = 10000.0
                recent_returns = [((pv - initial_balance) / initial_balance) * 100 
                                for pv in self.portfolio_values[-recent_count:]]
                avg_return = np.mean(recent_returns)
                win_rate = len([r for r in recent_returns if r > 0]) / len(recent_returns)
                
                self._logger.info(f"整体平均奖励: {total_avg_reward:+.3f} | 最近{recent_count}个: {recent_avg_reward:+.3f}")
                self._logger.info(f"整体平均步数: {total_avg_length:.0f} | 最近{recent_count}个: {recent_avg_length:.0f}")
                self._logger.info(f"最近{recent_count}个Episode平均收益: {avg_return:+.2f}%")
                self._logger.info(f"最近{recent_count}个Episode胜率: {win_rate:.2%}")
        
        self._logger.info("=" * 80)
    
    def _on_training_end(self) -> None:
        """训练结束时的回调"""
        if self.detailed_output:
            elapsed_time = datetime.now() - self.training_start_time if self.training_start_time else 0
            self._logger.info("=" * 80)
            self._logger.info("强化学习训练完成！")
            self._logger.info(f"总步数: {self.total_steps:,}")
            self._logger.info(f"总Episodes: {len(self.episode_rewards)}")
            self._logger.info(f"总用时: {elapsed_time}")
            
            if self.episode_rewards:
                avg_reward = np.mean(self.episode_rewards)
                best_reward = max(self.episode_rewards)
                worst_reward = min(self.episode_rewards)
                
                self._logger.info(f"平均奖励: {avg_reward:+.3f}")
                self._logger.info(f"最佳奖励: {best_reward:+.3f}")
                self._logger.info(f"最差奖励: {worst_reward:+.3f}")
                
                if self.portfolio_values:
                    initial_balance = 10000.0
                    final_returns = [((pv - initial_balance) / initial_balance) * 100 
                                   for pv in self.portfolio_values]
                    avg_return = np.mean(final_returns)
                    best_return = max(final_returns)
                    win_rate = len([r for r in final_returns if r > 0]) / len(final_returns)
                    
                    self._logger.info(f"平均收益率: {avg_return:+.2f}%")
                    self._logger.info(f"最佳收益率: {best_return:+.2f}%")
                    self._logger.info(f"整体胜率: {win_rate:.2%}")
            
            self._logger.info("=" * 80)
        
        # 生成最终可视化报告（如果启用）
        if self.enable_visualization and self.viz_manager:
            self._generate_final_training_report()
    
    def _generate_training_visualizations(self):
        """生成训练过程中的可视化"""
        try:
            training_data = {
                'episode_rewards': self.episode_rewards.copy(),
                'portfolio_values': self.portfolio_values.copy(),
                'actions_history': self.actions_history.copy(),
                'initial_balance': 10000.0
            }
            
            self._logger.info(f"生成训练可视化 - 步数: {self.total_steps}, Episodes: {len(self.episode_rewards)}")
            generated_files = self.viz_manager.generate_training_visualizations(
                training_data=training_data,
                experiment_name=f"{self.experiment_name}_step_{self.total_steps}",
                detailed=False  # 训练过程中使用简化版本
            )
            
            if generated_files:
                self._logger.info(f"已生成 {len(generated_files)} 类训练可视化图表")
                
        except Exception as e:
            self._logger.error(f"生成训练可视化时出错: {e}")
    
    def _generate_final_training_report(self):
        """生成最终训练报告"""
        try:
            training_data = {
                'episode_rewards': self.episode_rewards,
                'portfolio_values': self.portfolio_values,
                'actions_history': self.actions_history,
                'initial_balance': 10000.0
            }
            
            report_data = {
                'training_data': training_data,
                'experiment_name': self.experiment_name,
                'model_info': {
                    'total_steps': self.total_steps,
                    'total_episodes': len(self.episode_rewards),
                    'training_duration': str(datetime.now() - self.training_start_time) if self.training_start_time else "未知",
                    'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                    'final_portfolio_value': self.portfolio_values[-1] if self.portfolio_values else 10000
                }
            }
            
            self._logger.info("生成最终训练报告...")
            generated_reports = self.viz_manager.create_comprehensive_report(
                report_data=report_data,
                report_name=f"Training_Report_{self.experiment_name}",
                include_html=True
            )
            
            if generated_reports:
                self._logger.info(f"训练报告已生成: {generated_reports}")
            
        except Exception as e:
            self._logger.error(f"生成最终训练报告时出错: {e}")


class StableBaselinesTrainer:
    """
    交易智能体训练器
    
    使用Stable-Baselines3替代Ray RLlib，提供更简洁的训练接口
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化训练器
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.logger = setup_logger('StableBaselinesTrainer')
        
        # 训练参数
        self.model = None
        self.env = None
        self.eval_env = None
        
        # 存储训练配置以供评估使用
        self.training_config = {}
        
        # 支持的算法
        self.algorithms = {
            'ppo': PPO,
            'dqn': DQN,
            'sac': SAC
        }
        
    def create_environment(self, df: pd.DataFrame, reward_function=None, feature_columns=None, **env_kwargs) -> gym.Env:
        """
        创建交易环境
        
        Args:
            df: 交易数据
            reward_function: 奖励函数
            feature_columns: 要使用的特征列列表，如果为None则自动选择
            **env_kwargs: 环境参数
            
        Returns:
            gym.Env: 配置好的环境
        """
        # 默认奖励函数
        if reward_function is None:
            reward_function = SimpleReturnReward(initial_balance=10000)
        
        # 特征列选择逻辑
        available_columns = df.columns.tolist()
        
        if feature_columns is None:
            # 自动选择特征：检查是否为高维特征数据集（如117特征）
            non_ohlc_columns = [col for col in available_columns if col not in ['Open', 'High', 'Low', 'Close']]
            
            if len(non_ohlc_columns) >= 50:  # 高维特征数据集
                # 使用所有可用特征以发挥高维数据集的优势
                feature_columns = non_ohlc_columns
                self.logger.info("检测到高维特征数据集，将使用所有可用特征")
            else:
                # 标准数据集：优先选择常用特征
                preferred_features = ['SMA_20', 'RSI_14', 'MACD', 'ATR_14', 'Volume', 'EMA_12', 'BB_Upper_20']
                feature_columns = []
                
                for feature in preferred_features:
                    if feature in available_columns:
                        feature_columns.append(feature)
                
                # 如果首选特征不足，则使用所有可用特征
                if len(feature_columns) == 0:
                    feature_columns = non_ohlc_columns
        else:
            # 验证传入的特征列是否存在
            valid_features = [col for col in feature_columns if col in available_columns]
            if len(valid_features) != len(feature_columns):
                missing = [col for col in feature_columns if col not in available_columns]
                self.logger.warning(f"以下特征列在数据中不存在，将被忽略: {missing}")
            feature_columns = valid_features
            
            if len(feature_columns) == 0:
                raise ValueError("没有有效的特征列可用")
        
        self.logger.info(f"数据列总数: {len(available_columns)}")
        self.logger.info(f"选择的特征列: {len(feature_columns)}个 - {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
        
        default_kwargs = {
            'initial_balance': 10000,
            'window_size': 50,
            'feature_columns': feature_columns,  # 使用所有可用特征
            'transaction_costs': 0.001,
            'max_episode_steps': len(df) // 4  # 1/4数据长度作为一个episode
        }
        default_kwargs.update(env_kwargs)
        
        # 创建环境
        env = TradingEnvironment(
            df=df,
            config=self.config,
            reward_function=reward_function,
            **default_kwargs
        )
        
        # 环境检查
        try:
            check_env(env)
            self.logger.info("环境验证通过")
        except Exception as e:
            self.logger.warning(f"环境验证警告: {e}")
        
        return env
    
    def setup_training(self, 
                      df: pd.DataFrame,
                      algorithm: str = 'ppo',
                      reward_function=None,
                      feature_columns=None,
                      model_kwargs: Optional[Dict] = None,
                      env_kwargs: Optional[Dict] = None,
                      n_envs: int = 1) -> None:
        """
        设置训练环境和模型
        
        Args:
            df: 训练数据
            algorithm: 算法类型 ('ppo', 'dqn', 'sac')
            reward_function: 奖励函数
            feature_columns: 要使用的特征列列表，如果为None则自动选择
            model_kwargs: 模型参数
            env_kwargs: 环境参数
            n_envs: 并行环境数量，>1时使用多进程加速
        """
        self.logger.info(f"设置训练 - 算法: {algorithm}")
        self.logger.info(f"并行环境数量: {n_envs}")
        
        # 创建环境
        env_kwargs = env_kwargs or {}
        
        # 存储训练配置
        self.training_config = {
            'reward_function': reward_function,
            'env_kwargs': env_kwargs,
            'algorithm': algorithm,
            'df': df,
            'feature_columns': feature_columns
        }
        
        # 创建环境工厂函数
        def make_env():
            env = self.create_environment(df, reward_function, feature_columns, **env_kwargs)
            env = Monitor(env)
            return env
        
        # 根据环境数量选择向量化方式
        if n_envs == 1:
            # 单环境使用DummyVecEnv
            self.env = DummyVecEnv([make_env])
            self.logger.info("使用单环境模式 (DummyVecEnv)")
        else:
            # 多环境使用SubprocVecEnv实现真正并行
            self.env = SubprocVecEnv([make_env for _ in range(n_envs)])
            self.logger.info(f"使用多进程并行模式 (SubprocVecEnv) - {n_envs}个环境")
        
        # 默认模型参数
        default_model_kwargs = {
            'policy': 'MlpPolicy',
            'verbose': 1,
            'tensorboard_log': './logs/tensorboard/',
        }
        
        # 算法特定的默认参数
        if algorithm == 'ppo':
            default_model_kwargs.update({
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01
            })
        elif algorithm == 'sac':
            default_model_kwargs.update({
                'learning_rate': 3e-4,
                'buffer_size': 100000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'ent_coef': 'auto'
            })
        
        # 更新用户参数，过滤掉不支持的参数
        if model_kwargs:
            # 定义各算法支持的参数
            supported_params = {
                'ppo': {'policy', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 
                       'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 
                       'max_grad_norm', 'verbose', 'tensorboard_log'},
                'sac': {'policy', 'learning_rate', 'buffer_size', 'batch_size', 'tau',
                       'gamma', 'ent_coef', 'verbose', 'tensorboard_log'},
                'dqn': {'policy', 'learning_rate', 'buffer_size', 'batch_size', 'gamma',
                       'exploration_fraction', 'exploration_initial_eps', 'exploration_final_eps',
                       'verbose', 'tensorboard_log'}
            }
            
            # 过滤参数
            allowed_params = supported_params.get(algorithm, set())
            filtered_params = {k: v for k, v in model_kwargs.items() if k in allowed_params}
            default_model_kwargs.update(filtered_params)
            
            # 记录被过滤的参数
            filtered_out = set(model_kwargs.keys()) - allowed_params
            if filtered_out:
                self.logger.warning(f"过滤掉不支持的参数: {filtered_out}")
        
        # 创建模型
        algorithm_class = self.algorithms.get(algorithm)
        if algorithm_class is None:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        self.model = algorithm_class(env=self.env, **default_model_kwargs)
        self.logger.info(f"模型创建完成: {algorithm_class.__name__}")
    
    def train(self, 
              total_timesteps: int = 100000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 5,
              save_path: str = "models/trading_model",
              enable_visualization: bool = False,
              visualization_frequency: int = 50000,
              experiment_name: str = "",
              checkpoint_freq: int = 0,
              save_checkpoints: bool = False) -> Dict[str, Any]:
        """
        开始训练
        
        Args:
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            n_eval_episodes: 评估episode数
            save_path: 模型保存路径
            enable_visualization: 是否启用可视化
            visualization_frequency: 可视化生成频率
            experiment_name: 实验名称
            checkpoint_freq: 检查点保存频率（步数），0表示不保存检查点
            save_checkpoints: 是否启用检查点保存
            
        Returns:
            Dict: 训练结果
        """
        if self.model is None:
            raise RuntimeError("请先调用setup_training()设置训练")
        
        self.logger.info(f"开始训练 - 总步数: {total_timesteps}")
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 设置回调 - 启用详细输出和可视化
        callbacks = [TradingCallback(
            detailed_output=True, 
            enable_visualization=enable_visualization,
            visualization_frequency=visualization_frequency, 
            experiment_name=experiment_name
        )]
        
        # 检查点回调（如果启用）
        if save_checkpoints and checkpoint_freq > 0:
            # 创建检查点目录
            checkpoint_dir = os.path.dirname(save_path) + "/checkpoints/"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_dir,
                name_prefix="checkpoint",
                verbose=1
            )
            callbacks.append(checkpoint_callback)
            self.logger.info(f"启用检查点保存 - 频率: 每{checkpoint_freq}步, 路径: {checkpoint_dir}")
        
        # 评估回调（如果需要）
        if eval_freq > 0 and self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=save_path + "_best",
                log_path="./logs/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # 开始训练
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        training_time = datetime.now() - start_time
        
        # 保存模型
        self.model.save(save_path)
        self.logger.info(f"模型已保存到: {save_path}")
        
        # 返回训练结果
        result = {
            "training_time": training_time.total_seconds(),
            "total_timesteps": total_timesteps,
            "model_path": save_path,
            "algorithm": self.model.__class__.__name__
        }
        
        self.logger.info(f"训练完成 - 用时: {training_time}")
        return result
    
    def evaluate(self, 
                model_path: str,
                test_df: pd.DataFrame,
                n_episodes: int = 10,
                render: bool = False) -> Dict[str, Any]:
        """
        评估训练好的模型
        
        Args:
            model_path: 模型路径
            test_df: 测试数据
            n_episodes: 评估episode数
            render: 是否渲染
            
        Returns:
            Dict: 评估结果
        """
        self.logger.info(f"评估模型: {model_path}")
        
        # 加载模型 - 自动检测类型
        try:
            model = PPO.load(model_path)
            self.logger.info(f"成功加载PPO模型")
        except:
            try:
                model = SAC.load(model_path)
                self.logger.info(f"成功加载SAC模型")
            except:
                try:
                    model = DQN.load(model_path)
                    self.logger.info(f"成功加载DQN模型")
                except Exception as e:
                    self.logger.error(f"无法加载模型，尝试了PPO、SAC、DQN都失败: {e}")
                    raise
        
        # 创建评估环境 - 使用与训练时相同的参数
        if hasattr(self, 'training_config') and self.training_config:
            # 使用存储的训练配置
            reward_function = self.training_config.get('reward_function')
            env_kwargs = self.training_config.get('env_kwargs', {})
            
            eval_env = self.create_environment(
                test_df,
                reward_function=reward_function,
                **env_kwargs
            )
        else:
            eval_env = self.create_environment(test_df)
        
        # 运行评估
        self.logger.info("=" * 80)
        self.logger.info(f"开始模型评估 - 共 {n_episodes} 个回合")
        self.logger.info("=" * 80)
        
        episode_rewards = []
        episode_lengths = []
        portfolio_returns = []
        
        for episode in range(n_episodes):
            self.logger.info(f"\n回合 {episode + 1}/{n_episodes} 开始评估...")
            
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # 获取初始状态
            initial_portfolio = info.get('portfolio', {})
            initial_balance = initial_portfolio.get('total_value', 10000.0)
            self.logger.info(f"  初始资金: ${initial_balance:,.2f}")
            
            step_count = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                step_count += 1
                
                # 每1000步输出一次进度
                if step_count % 1000 == 0:
                    portfolio_info = info.get('portfolio', {})
                    current_value = portfolio_info.get('total_value', initial_balance)
                    profit_pct = ((current_value - initial_balance) / initial_balance) * 100
                    
                    self.logger.info(f"    步骤 {step_count:5d} | 资金: ${current_value:8,.0f} | "
                                   f"盈亏: {profit_pct:+6.2f}% | 奖励: {reward:+7.3f} | "
                                   f"动作: {float(action[0]) if hasattr(action, '__len__') else float(action):+6.3f}")
                
                if render:
                    eval_env.render()
            
            # Episode 结束统计
            portfolio_info = info.get('portfolio', {})
            final_value = portfolio_info.get('total_value', initial_balance)
            final_return = ((final_value - initial_balance) / initial_balance) * 100
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_returns.append(final_return)
            
            # 输出episode总结
            self.logger.info("-" * 50)
            self.logger.info(f"  回合 {episode + 1} 完成:")
            self.logger.info(f"    步数: {episode_length:,}")
            self.logger.info(f"    初始资金: ${initial_balance:,.2f}")
            self.logger.info(f"    最终资金: ${final_value:,.2f}")
            self.logger.info(f"    收益率: {final_return:+.2f}%")
            self.logger.info(f"    累计奖励: {episode_reward:+.3f}")
            self.logger.info("=" * 80)
        
        # 计算统计数据
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_return": np.mean(portfolio_returns),
            "std_return": np.std(portfolio_returns),
            "max_return": np.max(portfolio_returns),
            "min_return": np.min(portfolio_returns),
            "win_rate": len([r for r in portfolio_returns if r > 0]) / len(portfolio_returns)
        }
        
        self.logger.info(f"评估完成 - 平均奖励: {results['mean_reward']:.4f}, "
                        f"平均回报: {results['mean_return']:.2f}%")
        
        return results
    
    def load_checkpoint(self, checkpoint_path: str, algorithm: str = None) -> None:
        """
        从检查点加载模型
        
        Args:
            checkpoint_path: 检查点文件路径
            algorithm: 算法类型，如果为None则自动检测
        """
        self.logger.info(f"从检查点加载模型: {checkpoint_path}")
        
        if algorithm is None:
            # 尝试从文件名推断算法类型
            if 'ppo' in checkpoint_path.lower():
                algorithm = 'ppo'
            elif 'sac' in checkpoint_path.lower():
                algorithm = 'sac'
            elif 'dqn' in checkpoint_path.lower():
                algorithm = 'dqn'
            else:
                algorithm = 'ppo'  # 默认
        
        # 加载模型
        algorithm_class = self.algorithms.get(algorithm)
        if algorithm_class is None:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        try:
            self.model = algorithm_class.load(checkpoint_path, env=self.env)
            self.logger.info(f"成功从检查点加载{algorithm.upper()}模型")
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            raise
    
    def list_checkpoints(self, checkpoint_dir: str) -> List[str]:
        """
        列出检查点目录中的所有检查点文件
        
        Args:
            checkpoint_dir: 检查点目录路径
            
        Returns:
            List[str]: 检查点文件列表
        """
        if not os.path.exists(checkpoint_dir):
            self.logger.warning(f"检查点目录不存在: {checkpoint_dir}")
            return []
        
        checkpoints = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith('checkpoint') and file.endswith('.zip'):
                checkpoints.append(os.path.join(checkpoint_dir, file))
        
        # 按文件名排序（通常包含步数信息）
        checkpoints.sort()
        
        self.logger.info(f"找到 {len(checkpoints)} 个检查点文件")
        return checkpoints


# 便捷函数
def quick_train(df: pd.DataFrame, 
               algorithm: str = 'ppo',
               total_timesteps: int = 50000,
               save_path: str = "models/quick_model") -> Dict[str, Any]:
    """
    快速训练函数
    
    Args:
        df: 训练数据
        algorithm: 算法类型
        total_timesteps: 训练步数
        save_path: 保存路径
        
    Returns:
        Dict: 训练结果
    """
    trainer = StableBaselinesTrainer()
    trainer.setup_training(df, algorithm=algorithm)
    return trainer.train(total_timesteps=total_timesteps, save_path=save_path)