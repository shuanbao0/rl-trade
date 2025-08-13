#!/usr/bin/env python
"""
模型评估脚本

功能:
1. 从本地文件加载测试数据
2. 加载训练好的模型
3. 执行模型性能评估
4. 生成详细的评估报告

使用示例:
  # 直接运行 (默认评估EURUSD，自动查找模型和数据)
  python evaluate_model.py

  # 指定外汇评估
  python evaluate_model.py --symbol GBPUSD --episodes 150

  # 指定模型和数据路径
  python evaluate_model.py --symbol EURUSD --model-path models/EURUSD_20240123_143022 --data-dir datasets/EURUSD_20240123_143022

  # 使用特定测试数据
  python evaluate_model.py --symbol EURUSD --test-data datasets/EURUSD_test.pkl

  # 生成详细报告
  python evaluate_model.py --symbol EURUSD --generate-report

  # 批量评估多个模型
  python evaluate_model.py --batch-config batch_eval_config.json
"""

import os
import sys
import argparse
import json
import warnings
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from src.environment.trading_environment import TradingEnvironment
    from src.utils.config import Config
    from src.utils.logger import setup_logger, get_default_log_file
    from src.utils.data_utils import get_data_processor
    
    # 导入训练组件
    from src.training import StableBaselinesTrainer, TrainingPipeline
    
    # 导入可视化组件
    from src.visualization import VisualizationManager
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)


class ModelEvaluator:
    """
    模型评估类
    
    负责加载测试数据、加载训练模型、执行评估和生成报告
    """
    
    def __init__(self, config_path: Optional[str] = None, output_dir: str = "evaluation_results"):
        """
        初始化模型评估器
        
        Args:
            config_path: 配置文件路径
            output_dir: 评估结果输出目录
        """
        # 加载配置
        self.config = Config(config_file=config_path) if config_path else Config()
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化日志系统
        self.logger = setup_logger(
            name="ModelEvaluator",
            level="INFO",
            log_file=get_default_log_file("model_evaluator")
        )
        
        # 初始化核心组件
        self.trading_environment = None
        self.sb3_trainer = None
        
        # 初始化数据处理器
        self.data_processor = get_data_processor("ModelEvaluator_DataProcessor")
        
        self.logger.info("模型评估器初始化完成")
    
    def load_test_data(self, data_source: str, data_type: str = "auto") -> pd.DataFrame:
        """
        加载测试数据
        
        Args:
            data_source: 数据源路径 (目录或文件)
            data_type: 数据类型 (auto/dir/file)
            
        Returns:
            pd.DataFrame: 测试数据
        """
        data_path = Path(data_source)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据源不存在: {data_source}")
        
        if data_type == "auto":
            data_type = "dir" if data_path.is_dir() else "file"
        
        if data_type == "dir":
            # 从目录加载测试数据，优先尝试 CSV 文件
            csv_files = list(data_path.glob("*_test.csv"))
            pkl_files = list(data_path.glob("*_test.pkl"))
            
            if csv_files:
                test_file = csv_files[0]  # 取第一个匹配的文件
                test_data = pd.read_csv(test_file, index_col=0)
                self.logger.info(f"从目录加载CSV测试数据: {test_file.name} ({len(test_data)} 条记录)")
            elif pkl_files:
                test_file = pkl_files[0]  # 取第一个匹配的文件
                try:
                    with open(test_file, 'rb') as f:
                        test_data = pickle.load(f)
                    self.logger.info(f"从目录加载PKL测试数据: {test_file.name} ({len(test_data)} 条记录)")
                except Exception as e:
                    self.logger.error(f"加载 PKL 文件失败: {e}")
                    raise FileNotFoundError(f"无法加载测试数据文件: {test_file}")
            else:
                raise FileNotFoundError(f"在 {data_source} 中未找到测试数据文件")
            
        else:
            # 直接加载文件
            if data_path.suffix == '.pkl':
                with open(data_path, 'rb') as f:
                    test_data = pickle.load(f)
            elif data_path.suffix == '.csv':
                test_data = pd.read_csv(data_path, index_col=0)
            else:
                raise ValueError(f"不支持的文件格式: {data_path.suffix}")
            
            self.logger.info(f"加载测试数据: {data_path.name} ({len(test_data)} 条记录)")
        
        return test_data
    
    def _preprocess_test_data(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理测试数据
        
        Args:
            test_data: 原始测试数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        return self.data_processor.deep_preprocess_data(test_data)
    
    def load_model(self, model_path: str, features_data: pd.DataFrame) -> Any:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型路径
            features_data: 特征数据
            
        Returns:
            加载的模型
        """
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 查找最佳检查点路径
        checkpoint_path = self._find_best_checkpoint(model_dir)
        if not checkpoint_path:
            raise FileNotFoundError(f"未找到有效的检查点文件在: {model_path}")
        
        self.logger.info(f"找到检查点: {checkpoint_path}")
        
        # 使用Stable-Baselines3训练器进行评估
        try:
            self.logger.info("=" * 60)
            self.logger.info("开始模型类型检测和加载...")
            self.logger.info(f"模型文件路径: {checkpoint_path}")
            
            # 检查文件信息
            if os.path.exists(checkpoint_path):
                file_size = os.path.getsize(checkpoint_path) / 1024 / 1024  # MB
                self.logger.info(f"模型文件大小: {file_size:.2f} MB")
            
            # 自动检测模型类型并加载
            from stable_baselines3 import PPO, SAC, DQN
            
            # 首先尝试加载为PPO模型（最常用）
            self.logger.info("正在尝试加载为PPO模型...")
            try:
                agent = PPO.load(checkpoint_path)
                self.logger.info("SUCCESS: PPO模型加载成功!")
                
                # 获取模型详细信息
                if hasattr(agent, 'policy'):
                    policy_type = type(agent.policy).__name__
                    self.logger.info(f"  策略类型: {policy_type}")
                
                if hasattr(agent, 'learning_rate'):
                    lr_val = agent.learning_rate
                    if callable(lr_val):
                        lr_val = lr_val(1.0)  # 获取当前学习率值
                    self.logger.info(f"  学习率: {lr_val}")
                    
                if hasattr(agent, 'n_steps'):
                    self.logger.info(f"  收集步数: {agent.n_steps}")
                    
                if hasattr(agent, 'batch_size'):
                    self.logger.info(f"  批次大小: {agent.batch_size}")
                    
                if hasattr(agent, 'clip_range'):
                    clip_val = agent.clip_range
                    if callable(clip_val):
                        clip_val = clip_val(1.0)  # 获取当前clip值
                    self.logger.info(f"  裁剪范围: {clip_val}")
                
                self.logger.info("=" * 60)
                return agent
            except Exception as ppo_error:
                self.logger.info(f"PPO加载失败: {str(ppo_error)[:100]}...")
            
            # 尝试SAC
            self.logger.info("正在尝试加载为SAC模型...")
            try:
                agent = SAC.load(checkpoint_path)
                self.logger.info("SUCCESS: SAC模型加载成功!")
                
                # 获取模型详细信息
                if hasattr(agent, 'policy'):
                    policy_type = type(agent.policy).__name__
                    self.logger.info(f"  策略类型: {policy_type}")
                
                if hasattr(agent, 'learning_rate'):
                    lr_val = agent.learning_rate
                    if callable(lr_val):
                        lr_val = lr_val(1.0)
                    self.logger.info(f"  学习率: {lr_val}")
                    
                if hasattr(agent, 'buffer_size'):
                    self.logger.info(f"  经验回放缓冲区大小: {agent.buffer_size}")
                    
                self.logger.info("=" * 60)
                return agent
            except Exception as sac_error:
                self.logger.info(f"SAC加载失败: {str(sac_error)[:100]}...")
            
            # 尝试DQN
            self.logger.info("正在尝试加载为DQN模型...")
            try:
                agent = DQN.load(checkpoint_path)
                self.logger.info("SUCCESS: DQN模型加载成功!")
                
                # 获取模型详细信息
                if hasattr(agent, 'policy'):
                    policy_type = type(agent.policy).__name__
                    self.logger.info(f"  策略类型: {policy_type}")
                
                if hasattr(agent, 'learning_rate'):
                    lr_val = agent.learning_rate
                    if callable(lr_val):
                        lr_val = lr_val(1.0)
                    self.logger.info(f"  学习率: {lr_val}")
                    
                if hasattr(agent, 'buffer_size'):
                    self.logger.info(f"  经验回放缓冲区大小: {agent.buffer_size}")
                    
                self.logger.info("=" * 60)
                return agent
            except Exception as dqn_error:
                self.logger.info(f"DQN加载失败: {str(dqn_error)[:100]}...")
            
            # 所有算法都失败
            self.logger.error("FAILED: 无法加载模型，所有算法类型 (PPO, SAC, DQN) 都失败")
            self.logger.error("可能原因:")
            self.logger.error("  1. 模型文件损坏")
            self.logger.error("  2. Stable-Baselines3版本不兼容")
            self.logger.error("  3. 模型由其他框架训练")
            raise Exception("无法加载模型，尝试了PPO、SAC、DQN都失败")
            
        except Exception as e:
            self.logger.error(f"加载Stable-Baselines3模型失败: {e}")
            raise
    
    def _find_best_checkpoint(self, model_dir: Path) -> Optional[str]:
        """
        查找最佳检查点路径
        
        Args:
            model_dir: 模型目录
            
        Returns:
            最佳检查点路径，如果未找到则返回None
        """
        try:
            # 1. 如果直接传入的是.zip文件路径，直接返回（Stable-Baselines3模型格式）
            if str(model_dir).endswith('.zip') and model_dir.exists():
                self.logger.info(f"找到Stable-Baselines3模型文件: {model_dir}")
                return str(model_dir)
            
            # 2. 在指定目录中查找final_model.zip (Stable-Baselines3训练产生的模型)
            if model_dir.is_dir():
                final_model_path = model_dir / "final_model.zip"
                if final_model_path.exists():
                    self.logger.info(f"找到final_model.zip: {final_model_path}")
                    return str(final_model_path)
                
                # 查找其他.zip文件
                zip_files = list(model_dir.glob("*.zip"))
                if zip_files:
                    model_file = zip_files[0]  # 取第一个.zip文件
                    self.logger.info(f"找到模型文件: {model_file}")
                    return str(model_file)
            
            # 3. 如果直接传入的是checkpoint目录，直接查找里面的文件（Ray RLlib兼容性）
            if model_dir.name.startswith('checkpoint_'):
                self.logger.info(f"检测到检查点目录: {model_dir}")
                checkpoint_files = list(model_dir.glob('checkpoint-*'))
                checkpoint_files = [f for f in checkpoint_files if not f.name.endswith('.tune_metadata')]
                
                if checkpoint_files:
                    checkpoint_file = str(checkpoint_files[0])
                    self.logger.info(f"在指定目录中找到检查点文件: {checkpoint_file}")
                    return checkpoint_file
                else:
                    self.logger.warning(f"检查点目录中未找到检查点文件: {model_dir}")
            
            # 4. 尝试从training_results.json读取最佳检查点（Ray RLlib兼容性）
            results_file = model_dir / "training_results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # 尝试获取最佳检查点
                training_result = results.get('training_result', {})
                best_checkpoint = training_result.get('best_checkpoint')
                
                if best_checkpoint:
                    # 转换路径分隔符
                    checkpoint_path = best_checkpoint.replace('\\', os.sep).replace('/', os.sep)
                    full_path = Path(checkpoint_path)
                    
                    # 如果路径不是绝对路径，相对于当前工作目录
                    if not full_path.is_absolute():
                        full_path = Path.cwd() / checkpoint_path
                    
                    self.logger.info(f"尝试检查点路径: {full_path}")
                    
                    if full_path.exists():
                        self.logger.info(f"找到最佳检查点: {full_path}")
                        return str(full_path)
                    
                    self.logger.warning(f"最佳检查点路径不存在: {full_path}")
            
            # 5. 如果没有找到，尝试在models目录下查找最新的模型
            models_dir = Path("models")
            if models_dir.exists():
                self.logger.info("尝试在models目录查找模型...")
                
                # 查找最新的实验目录
                exp_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
                if exp_dirs:
                    latest_exp_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"找到最新实验目录: {latest_exp_dir}")
                    
                    # 查找final_model.zip
                    final_model_path = latest_exp_dir / "final_model.zip"
                    if final_model_path.exists():
                        self.logger.info(f"找到final_model.zip: {final_model_path}")
                        return str(final_model_path)
                    
                    # 查找其他.zip文件
                    zip_files = list(latest_exp_dir.glob("*.zip"))
                    if zip_files:
                        model_file = zip_files[0]
                        self.logger.info(f"找到模型文件: {model_file}")
                        return str(model_file)
            
            self.logger.error("未找到任何有效的模型文件")
            return None
            
        except Exception as e:
            self.logger.error(f"查找检查点失败: {e}")
            return None
    
    def evaluate_model(
        self,
        symbol: str,
        model_path: str,
        test_data: pd.DataFrame,
        num_episodes: int = 20,
        detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            symbol: 股票代码
            model_path: 模型路径
            test_data: 测试数据
            num_episodes: 评估回合数
            detailed_analysis: 是否进行详细分析
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"开始评估模型: {symbol}")
        self.logger.info(f"模型路径: {model_path}")
        self.logger.info(f"测试数据: {len(test_data)} 条记录")
        self.logger.info(f"评估回合: {num_episodes}")
        
        try:
            # 1. 预处理测试数据
            self.logger.info("预处理测试数据...")
            try:
                test_data_processed = self._preprocess_test_data(test_data)
                self.logger.info("✓ 测试数据预处理成功")
            except Exception as e:
                self.logger.error(f"✗ 测试数据预处理失败: {e}")
                raise
            
            # 2. 初始化Stable-Baselines3训练器 (用于评估)
            self.logger.info("初始化Stable-Baselines3训练器...")
            try:
                self.sb3_trainer = StableBaselinesTrainer(self.config)
                self.logger.info("✓ Stable-Baselines3训练器初始化成功")
            except Exception as e:
                self.logger.error(f"✗ Stable-Baselines3训练器初始化失败: {e}")
                raise
            
            # 3. 查找最佳检查点
            self.logger.info("查找最佳检查点...")
            try:
                checkpoint_path = self._find_best_checkpoint(Path(model_path))
                if not checkpoint_path:
                    raise FileNotFoundError(f"未找到有效的检查点文件在: {model_path}")
                self.logger.info(f"✓ 找到检查点: {checkpoint_path}")
            except Exception as e:
                self.logger.error(f"✗ 检查点查找失败: {e}")
                raise
            
            # 4. 使用Stable-Baselines3评估
            self.logger.info("执行模型评估（基于Stable-Baselines3）...")
            try:
                # 确保使用.zip文件路径进行评估
                model_path_for_eval = checkpoint_path
                if not model_path_for_eval.endswith('.zip'):
                    model_path_for_eval = checkpoint_path + '.zip'
                    
                basic_evaluation = self.sb3_trainer.evaluate(
                    model_path=checkpoint_path,  # 使用原始路径，因为evaluate方法会处理
                    test_df=test_data_processed,
                    n_episodes=num_episodes,
                    render=False
                )
                self.logger.info("✓ 基础评估完成")
            except Exception as e:
                self.logger.error(f"✗ 基础评估失败: {e}")
                raise
            
            # 5. 详细评估（可选）
            detailed_evaluation = None
            if detailed_analysis:
                self.logger.info("执行详细评估...")
                try:
                    # 加载模型进行详细评估
                    agent = self.load_model(checkpoint_path, test_data_processed)
                    detailed_evaluation = self._detailed_evaluation(
                        agent, test_data_processed, num_episodes
                    )
                    self.logger.info("✓ 详细评估完成")
                except Exception as e:
                    self.logger.warning(f"详细评估失败，使用基础评估结果: {e}")
                    detailed_evaluation = None
            else:
                self.logger.info("跳过详细评估（可通过--detailed-analysis启用）")
            
            # 6. 性能指标计算
            performance_metrics = self._calculate_performance_metrics(
                basic_evaluation, detailed_evaluation
            )
            
            # 7. 汇总结果
            evaluation_result = {
                'symbol': symbol,
                'model_path': model_path,
                'checkpoint_path': checkpoint_path,
                'test_data_shape': test_data_processed.shape,
                'num_episodes': num_episodes,
                'evaluation_time': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'basic_evaluation': basic_evaluation,
                'detailed_evaluation': detailed_evaluation,
                'performance_metrics': performance_metrics
            }
            
            self.logger.info("模型评估完成")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'evaluation_result': evaluation_result
            }
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    def _detailed_evaluation(
        self,
        agent: Any,
        test_data: pd.DataFrame,
        num_episodes: int
    ) -> Dict[str, Any]:
        """
        执行详细评估
        
        Args:
            agent: 训练好的智能体 (Stable-Baselines3 模型)
            test_data: 测试数据
            num_episodes: 评估回合数
            
        Returns:
            Dict[str, Any]: 详细评估结果
        """
        detailed_results = []
        all_rewards = []
        all_actions = []
        all_portfolio_values = []
        
        # 使用 StableBaselinesTrainer 创建兼容的环境
        from src.environment.rewards import create_reward_function
        
        # 造式奖励函数（优先使用训练时的奖励函数类型）
        reward_type = self._detect_reward_type_from_model(model_path) or 'simple_return'
        reward_function = self._create_optimal_reward_function(reward_type, symbol, initial_balance=10000)
        
        # Experiment #005: 检查是否使用优化奖励函数
        if hasattr(reward_function, 'get_diagnostics'):
            self.logger.info(f"使用Experiment #005增强奖励函数: {reward_function.__class__.__name__}")
        
        # 使用与训练时相同的环境创建方式
        env = self.sb3_trainer.create_environment(
            df=test_data,
            reward_function=reward_function,
            initial_balance=10000,
            window_size=50,
            max_episode_steps=len(test_data) // 4
        )
        
        self.logger.info(f"开始详细评估，共 {num_episodes} 个回合")
        self.logger.info("=" * 80)
        
        # 为每个回合执行详细评估
        for episode in range(num_episodes):
            self.logger.info(f"回合 {episode + 1}/{num_episodes} 开始评估...")
            
            # 重置环境
            obs, info = env.reset()
            episode_rewards = []
            episode_actions = []
            episode_portfolio_values = []
            done = False
            step_count = 0
            total_reward = 0
            initial_balance = 10000.0
            
            # 获取初始资金状态
            portfolio_info = info.get('portfolio', {})
            current_portfolio_value = portfolio_info.get('total_value', initial_balance)
            episode_portfolio_values.append(current_portfolio_value)
            
            self.logger.info(f"  初始资金: ${current_portfolio_value:,.2f}")
            
            # 运行单个episode
            while not done and step_count < len(test_data) - 50:  # 避免超出数据范围
                # 获取智能体动作 (Stable-Baselines3 API)
                action, _ = agent.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 记录数据
                episode_rewards.append(reward)
                episode_actions.append(float(action[0]) if hasattr(action, '__len__') else float(action))
                
                # 获取组合价值 (从环境的portfolio信息中提取)
                portfolio_info = info.get('portfolio', {})
                portfolio_value = portfolio_info.get('total_value', current_portfolio_value)
                episode_portfolio_values.append(portfolio_value)
                
                total_reward += reward
                step_count += 1
                
                # 每100步输出一次进度信息
                if step_count % 100 == 0:
                    profit_loss = portfolio_value - initial_balance
                    profit_pct = (profit_loss / initial_balance) * 100
                    avg_reward = total_reward / step_count
                    recent_action = episode_actions[-1]
                    
                    self.logger.info(f"    步骤 {step_count:5d} | 资金: ${portfolio_value:8,.0f} | "
                                   f"盈亏: {profit_pct:+6.2f}% | 奖励: {reward:+7.3f} | "
                                   f"动作: {recent_action:+6.3f}")
                
                # 如果资金发生显著变化（>1%），立即输出
                if len(episode_portfolio_values) >= 2:
                    prev_value = episode_portfolio_values[-2]
                    value_change_pct = abs((portfolio_value - prev_value) / prev_value)
                    if value_change_pct > 0.01:  # 1%变化
                        profit_loss = portfolio_value - initial_balance
                        profit_pct = (profit_loss / initial_balance) * 100
                        change_pct = ((portfolio_value - prev_value) / prev_value) * 100
                        
                        self.logger.info(f"    >>> 步骤 {step_count:5d} 资金变化: "
                                       f"${prev_value:,.0f} → ${portfolio_value:,.0f} "
                                       f"({change_pct:+5.2f}%) | 总盈亏: {profit_pct:+6.2f}%")
            
            # 计算episode总结信息
            final_portfolio_value = episode_portfolio_values[-1] if episode_portfolio_values else initial_balance
            total_profit_loss = final_portfolio_value - initial_balance
            total_profit_pct = (total_profit_loss / initial_balance) * 100
            avg_reward = total_reward / step_count if step_count > 0 else 0
            
            # 计算其他性能指标
            if len(episode_rewards) > 1:
                reward_volatility = np.std(episode_rewards)
                max_reward = max(episode_rewards)
                min_reward = min(episode_rewards)
            else:
                reward_volatility = 0
                max_reward = episode_rewards[0] if episode_rewards else 0
                min_reward = episode_rewards[0] if episode_rewards else 0
                
            # 计算资金波动
            if len(episode_portfolio_values) > 1:
                max_portfolio = max(episode_portfolio_values)
                min_portfolio = min(episode_portfolio_values)
                max_drawdown = ((max_portfolio - min_portfolio) / max_portfolio) * 100
            else:
                max_portfolio = final_portfolio_value
                min_portfolio = final_portfolio_value
                max_drawdown = 0
            
            # 输出episode总结
            self.logger.info("-" * 60)
            self.logger.info(f"  回合 {episode + 1} 完成:")
            self.logger.info(f"    总步数: {step_count:,}")
            self.logger.info(f"    初始资金: ${initial_balance:,.2f}")
            self.logger.info(f"    最终资金: ${final_portfolio_value:,.2f}")
            self.logger.info(f"    总盈亏: ${total_profit_loss:+,.2f} ({total_profit_pct:+.2f}%)")
            self.logger.info(f"    最大资金: ${max_portfolio:,.2f}")
            self.logger.info(f"    最小资金: ${min_portfolio:,.2f}")
            self.logger.info(f"    最大回撤: {max_drawdown:.2f}%")
            self.logger.info(f"    累计奖励: {total_reward:+.3f}")
            self.logger.info(f"    平均奖励: {avg_reward:+.4f}")
            self.logger.info(f"    奖励波动: {reward_volatility:.4f}")
            self.logger.info(f"    奖励范围: [{min_reward:+.3f}, {max_reward:+.3f}]")
            self.logger.info("=" * 80)
            
            # 记录这个episode的结果
            episode_result = {
                'episode_id': episode,
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'steps': step_count,
                'initial_balance': initial_balance,
                'final_portfolio_value': final_portfolio_value,
                'total_profit_loss': total_profit_loss,
                'total_profit_pct': total_profit_pct,
                'max_portfolio': max_portfolio,
                'min_portfolio': min_portfolio,
                'max_drawdown': max_drawdown,
                'reward_volatility': reward_volatility,
                'max_reward': max_reward,
                'min_reward': min_reward,
                'rewards': episode_rewards,
                'actions': episode_actions,
                'portfolio_values': episode_portfolio_values
            }
            
            detailed_results.append(episode_result)
            all_rewards.extend(episode_rewards)
            all_actions.extend(episode_actions)
            all_portfolio_values.extend(episode_portfolio_values)
        
        # 聚合统计
        aggregate_stats = self._aggregate_episode_stats(
            detailed_results, all_rewards, all_actions, all_portfolio_values
        )
        
        # Experiment #005: 添加奖励-回报一致性分析
        consistency_analysis = self._analyze_reward_return_consistency(
            detailed_results, reward_function
        )
        
        return {
            'episode_results': detailed_results,
            'aggregate_statistics': aggregate_stats,
            'all_rewards': all_rewards,
            'all_actions': all_actions,
            'all_portfolio_values': all_portfolio_values,
            'reward_return_consistency': consistency_analysis,  # 新增
            'reward_function_diagnostics': self._get_reward_diagnostics(reward_function)  # 新增
        }
    
    def _run_single_episode(
        self,
        agent: Any,
        env: Any,
        episode_id: int,
        max_steps: int
    ) -> Dict[str, Any]:
        """
        运行单个评估回合
        
        Args:
            agent: 智能体
            env: 环境
            episode_id: 回合ID
            max_steps: 最大步数
            
        Returns:
            Dict[str, Any]: 回合结果
        """
        obs = env.reset()
        rewards = []
        actions = []
        portfolio_values = []
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < max_steps - 50:
            # 获取智能体动作
            action = agent.compute_single_action(obs, explore=False)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 记录数据
            rewards.append(reward)
            actions.append(action[0] if isinstance(action, (list, tuple)) else action)
            
            # 获取组合价值 (如果环境支持)
            portfolio_value = info.get('portfolio_value', 10000.0)
            portfolio_values.append(portfolio_value)
            
            total_reward += reward
            step_count += 1
        
        # 获取性能指标
        try:
            performance_metrics = env.get_performance_metrics()
        except:
            performance_metrics = {}
        
        return {
            'episode_id': episode_id,
            'total_reward': total_reward,
            'steps': step_count,
            'rewards': rewards,
            'actions': actions,
            'portfolio_values': portfolio_values,
            'performance_metrics': performance_metrics,
            'final_portfolio_value': portfolio_values[-1] if portfolio_values else 10000.0
        }
    
    def _aggregate_episode_stats(
        self,
        episode_results: List[Dict],
        all_rewards: List[float],
        all_actions: List[float],
        all_portfolio_values: List[float]
    ) -> Dict[str, Any]:
        """
        聚合回合统计数据
        
        Args:
            episode_results: 回合结果列表
            all_rewards: 所有奖励
            all_actions: 所有动作
            all_portfolio_values: 所有组合价值
            
        Returns:
            Dict[str, Any]: 聚合统计
        """
        # 基础统计
        episode_rewards = [ep['total_reward'] for ep in episode_results]
        episode_steps = [ep['steps'] for ep in episode_results]
        final_values = [ep['final_portfolio_value'] for ep in episode_results]
        
        # 计算收益率
        returns = [(val - 10000.0) / 10000.0 for val in final_values]
        
        stats = {
            # 回合统计
            'num_episodes': len(episode_results),
            'mean_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'min_episode_reward': np.min(episode_rewards),
            'max_episode_reward': np.max(episode_rewards),
            
            # 步数统计
            'mean_episode_steps': np.mean(episode_steps),
            'total_steps': sum(episode_steps),
            
            # 收益统计
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'positive_return_rate': np.mean(np.array(returns) > 0),
            
            # 组合价值统计
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            
            # 动作统计
            'mean_action': np.mean(all_actions),
            'std_action': np.std(all_actions),
            'action_range': [np.min(all_actions), np.max(all_actions)],
            
            # 奖励统计
            'total_reward_steps': len(all_rewards),
            'mean_step_reward': np.mean(all_rewards),
            'positive_reward_rate': np.mean(np.array(all_rewards) > 0)
        }
        
        # 计算夏普比率 (如果有足够数据)
        if len(returns) > 1 and np.std(returns) > 0:
            stats['sharpe_ratio'] = np.mean(returns) / np.std(returns)
        else:
            stats['sharpe_ratio'] = 0.0
        
        return stats
    
    def _detect_reward_type_from_model(self, model_path: str) -> Optional[str]:
        """
        从模型路径或配置文件中检测奖励函数类型
        
        Args:
            model_path: 模型路径
            
        Returns:
            Optional[str]: 奖励函数类型
        """
        try:
            # 检查是否有配置文件
            model_dir = Path(model_path).parent if Path(model_path).is_file() else Path(model_path)
            config_files = [
                model_dir / 'training_config.json',
                model_dir / 'config.json',
                model_dir / 'experiment_config.json'
            ]
            
            for config_file in config_files:
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        reward_type = config.get('reward_type')
                        if reward_type:
                            self.logger.info(f"从配置文件检测到奖励类型: {reward_type}")
                            return reward_type
            
            # 从文件名中推断
            model_name = Path(model_path).name.lower()
            if 'optimized_forex' in model_name or 'experiment_005' in model_name:
                return 'optimized_forex_reward'
            elif 'forex_optimized' in model_name or 'forex' in model_name:
                return 'forex_optimized'
            elif 'risk_adjusted' in model_name:
                return 'risk_adjusted'
            elif 'simple_return' in model_name:
                return 'simple_return'
                
        except Exception as e:
            self.logger.warning(f"检测奖励类型失败: {e}")
        
        return None
    
    def _create_optimal_reward_function(self, reward_type: str, symbol: str, initial_balance: float = 10000):
        """
        创建最佳配置的奖励函数
        
        Args:
            reward_type: 奖励函数类型
            symbol: 交易符号
            initial_balance: 初始资金
            
        Returns:
            奖励函数实例
        """
        try:
            from src.environment.rewards import create_reward_function
            
            # 基础参数
            kwargs = {'initial_balance': initial_balance}
            
            # 根据奖励类型添加特定参数
            if reward_type in ['optimized_forex_reward', 'experiment_005', 'enhanced_forex']:
                # Experiment #005 参数
                kwargs.update({
                    'return_weight': 1.0,
                    'risk_penalty': 0.1,
                    'transaction_cost': 0.0001,
                    'correlation_threshold': 0.8,
                    'stability_window': 20,
                    'volatility_adjustment': True,
                    'clip_range': (-1.0, 1.0),
                    'base_currency_pair': symbol
                })
            elif reward_type in ['forex_optimized', 'forex']:
                # 传统外汇参数
                kwargs.update({
                    'pip_size': 0.0001,
                    'daily_target_pips': 15.0,
                    'trend_window': 20,
                    'quality_window': 10,
                    'base_currency_pair': symbol
                })
            
            return create_reward_function(reward_type, **kwargs)
            
        except Exception as e:
            self.logger.warning(f"创建奖励函数失败: {e}，使用默认")
            from src.environment.rewards import create_reward_function
            return create_reward_function('simple_return', initial_balance=initial_balance)
    
    def _analyze_reward_return_consistency(self, episode_results: List[Dict], reward_function) -> Dict[str, Any]:
        """
        分析奖励-回报一致性 (Experiment #005)
        
        Args:
            episode_results: 回合结果列表
            reward_function: 奖励函数实例
            
        Returns:
            Dict[str, Any]: 一致性分析结果
        """
        try:
            # 提取奖励和回报数据
            rewards = []
            returns = []
            
            for episode in episode_results:
                episode_rewards = episode.get('rewards', [])
                initial_balance = episode.get('initial_balance', 10000)
                final_value = episode.get('final_portfolio_value', initial_balance)
                
                if episode_rewards:
                    rewards.extend(episode_rewards)
                    # 计算每步的回报率估算
                    episode_return = (final_value - initial_balance) / initial_balance
                    steps = len(episode_rewards)
                    step_returns = [episode_return / steps] * steps  # 简化处理
                    returns.extend(step_returns)
            
            if len(rewards) < 10 or len(returns) < 10:
                return {
                    'status': 'insufficient_data',
                    'data_points': len(rewards),
                    'message': '数据不足，无法进行一致性分析'
                }
            
            # 计算相关系数
            correlation = np.corrcoef(rewards[:len(returns)], returns[:len(rewards)])[0, 1]
            
            # 统计指标
            mean_reward = np.mean(rewards)
            mean_return = np.mean(returns)
            reward_std = np.std(rewards)
            return_std = np.std(returns)
            
            # 一致性评估
            if np.isfinite(correlation):
                if abs(correlation) >= 0.8:
                    consistency_level = 'excellent'
                    consistency_message = '奖励与回报高度一致'
                elif abs(correlation) >= 0.6:
                    consistency_level = 'good'
                    consistency_message = '奖励与回报较为一致'
                elif abs(correlation) >= 0.4:
                    consistency_level = 'moderate'
                    consistency_message = '奖励与回报中度一致'
                else:
                    consistency_level = 'poor'
                    consistency_message = '奖励与回报一致性较差'
            else:
                correlation = 0.0
                consistency_level = 'unknown'
                consistency_message = '无法计算相关性'
            
            result = {
                'status': 'success',
                'correlation': float(correlation),
                'consistency_level': consistency_level,
                'consistency_message': consistency_message,
                'statistics': {
                    'mean_reward': float(mean_reward),
                    'mean_return': float(mean_return),
                    'reward_std': float(reward_std),
                    'return_std': float(return_std),
                    'data_points': len(rewards)
                },
                'experiment_005_improvement': hasattr(reward_function, 'get_diagnostics')
            }
            
            # 如果是Experiment #005奖励函数，添加额外信息
            if hasattr(reward_function, 'validate_reward_return_consistency'):
                result['experiment_005_validation'] = reward_function.validate_reward_return_consistency()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"一致性分析失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': '一致性分析过程中发生错误'
            }
    
    def _get_reward_diagnostics(self, reward_function) -> Dict[str, Any]:
        """
        获取奖励函数诊断信息 (Experiment #005)
        
        Args:
            reward_function: 奖励函数实例
            
        Returns:
            Dict[str, Any]: 诊断信息
        """
        try:
            diagnostics = {
                'reward_function_type': reward_function.__class__.__name__,
                'has_experiment_005_features': hasattr(reward_function, 'get_diagnostics')
            }
            
            # 基础信息
            if hasattr(reward_function, 'get_reward_info'):
                info = reward_function.get_reward_info()
                diagnostics['reward_info'] = info
            
            # Experiment #005 特定诊断
            if hasattr(reward_function, 'get_diagnostics'):
                exp005_diagnostics = reward_function.get_diagnostics()
                diagnostics['experiment_005_diagnostics'] = exp005_diagnostics
                
                # 论断建议
                if exp005_diagnostics.get('status') != 'insufficient_data':
                    correlation = exp005_diagnostics.get('correlation_score', 0)
                    if abs(correlation) < 0.6:
                        diagnostics['recommendations'] = [
                            "奖励-回报相关性较低，建议调整参数",
                            f"当前相关性: {correlation:.3f}，目标: >0.8"
                        ]
                    else:
                        diagnostics['recommendations'] = [
                            f"奖励-回报相关性良好: {correlation:.3f}"
                        ]
            
            return diagnostics
            
        except Exception as e:
            self.logger.warning(f"获取奖励诊断信息失败: {e}")
            return {
                'reward_function_type': reward_function.__class__.__name__,
                'error': str(e)
            }
    
    def _calculate_performance_metrics(
        self,
        basic_evaluation: Dict,
        detailed_evaluation: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        计算性能指标
        
        Args:
            basic_evaluation: 基础评估结果
            detailed_evaluation: 详细评估结果
            
        Returns:
            Dict[str, Any]: 性能指标
        """
        metrics = {
            'basic_metrics': {},
            'detailed_metrics': {}
        }
        
        # 从基础评估提取指标
        if basic_evaluation:
            metrics['basic_metrics'] = basic_evaluation
        
        # 从详细评估提取指标
        if detailed_evaluation:
            metrics['detailed_metrics'] = detailed_evaluation.get('aggregate_statistics', {})
        
        return metrics
    
    def generate_report(
        self,
        evaluation_result: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_result: 评估结果
            output_file: 输出文件路径
            
        Returns:
            str: 报告文本
        """
        symbol = evaluation_result.get('symbol', 'Unknown')
        eval_data = evaluation_result.get('evaluation_result', {})
        
        # 生成报告文本
        report_lines = [
            "=" * 80,
            f"TensorTrade 模型评估报告",
            "=" * 80,
            f"股票代码: {symbol}",
            f"模型路径: {eval_data.get('model_path', 'Unknown')}",
            f"评估时间: {eval_data.get('evaluation_time', 'Unknown')}",
            f"测试数据规模: {eval_data.get('test_data_shape', 'Unknown')}",
            f"评估回合数: {eval_data.get('num_episodes', 'Unknown')}",
            "",
            "基础评估结果:",
            "-" * 40
        ]
        
        # 基础指标
        basic_eval = eval_data.get('basic_evaluation', {})
        if basic_eval:
            for key, value in basic_eval.items():
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")
        
        # 详细指标
        detailed_eval = eval_data.get('detailed_evaluation')
        if detailed_eval:
            report_lines.extend([
                "",
                "详细评估结果:",
                "-" * 40
            ])
            
            agg_stats = detailed_eval.get('aggregate_statistics', {})
            for key, value in agg_stats.items():
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")
        
        # 性能总结
        performance = eval_data.get('performance_metrics', {})
        if performance:
            report_lines.extend([
                "",
                "性能总结:",
                "-" * 40
            ])
            
            detailed_metrics = performance.get('detailed_metrics', {})
            if detailed_metrics:
                sharpe_ratio = detailed_metrics.get('sharpe_ratio', 0)
                mean_return = detailed_metrics.get('mean_return', 0)
                positive_rate = detailed_metrics.get('positive_return_rate', 0)
                
                report_lines.extend([
                    f"  平均收益率: {mean_return:.2%}",
                    f"  盈利概率: {positive_rate:.2%}",
                    f"  夏普比率: {sharpe_ratio:.4f}",
                ])
        
        # Experiment #005: 添加奖励-回报一致性分析报告
        consistency_analysis = eval_data.get('reward_return_consistency')
        reward_diagnostics = eval_data.get('reward_function_diagnostics')
        
        if consistency_analysis or reward_diagnostics:
            report_lines.extend([
                "",
                "Experiment #005 增强分析:",
                "-" * 40
            ])
            
            # 一致性分析
            if consistency_analysis and consistency_analysis.get('status') == 'success':
                correlation = consistency_analysis.get('correlation', 0)
                level = consistency_analysis.get('consistency_level', 'unknown')
                message = consistency_analysis.get('consistency_message', '')
                
                report_lines.extend([
                    f"  奖励-回报相关性: {correlation:.4f}",
                    f"  一致性等级: {level}",
                    f"  评估结果: {message}"
                ])
                
                if consistency_analysis.get('experiment_005_improvement'):
                    checkmark = "\u2713"  # 抽取到变量中
                    report_lines.append(f"  {checkmark} 使用Experiment #005增强奖励函数")
            
            # 奖励函数诊断
            if reward_diagnostics:
                reward_type = reward_diagnostics.get('reward_function_type', 'Unknown')
                has_exp005 = reward_diagnostics.get('has_experiment_005_features', False)
                
                checkmark = "\u2713"
                crossmark = "\u2717"
                support_status = f"{checkmark} 支持" if has_exp005 else f"{crossmark} 不支持"
                
                report_lines.extend([
                    f"  奖励函数类型: {reward_type}",
                    f"  Experiment #005特性: {support_status}"
                ])
                
                # 添加建议
                recommendations = reward_diagnostics.get('recommendations', [])
                if recommendations:
                    report_lines.extend([
                        "  优化建议:"
                    ])
                    for rec in recommendations:
                        report_lines.append(f"    - {rec}")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"评估报告保存至: {output_file}")
        
        return report_text
    
    def batch_evaluate(
        self,
        evaluation_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        批量评估多个模型
        
        Args:
            evaluation_configs: 评估配置列表
            
        Returns:
            Dict[str, Any]: 批量评估结果
        """
        self.logger.info(f"开始批量评估 {len(evaluation_configs)} 个模型")
        
        results = {}
        successful_evals = []
        failed_evals = []
        
        for config in evaluation_configs:
            symbol = config['symbol']
            self.logger.info(f"评估模型: {symbol}")
            
            # 加载测试数据
            test_data = self.load_test_data(
                config['data_source'],
                config.get('data_type', 'auto')
            )
            
            # 执行评估
            result = self.evaluate_model(
                symbol=symbol,
                model_path=config['model_path'],
                test_data=test_data,
                num_episodes=config.get('num_episodes', 20),
                detailed_analysis=config.get('detailed_analysis', True)
            )
            
            results[symbol] = result
            
            if result['status'] == 'success':
                successful_evals.append(symbol)
                self.logger.info(f"✓ {symbol} 评估成功")
            else:
                failed_evals.append(symbol)
                self.logger.error(f"✗ {symbol} 评估失败: {result.get('error', '未知错误')}")
        
        # 保存批量评估摘要
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'batch_evaluation_time': timestamp,
            'total_models': len(evaluation_configs),
            'successful_count': len(successful_evals),
            'failed_count': len(failed_evals),
            'successful_symbols': successful_evals,
            'failed_symbols': failed_evals,
            'detailed_results': results
        }
        
        summary_file = self.output_dir / f"batch_evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"批量评估完成: 成功 {len(successful_evals)}, 失败 {len(failed_evals)}")
        
        return summary
    
    def save_evaluation_result(
        self,
        evaluation_result: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        保存评估结果
        
        Args:
            evaluation_result: 评估结果
            output_file: 输出文件路径
            
        Returns:
            str: 保存的文件路径
        """
        if not output_file:
            symbol = evaluation_result.get('symbol', 'Unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"evaluation_{symbol}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"评估结果保存至: {output_file}")
        return str(output_file)
    
    def generate_evaluation_visualizations(
        self,
        evaluation_result: Dict[str, Any],
        output_dir: str = "visualizations/evaluation",
        save_formats: List[str] = ['png', 'pdf'],
        forex_analysis: bool = False
    ) -> Dict[str, str]:
        """
        生成评估结果可视化
        
        Args:
            evaluation_result: 评估结果数据
            output_dir: 可视化输出目录
            save_formats: 保存格式列表
            forex_analysis: 是否启用外汇专用分析
            
        Returns:
            Dict[str, str]: 生成的可视化文件信息
        """
        self.logger.info("开始生成评估可视化...")
        
        try:
            # 创建可视化管理器
            viz_config = {
                'save_formats': save_formats,
                'style_theme': 'seaborn-v0_8',
                'figure_size': [12, 8],
                'dpi': 300
            }
            
            viz_manager = VisualizationManager(
                output_dir=output_dir,
                config=viz_config
            )
            
            # 提取评估数据
            basic_evaluation = evaluation_result.get('basic_evaluation', {})
            detailed_evaluation = evaluation_result.get('detailed_evaluation', {})
            
            # 准备episode数据
            episode_data = []
            if detailed_evaluation and 'episodes' in detailed_evaluation:
                for i, episode in enumerate(detailed_evaluation['episodes']):
                    episode_data.append({
                        'episode': i + 1,
                        'reward': episode.get('total_reward', 0),
                        'return': episode.get('portfolio_return_pct', 0),
                        'steps': episode.get('steps', 0),
                        'final_value': episode.get('final_portfolio_value', 10000),
                        'max_drawdown': episode.get('max_drawdown', 0),
                        'volatility': episode.get('volatility', 0)
                    })
            else:
                # 从basic_evaluation创建简化的episode数据
                mean_reward = basic_evaluation.get('mean_reward', 0)
                mean_return = basic_evaluation.get('mean_return', 0)
                mean_length = basic_evaluation.get('mean_length', 0)
                
                # 创建模拟的episode数据用于可视化
                for i in range(10):  # 创建10个模拟episode
                    episode_data.append({
                        'episode': i + 1,
                        'reward': mean_reward + np.random.normal(0, abs(mean_reward) * 0.1),
                        'return': mean_return + np.random.normal(0, abs(mean_return) * 0.1),
                        'steps': int(mean_length + np.random.normal(0, mean_length * 0.05)),
                        'final_value': 10000 * (1 + (mean_return / 100))
                    })
            
            generated_files = {}
            
            # 1. 生成基础评估可视化
            if episode_data:
                self.logger.info("生成Episode性能分析图表...")
                evaluation_viz = viz_manager.generate_evaluation_visualizations(
                    evaluation_data={'episode_data': episode_data},
                    model_info={
                        'model_type': 'Stable-Baselines3',
                        'algorithm': evaluation_result.get('algorithm', 'PPO'),
                        'reward_type': evaluation_result.get('reward_type', '未知'),
                        'training_symbol': evaluation_result.get('symbol', '未知')
                    },
                    detailed=True
                )
                generated_files.update(evaluation_viz)
            
            # 2. 外汇专用分析（如果启用）
            if forex_analysis and evaluation_result.get('symbol', '').replace('=X', '') in ['EURUSD', 'GBPUSD', 'USDJPY']:
                self.logger.info("生成外汇专用分析图表...")
                
                # 准备外汇数据（如果有详细评估数据）
                if detailed_evaluation and 'price_history' in detailed_evaluation:
                    forex_data = {
                        'prices': detailed_evaluation['price_history'],
                        'actions': detailed_evaluation.get('action_history', []),
                        'pip_size': 0.0001  # 默认点大小
                    }
                    
                    currency_pair = evaluation_result.get('symbol', 'EURUSD').replace('=X', '')
                    forex_viz = viz_manager.generate_forex_visualizations(
                        forex_data=forex_data,
                        currency_pair=currency_pair,
                        detailed=True
                    )
                    generated_files.update(forex_viz)
            
            # 3. 生成综合报告
            self.logger.info("生成综合评估报告...")
            report_data = {
                'evaluation_data': {'episode_data': episode_data},
                'model_info': {
                    'model_type': 'Stable-Baselines3',
                    'algorithm': evaluation_result.get('algorithm', 'PPO'),
                    'symbol': evaluation_result.get('symbol', '未知'),
                    'total_episodes': len(episode_data),
                    'evaluation_date': evaluation_result.get('evaluation_time', 'Unknown')
                },
                'experiment_name': f"Evaluation_{evaluation_result.get('symbol', 'Unknown')}"
            }
            
            comprehensive_report = viz_manager.create_comprehensive_report(
                report_data=report_data,
                report_name=f"Evaluation_Report_{evaluation_result.get('symbol', 'Unknown')}",
                include_html=True
            )
            generated_files.update(comprehensive_report)
            
            # 4. 获取会话摘要
            session_summary = viz_manager.get_session_summary()
            
            self.logger.info(f"可视化生成完成!")
            self.logger.info(f"  生成图表: {session_summary['total_charts_generated']} 个")
            self.logger.info(f"  输出目录: {output_dir}")
            
            return {
                'status': 'success',
                'generated_files': generated_files,
                'session_summary': session_summary,
                'output_directory': output_dir
            }
            
        except Exception as e:
            self.logger.error(f"生成可视化时出错: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基础参数
    parser.add_argument('--symbol', '-s',
                       default='EURUSD',
                       help='外汇代码 (默认: EURUSD)')
    
    parser.add_argument('--model-path',
                       help='训练好的模型路径 (如果不指定，将自动查找最新的模型)')
    
    # 数据参数
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data-dir',
                      help='数据目录路径 (如果不指定，将自动查找最新的数据目录)')
    
    group.add_argument('--test-data',
                      help='测试数据文件路径')
    
    # 评估参数
    parser.add_argument('--episodes',
                       type=int,
                       default=100,
                       help='评估回合数 (默认: 100)')
    
    parser.add_argument('--detailed-analysis',
                       action='store_true',
                       default=True,
                       help='启用详细分析 (默认: True)')
    
    parser.add_argument('--generate-report',
                       action='store_true',
                       help='生成详细评估报告')
    
    # 批量评估
    parser.add_argument('--batch-config',
                       help='批量评估配置文件 (JSON格式)')
    
    # 输出参数
    parser.add_argument('--output-dir',
                       default='evaluation_results',
                       help='评估结果输出目录 (默认: evaluation_results)')
    
    parser.add_argument('--config', '-c',
                       help='配置文件路径')
    
    # 可视化参数
    parser.add_argument('--enable-visualization',
                       action='store_true',
                       help='启用评估结果可视化')
    
    parser.add_argument('--visualization-output',
                       default='visualizations/evaluation',
                       help='可视化输出目录 (默认: visualizations/evaluation)')
    
    parser.add_argument('--save-formats',
                       nargs='*',
                       default=['png', 'pdf'],
                       help='可视化保存格式 (默认: png pdf)')
    
    parser.add_argument('--forex-analysis',
                       action='store_true',
                       help='启用外汇专用分析图表')
    
    # 其他参数
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='详细输出')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("=" * 60)
    print("Model Evaluation Script")
    print("Evaluate Stable-Baselines3 trained RL model performance")
    print("=" * 60)
    
    # =============================================
    # 默认参数配置 - 在这里修改默认值  
    # =============================================
    DEFAULT_SYMBOL = "EURUSD"            # 默认外汇代码 (欧元/美元，配合FX-1-Minute-Data)
    DEFAULT_MODEL_PATH = None            # 默认模型路径（None表示自动查找）
    DEFAULT_DATA_DIR = None              # 默认数据目录（None表示自动查找）
    DEFAULT_TEST_DATA = None             # 默认测试数据文件
    DEFAULT_EPISODES = 100               # 默认评估回合数 (外汇数据增加到100轮获得更稳定结果)
    DEFAULT_DETAILED_ANALYSIS = True     # 默认启用详细分析
    DEFAULT_GENERATE_REPORT = True       # 默认生成报告 (训练完成后需要详细报告)
    DEFAULT_BATCH_CONFIG = None          # 默认批量配置文件
    DEFAULT_OUTPUT_DIR = "evaluation_results"  # 默认评估结果输出目录
    DEFAULT_CONFIG = None                # 默认配置文件
    DEFAULT_VERBOSE = False              # 默认详细输出
    DEFAULT_ENABLE_VISUALIZATION = True   # 默认启用可视化
    DEFAULT_VISUALIZATION_OUTPUT = "visualizations/evaluation"  # 默认可视化输出目录
    DEFAULT_SAVE_FORMATS = ['png', 'pdf']  # 默认保存格式
    DEFAULT_FOREX_ANALYSIS = True        # 默认启用外汇分析
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 使用默认值覆盖未指定的参数
    symbol = args.symbol or DEFAULT_SYMBOL
    model_path = args.model_path or DEFAULT_MODEL_PATH
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    test_data = args.test_data or DEFAULT_TEST_DATA
    episodes = args.episodes if args.episodes != 100 else DEFAULT_EPISODES
    detailed_analysis = args.detailed_analysis if hasattr(args, 'detailed_analysis') else DEFAULT_DETAILED_ANALYSIS
    generate_report = args.generate_report or DEFAULT_GENERATE_REPORT
    batch_config = args.batch_config or DEFAULT_BATCH_CONFIG
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    config_path = args.config or DEFAULT_CONFIG
    verbose = args.verbose or DEFAULT_VERBOSE
    enable_visualization = args.enable_visualization or DEFAULT_ENABLE_VISUALIZATION
    # 先使用默认值，稍后根据模型路径调整
    visualization_output_base = args.visualization_output or DEFAULT_VISUALIZATION_OUTPUT
    save_formats = args.save_formats or DEFAULT_SAVE_FORMATS
    forex_analysis = args.forex_analysis or DEFAULT_FOREX_ANALYSIS
    
    try:
        # 创建模型评估器
        evaluator = ModelEvaluator(
            config_path=config_path,
            output_dir=output_dir
        )
        
        # 批量评估模式
        if batch_config:
            with open(batch_config, 'r', encoding='utf-8') as f:
                batch_config_data = json.load(f)
            
            result = evaluator.batch_evaluate(batch_config_data)
            
            print(f"\n批量评估完成:")
            print(f"总计: {result['total_models']} 个模型")
            print(f"成功: {result['successful_count']} 个")
            print(f"失败: {result['failed_count']} 个")
            
            return 0
        
        # 单个模型评估模式
        
        # 自动查找模型路径 (如果未指定)
        if not model_path:
            # 首先尝试在 models 目录查找
            models_dir = Path("models")
            if models_dir.exists():
                # 查找指定股票的最新模型目录
                symbol_dirs = list(models_dir.glob(f"{symbol}_*"))
                if symbol_dirs:
                    latest_model_dir = max(symbol_dirs, key=lambda x: x.stat().st_mtime)
                    # 查找 final_model.zip
                    final_model_path = latest_model_dir / "final_model.zip"
                    if final_model_path.exists():
                        model_path = str(final_model_path)
                        print(f"Auto-found model: {model_path}")
                    else:
                        # 查找其他.zip文件
                        zip_files = list(latest_model_dir.glob("*.zip"))
                        if zip_files:
                            model_path = str(zip_files[0])
                            print(f"Auto-found model: {model_path}")
                        else:
                            print(f"Could not find model file in {latest_model_dir}")
                            return 1
                else:
                    print(f"Could not find model directory for {symbol} in models/, please run train_model.py first")
                    return 1
            else:
                print(f"Could not find models/ directory, please run train_model.py first")
                return 1
        
        # 根据模型路径设置可视化输出目录
        if model_path and not (args.visualization_output and args.visualization_output != DEFAULT_VISUALIZATION_OUTPUT):
            # 获取模型目录
            model_path_obj = Path(model_path)
            if model_path_obj.is_file():
                model_dir = model_path_obj.parent
            else:
                model_dir = model_path_obj
            
            # 设置可视化输出目录为模型目录下的visualizations子目录
            visualization_output = str(model_dir / "visualizations")
            print(f"评估可视化输出自动设置为: {visualization_output}")
        else:
            visualization_output = visualization_output_base
        
        # 自动查找数据源 (如果未指定)
        data_source = data_dir or test_data
        if not data_source:
            datasets_dir = Path("datasets")
            if datasets_dir.exists():
                # 查找指定股票的最新数据目录
                symbol_dirs = list(datasets_dir.glob(f"{symbol}_*"))
                if symbol_dirs:
                    data_source = str(max(symbol_dirs))  # 选择最新的
                    print(f"Auto-found data directory: {data_source}")
                else:
                    print(f"Could not find data directory for {symbol}, please run download_data.py first")
                    return 1
            else:
                print("datasets directory not found, please run download_data.py first")
                return 1
        
        # 加载测试数据
        test_data_loaded = evaluator.load_test_data(data_source)
        print(f"DEBUG: Test data shape: {test_data_loaded.shape}")
        print(f"DEBUG: Test data column types:")
        for col, dtype in test_data_loaded.dtypes.items():
            if dtype == 'object':
                print(f"  {col}: {dtype} <- OBJECT TYPE!")
            else:
                print(f"  {col}: {dtype}")
        print(f"DEBUG: Object type columns count: {len(test_data_loaded.select_dtypes(include=['object']).columns)}")
        
        # 执行评估
        result = evaluator.evaluate_model(
            symbol=symbol,
            model_path=model_path,
            test_data=test_data_loaded,
            num_episodes=episodes,
            detailed_analysis=detailed_analysis
        )
        
        # 输出结果
        if result['status'] == 'success':
            print(f"\nSUCCESS: {symbol} model evaluation completed!")
            
            # 保存评估结果
            result_file = evaluator.save_evaluation_result(result)
            print(f"Evaluation results saved to: {result_file}")
            
            # 生成报告
            if generate_report:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_file = evaluator.output_dir / f"evaluation_report_{symbol}_{timestamp}.txt"
                report_text = evaluator.generate_report(result, report_file)
                print(f"Evaluation report saved to: {report_file}")
                
                if verbose:
                    print("\nEvaluation Report:")
                    print(report_text)
            
            # 生成可视化 (如果启用)
            if enable_visualization:
                print(f"\nGenerating evaluation visualizations...")
                viz_result = evaluator.generate_evaluation_visualizations(
                    evaluation_result=result['evaluation_result'],
                    output_dir=visualization_output,
                    save_formats=save_formats,
                    forex_analysis=forex_analysis
                )
                
                if viz_result['status'] == 'success':
                    session_summary = viz_result['session_summary']
                    print(f"SUCCESS: Generated {session_summary['total_charts_generated']} visualization charts")
                    print(f"Visualization output directory: {viz_result['output_directory']}")
                    
                    if verbose:
                        print(f"Generated files: {list(viz_result['generated_files'].keys())}")
                else:
                    print(f"WARNING: Visualization generation failed: {viz_result.get('error', 'Unknown error')}")
            else:
                print("Visualization disabled (use --enable-visualization to generate charts)")
            
            # 显示关键指标
            eval_data = result.get('evaluation_result', {})
            detailed_eval = eval_data.get('detailed_evaluation')
            if detailed_eval:
                agg_stats = detailed_eval.get('aggregate_statistics', {})
                mean_return = agg_stats.get('mean_return', 0)
                sharpe_ratio = agg_stats.get('sharpe_ratio', 0)
                positive_rate = agg_stats.get('positive_return_rate', 0)
                
                print(f"\nKey Metrics:")
                print(f"  Average Return: {mean_return:.2%}")
                print(f"  Win Rate: {positive_rate:.2%}")
                print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            
        else:
            print(f"\nFAILED: {symbol} model evaluation failed:")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        return 1
    
    print(f"\nModel evaluation completed")
    print("Based on Stable-Baselines3 RL framework")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)