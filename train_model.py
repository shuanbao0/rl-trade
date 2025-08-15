#!/usr/bin/env python
"""
模型训练脚本

功能:
1. 从本地文件加载预处理数据
2. 创建和配置交易环境
3. 训练强化学习智能体
4. 保存训练好的模型和结果

使用示例:
  # 直接运行 (默认训练EURUSD，使用预设数据目录，150轮训练，forex_optimized奖励)
  # 【实验#003汇率专用配置】基于汇率交易特征优化的专门训练设置
  python train_model.py

  # 指定外汇和训练轮次
  python train_model.py --symbol GBPUSD --iterations 300

  # 指定数据目录和目标奖励
  python train_model.py --symbol EURUSD --data-dir datasets/EURUSD_20250809_170516 --target-reward 3.0

  # 启用超参数搜索 (自动优化训练参数)
  python train_model.py --symbol EURUSD --hyperparameter-search --iterations 500

  # 从检查点恢复训练
  python train_model.py --symbol EURUSD --resume-from models/EURUSD_20250724/checkpoint_100

  # 高级训练配置 (长时间训练)
  python train_model.py --symbol EURUSD --iterations 1000 --timesteps-total 2000000 --target-reward 5.0

  # 奖励函数配置示例
  python train_model.py --reward-type simple_return                    # 简单收益率奖励
  python train_model.py --reward-type profit_loss                     # 盈亏比奖励  
  python train_model.py --reward-type diversified                     # 多指标综合奖励
  python train_model.py --reward-type risk_adjusted --risk-free-rate 0.03  # 风险调整奖励
  python train_model.py --reward-type log_sharpe --eta 0.01 --scale-factor 100  # 对数夏普奖励
  python train_model.py --reward-type differential_sharpe --adaptive-eta  # 差分夏普奖励
  python train_model.py --reward-type return_drawdown --return-weight 0.6 --drawdown-weight 0.4  # 收益回撤奖励
  python train_model.py --reward-type calmar --calmar-scale 10 --adaptive-weights  # Calmar比率奖励
  python train_model.py --reward-type dynamic_sortino --base-window-size 50 --volatility-threshold 0.02  # 动态索提诺奖励
  python train_model.py --reward-type dts --trend-sensitivity 0.1 --time-decay-factor 0.95  # DTS奖励
  python train_model.py --reward-type regime_aware --detection-window 50 --weight-adaptation-rate 0.1  # 市场状态感知奖励
  python train_model.py --reward-type adaptive_expert --expert-confidence-threshold 0.6 --volatility-regime-threshold 0.02  # 自适应专家奖励
  python train_model.py --reward-type expert_committee --committee-weight-adaptation-rate 0.1 --tchebycheff-rho 0.1  # 专家委员会奖励
  python train_model.py --reward-type multi_objective --pareto-archive-size 100 --committee-temperature 1.0  # 多目标优化奖励
  python train_model.py --reward-type uncertainty_aware --uncertainty-lambda 1.0 --cvar-gamma 0.5  # 不确定性感知奖励
  python train_model.py --reward-type bayesian --cvar-alpha 0.05 --confidence-threshold 0.3  # 贝叶斯不确定性奖励
  python train_model.py --reward-type risk_sensitive --ensemble-size 5 --uncertainty-lambda 1.5  # 风险敏感奖励
  python train_model.py --reward-type curiosity_driven --alpha-curiosity 0.6 --beta-progress 0.4  # 好奇心驱动奖励
  python train_model.py --reward-type intrinsic --forward-model-lr 0.02 --enable-dinat  # 内在动机奖励
  python train_model.py --reward-type exploration --gamma-hierarchical 0.3 --skill-discovery-threshold 0.15  # 探索奖励
  python train_model.py --reward-type self_rewarding --base-reward-weight 0.7 --self-evaluation-weight 0.2  # 自我奖励奖励
  python train_model.py --reward-type meta_ai --enable-meta-judge --dpo-beta 0.15  # Meta AI自我评判奖励
  python train_model.py --reward-type llm_judge --evaluation-window 25 --bias-detection-threshold 0.2  # LLM评判奖励
  python train_model.py --reward-type causal_reward --adjustment-method backdoor --confounding-penalty 0.4  # 因果奖励
  python train_model.py --reward-type causal_inference --adjustment-method dovi --dovi-confidence-level 0.9  # 因果推理奖励
  python train_model.py --reward-type backdoor --temporal-window 60 --confounding-detection-threshold 0.25  # 后门调整奖励
  python train_model.py --reward-type llm_guided --natural-language-spec "Maximize returns with low risk"  # LLM引导奖励
  python train_model.py --reward-type eureka --natural-language-spec "Focus on Sharpe ratio and minimize drawdown"  # EUREKA风格奖励
  python train_model.py --reward-type constitutional --safety-level high --explanation-detail high  # Constitutional AI奖励
  python train_model.py --reward-type ai_guided --enable-iterative-improvement --natural-language-spec "Balance profit and stability"  # AI引导奖励
  python train_model.py --reward-type curriculum_reward --enable-auto-progression --progression-sensitivity 1.2  # 课程学习奖励
  python train_model.py --reward-type progressive --manual-stage intermediate --performance-window 100  # 渐进式奖励
  python train_model.py --reward-type adaptive_difficulty --enable-auto-progression --progression-sensitivity 0.8  # 自适应难度奖励
  python train_model.py --reward-type federated_reward --num-clients 20 --privacy-epsilon 2.0  # 联邦学习奖励
  python train_model.py --reward-type federated --enable-reputation --enable-smart-contracts  # 声誉+智能合约联邦奖励
  python train_model.py --reward-type distributed --privacy-epsilon 0.5 --collaboration-weight 0.5  # 高隐私分布式奖励
  python train_model.py --reward-type blockchain --aggregation-method weighted_avg --min-clients-per-round 5  # 区块链联邦奖励
  python train_model.py --reward-type meta_learning_reward --alpha 0.02 --beta 0.002  # 元学习奖励
  python train_model.py --reward-type maml --enable-self-rewarding --enable-memory-augmentation  # MAML自我奖励
  python train_model.py --reward-type adaptive --adaptation-steps 10 --meta-update-frequency 50  # 自适应元学习
  python train_model.py --reward-type task_adaptive --task-detection-window 30 --adaptation-threshold 0.05  # 任务自适应
  python train_model.py --reward-type forex_optimized --pip-size 0.0001 --daily-target-pips 20  # 汇率专用奖励
  python train_model.py --reward-type forex_simple --trend-window 15 --quality-window 8  # 汇率简化奖励
  python train_model.py --reward-type optimized_forex_reward --exp005-return-weight 1.0 --exp005-risk-penalty 0.1  # Experiment #005 优化外汇奖励
  python train_model.py --reward-type experiment_005 --correlation-threshold 0.8 --stability-window 20  # 实验005 增强版
  python train_model.py --reward-type enhanced_forex --clip-range -1.0 1.0 --volatility-adjustment  # 增强外汇奖励
  python train_model.py --list-reward-types                           # 查看所有可用奖励函数

训练参数说明 (实验#003汇率专用配置):
  - iterations: 训练迭代次数 (默认150轮，汇率训练优化)
  - checkpoint-freq: 检查点保存频率 (默认10轮保存一次)
  - target-reward: 目标奖励值 (默认4.0，适合汇率盈利目标)
  - timesteps-total: 总训练步数 (默认300万步，充分学习汇率规律)
  - hyperparameter-search: 自动优化超参数 (耗时较长但效果更好)

奖励函数参数说明 (汇率专用优化):
  - reward-type: 奖励函数类型 (默认: forex_optimized，汇率专用设计)
    * forex_optimized: 汇率专用综合奖励 (默认，点数收益+趋势跟随+质量控制)
    * forex_simple: 汇率简化奖励 (简单有效的汇率奖励)
    * optimized_forex_reward: Experiment #005 优化奖励 (解决#004奖励-回报不一致问题)
    * experiment_005: 实验005增强版 (直接基于回报计算+数值稳定性控制)
    * enhanced_forex: 增强外汇奖励 (实时相关性监控+稳定范围限制)
    * risk_adjusted: 基于夏普比率的风险调整奖励 (通用型，实验#002使用)
    * simple_return: 基于简单收益率的奖励 (实验#001表现差，不推荐)
  - pip-size: 点大小 (默认: 0.0001，EURUSD标准)
  - daily-target-pips: 日目标点数 (默认: 15点，适中目标)
  - trend-window: 趋势窗口 (默认: 20，适合汇率趋势判断)

环境优化参数 (汇率专用优化):
  - initial-balance: 初始资金 (默认: 50,000，适合汇率规模)
  - transaction-costs: 交易成本 (默认: 0.005%，汇率低成本)
  - learning-rate: 学习率 (默认: 5e-5，适应汇率小波动)
  - clip-range: PPO裁剪范围 (默认: 0.08，保守更新)
  - n-steps: 批次大小 (默认: 2048，获取更多汇率信息)

数据要求:
  - 需要先运行 download_data.py 下载和预处理数据
  - 自动查找 datasets/ 目录下的最新数据
  - 支持训练集、验证集和测试集的完整数据流程
"""

import os
import sys
import argparse
import json
import warnings
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 抑制所有警告（包括兼容性警告）
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow警告

# 抑制numpy弃用警告
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from src.utils.config import Config
    from src.utils.logger import setup_logger, get_default_log_file
    from src.utils.data_utils import get_data_processor
    
    # 导入训练组件
    from src.training import ModernTrainer, HyperparameterOptimizer, TrainingPipeline
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)


# =============================================
# 默认参数配置 - 实验#003 汇率专用优化设置
# =============================================
DEFAULT_SYMBOL = "EURUSD"            # 默认外汇代码 (使用EURUSD外汇数据)
DEFAULT_DATA_DIR = "datasets/EURUSD_20250811_174255"  # 默认数据目录 (使用新的117特征数据集)
DEFAULT_ITERATIONS = 150             # 默认训练迭代次数 (优化到150轮，平衡训练质量和效率)
DEFAULT_CHECKPOINT_FREQ = 10         # 默认检查点保存频率 (每10轮保存一次)
DEFAULT_HYPERPARAMETER_SEARCH = False # 默认不启用超参数搜索
DEFAULT_RESUME_FROM = None           # 默认不从检查点恢复
DEFAULT_TARGET_REWARD = 3.0          # 默认目标奖励值 
DEFAULT_TIMESTEPS_TOTAL = 2000000    # 默认总训练步数 (200万步，平衡训练质量和速度)
DEFAULT_OUTPUT_DIR = "models"        # 默认模型输出目录
DEFAULT_CONFIG = None                # 默认配置文件
DEFAULT_VERBOSE = False              # 默认详细输出
DEFAULT_USE_GPU = True               # 默认启用GPU训练
DEFAULT_NUM_GPUS = 1                 # 默认使用1个GPU

# 可视化默认参数 - 训练过程可视化配置
DEFAULT_ENABLE_VISUALIZATION = True   # 默认启用训练可视化
DEFAULT_VISUALIZATION_FREQ = 50000    # 默认可视化生成频率 (每5万步生成一次)
DEFAULT_VISUALIZATION_OUTPUT = "visualizations/training"  # 默认训练可视化输出目录
DEFAULT_SAVE_FORMATS = ['png', 'pdf'] # 默认保存格式
DEFAULT_GENERATE_FINAL_REPORT = True  # 默认生成最终训练报告

# 奖励函数默认参数 - 实验#003 汇率专用配置
DEFAULT_REWARD_TYPE = "forex_optimized"  # 默认奖励函数类型 (使用汇率专用优化奖励)
DEFAULT_PIP_SIZE = 0.0001             # 默认点大小 (EURUSD标准点)
DEFAULT_DAILY_TARGET_PIPS = 15.0      # 默认日目标点数 (15点，适中目标)
DEFAULT_TREND_WINDOW = 20             # 默认趋势判断窗口
DEFAULT_QUALITY_WINDOW = 10           # 默认交易质量评估窗口 

# 训练稳定性优化参数 - 实验#003 汇率专用优化
DEFAULT_LEARNING_RATE = 5e-5          # 默认学习率 (进一步降低，适应汇率小波动)
DEFAULT_CLIP_RANGE = 0.08             # 默认PPO裁剪范围 (更保守裁剪，适合汇率)
DEFAULT_N_STEPS = 2048                # 默认每次更新步数 (增加批次获取更多汇率信息)
DEFAULT_BATCH_SIZE = 128              # 默认批次大小 (适合汇率训练)
DEFAULT_N_EPOCHS = 6                  # 默认训练轮次 (适度训练避免过拟合)

# GPU多线程优化参数 - 提升训练速度
DEFAULT_N_ENVS = 3                    # 并行环境数量 (GPU加速)
DEFAULT_NUM_THREADS = 3               # 数据加载线程数
DEFAULT_DEVICE = "cuda"               # 设备选择 (默认使用GPU)
DEFAULT_TENSORBOARD_LOG = "./logs/tensorboard"  # TensorBoard日志目录

# 环境参数优化 - 汇率交易专用配置
DEFAULT_INITIAL_BALANCE = 50000       # 默认初始资金 (5万，适合汇率交易规模)
DEFAULT_TRANSACTION_COSTS = 0.00005  # 默认交易成本 (0.005%，汇率市场较低成本)
DEFAULT_MAX_POSITION_SIZE = 0.8       # 默认最大仓位 (80%，汇率交易风控)
DEFAULT_STOP_LOSS_THRESHOLD = 0.05    # 默认止损阈值 (5%)


class ModelTrainer:
    """
    模型训练类
    
    负责加载预处理数据、配置训练环境、训练模型
    基于Stable-Baselines3 RL框架
    """
    
    def __init__(self, config_path: Optional[str] = None, output_dir: str = "models"):
        """
        初始化模型训练器
        
        Args:
            config_path: 配置文件路径
            output_dir: 模型输出目录
        """
        # 加载配置
        self.config = Config(config_file=config_path) if config_path else Config()
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化日志系统
        self.logger = setup_logger(
            name="ModelTrainer",
            level="INFO",
            log_file=get_default_log_file("model_trainer")
        )
        
        # 初始化训练组件
        self.sb3_trainer = None
        self.training_pipeline = None
        self.hyperparameter_optimizer = None
        
        # 初始化数据处理器
        self.data_processor = get_data_processor("ModelTrainer_DataProcessor")
        
        self.logger.info("模型训练器初始化完成 (Stable-Baselines3)")
    
    def load_training_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        加载训练数据 - 使用统一的数据处理工具
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            Dict[str, pd.DataFrame]: 加载的数据集
        """
        return self.data_processor.load_training_data(data_dir)
    
    def train_model(
        self,
        symbol: str,
        data_dir: str,
        iterations: int = 100,
        checkpoint_freq: int = 10,
        hyperparameter_search: bool = False,
        resume_from: Optional[str] = None,
        custom_config: Optional[Dict] = None,
        stop_config: Optional[Dict] = None,
        feature_columns: Optional[List[str]] = None,
        n_envs: int = 1
    ) -> Dict[str, Any]:
        """
        训练模型 (Stable-Baselines3)
        
        Args:
            symbol: 股票代码
            data_dir: 数据目录
            iterations: 训练迭代次数
            checkpoint_freq: 检查点保存频率
            hyperparameter_search: 是否进行超参数搜索
            resume_from: 从检查点恢复训练
            custom_config: 自定义训练配置
            stop_config: 自定义停止条件配置
            feature_columns: 要使用的特征列列表，如果为None则自动选择
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        self.logger.info(f"开始训练: {symbol}")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info(f"训练参数: {iterations} 迭代")
        
        return self._train_with_stable_baselines3(
            symbol=symbol,
            data_dir=data_dir,
            iterations=iterations,
            hyperparameter_search=hyperparameter_search,
            custom_config=custom_config,
            feature_columns=feature_columns,
            n_envs=n_envs
        )
    
    def _train_with_stable_baselines3(
        self,
        symbol: str,
        data_dir: str,
        iterations: int,
        hyperparameter_search: bool,
        custom_config: Optional[Dict] = None,
        feature_columns: Optional[List[str]] = None,
        n_envs: int = 1
    ) -> Dict[str, Any]:
        """
        使用Stable-Baselines3训练模型
        """
        self.logger.info("使用Stable-Baselines3训练")
        
        # 1. 加载数据
        datasets = self.load_training_data(data_dir)
        if 'train' not in datasets:
            raise ValueError("缺少训练数据集")
        
        train_data = datasets['train']
        self.logger.info(f"训练数据: {len(train_data)} 条记录, {len(train_data.columns)} 个特征")
        
        # 处理特征列选择
        if feature_columns == "use_all":
            # 使用所有非OHLC特征
            selected_features = [col for col in train_data.columns if col not in ['Open', 'High', 'Low', 'Close']]
            self.logger.info(f"使用所有特征: {len(selected_features)} 个特征")
        elif isinstance(feature_columns, list):
            # 使用指定的特征列表
            available_features = train_data.columns.tolist()
            selected_features = [col for col in feature_columns if col in available_features]
            missing_features = [col for col in feature_columns if col not in available_features]
            if missing_features:
                self.logger.warning(f"以下特征在数据中不存在，将被忽略: {missing_features}")
            self.logger.info(f"使用指定特征: {len(selected_features)} 个特征")
        else:
            # 自动选择特征（默认行为）
            selected_features = None
            self.logger.info("自动选择特征（根据数据集特征数量决定）")
        
        # 2. 创建训练流水线
        experiment_name = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.training_pipeline = TrainingPipeline(
            config=self.config,
            experiment_name=experiment_name,
            base_path=str(self.output_dir)
        )
        
        # 3. 计算训练步数 - 优先使用用户指定的参数
        if custom_config and 'total_timesteps' in custom_config:
            total_timesteps = custom_config['total_timesteps']
        else:
            total_timesteps = max(iterations * 2000, 50000)  # 确保足够的训练步数
        
        # 4. 训练配置
        training_config = {
            'algorithm': 'ppo',  # 默认使用PPO
            'total_timesteps': total_timesteps,
            'optimize_hyperparameters': hyperparameter_search,
            'evaluate_model': True
        }
        
        # 添加目标奖励参数（如果指定）
        if custom_config and 'target_reward' in custom_config:
            training_config['target_reward'] = custom_config['target_reward']
        
        # 合并自定义配置
        if custom_config:
            training_config.update(custom_config)
        
        # 5. 执行训练实验（使用本地数据，不重新下载）
        self.logger.info(f"开始训练: {total_timesteps} 步")
        
        try:
            # 直接使用已加载的本地数据进行训练，避免重新下载
            # 获取奖励函数配置
            reward_type = training_config.get('reward_type', 'risk_adjusted')
            reward_kwargs = training_config.get('reward_kwargs', {})
            
            if hyperparameter_search:
                self.logger.info("执行超参数优化...")
                self.training_pipeline.run_optimization(
                    data=train_data,
                    algorithm=training_config.get('algorithm', 'ppo'),
                    reward_type=reward_type,
                    n_trials=20,
                    total_timesteps=total_timesteps // 10
                )
            
            # 训练最终模型 - 传递模型超参数
            model_kwargs = {}
            env_kwargs = {}
            
            # 从custom_config中提取模型参数
            if custom_config:
                if 'learning_rate' in custom_config:
                    model_kwargs['learning_rate'] = custom_config['learning_rate']
                if 'clip_range' in custom_config:
                    model_kwargs['clip_range'] = custom_config['clip_range'] 
                if 'n_steps' in custom_config:
                    model_kwargs['n_steps'] = custom_config['n_steps']
                if 'transaction_costs' in custom_config:
                    env_kwargs['transaction_costs'] = custom_config['transaction_costs']
            
            training_result = self.training_pipeline.train_final_model(
                data=train_data,
                total_timesteps=total_timesteps,
                use_best_params=hyperparameter_search,
                save_checkpoints=True,
                checkpoint_freq=checkpoint_freq * 1000,  # 将轮次转换为步数
                reward_type=reward_type,
                reward_kwargs=reward_kwargs,
                model_kwargs=model_kwargs if model_kwargs else None,
                env_kwargs=env_kwargs if env_kwargs else None,
                feature_columns=selected_features,
                n_envs=n_envs
            )
            
            # 评估模型（如果有验证集）
            if 'val' in datasets:
                self.logger.info("执行模型评估...")
                eval_result = self.training_pipeline.evaluate_model(
                    test_data=datasets['val'],
                    n_episodes=10,
                    save_results=True
                )
                training_result['evaluation_result'] = eval_result
                
        except Exception as e:
            self.logger.error(f"训练过程出错: {e}")
            # 使用基础方法作为备选方案，但不重新获取数据
            self.logger.info("使用基础方法重试训练...")
            # 直接使用已加载的本地数据，不调用 prepare_data 避免网络请求
            train_data = datasets['train']  # 使用已加载的本地训练数据
            
            if hyperparameter_search:
                self.training_pipeline.run_optimization(
                    data=train_data,
                    n_trials=20,
                    total_timesteps=total_timesteps // 10
                )
            # 提取模型和环境参数
            model_kwargs = {}
            env_kwargs = {}
            if training_config:
                if 'learning_rate' in training_config:
                    model_kwargs['learning_rate'] = training_config['learning_rate']
                if 'clip_range' in training_config:
                    model_kwargs['clip_range'] = training_config['clip_range'] 
                if 'n_steps' in training_config:
                    model_kwargs['n_steps'] = training_config['n_steps']
                if 'transaction_costs' in training_config:
                    env_kwargs['transaction_costs'] = training_config['transaction_costs']
            
            training_result = self.training_pipeline.train_final_model(
                data=train_data,
                total_timesteps=total_timesteps,
                save_checkpoints=True,
                checkpoint_freq=checkpoint_freq * 1000,  # 将轮次转换为步数
                reward_type=training_config.get('reward_type', 'risk_adjusted'),
                reward_kwargs=training_config.get('reward_kwargs', {}),
                model_kwargs=model_kwargs if model_kwargs else None,
                env_kwargs=env_kwargs if env_kwargs else None,
                feature_columns=selected_features,
                n_envs=n_envs
            )
        
        # 6. 构造返回结果
        model_path = self.training_pipeline.experiment_path
        
        return {
            'status': 'success',
            'symbol': symbol,
            'model_path': str(model_path),
            'training_result': training_result,
            'iterations': iterations,
            'framework': 'stable-baselines3',
            'total_timesteps': total_timesteps
        }
    
    
    def batch_train(
        self,
        symbols_config: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        批量训练多个模型
        
        Args:
            symbols_config: 包含每个股票训练配置的列表
                格式: [{'symbol': 'AAPL', 'data_dir': 'path', 'iterations': 100, ...}]
            
        Returns:
            Dict[str, Any]: 批量训练结果
        """
        self.logger.info(f"开始批量训练 {len(symbols_config)} 个模型")
        
        results = {}
        successful_trains = []
        failed_trains = []
        
        for config in symbols_config:
            symbol = config['symbol']
            self.logger.info(f"训练模型: {symbol}")
            
            result = self.train_model(**config)
            results[symbol] = result
            
            if result['status'] == 'success':
                successful_trains.append(symbol)
                self.logger.info(f"SUCCESS: {symbol} 训练成功")
            else:
                failed_trains.append(symbol)
                self.logger.error(f"FAILED: {symbol} 训练失败: {result.get('error', '未知错误')}")
        
        # 保存批量训练摘要
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'batch_train_time': timestamp,
            'total_models': len(symbols_config),
            'successful_count': len(successful_trains),
            'failed_count': len(failed_trains),
            'successful_symbols': successful_trains,
            'failed_symbols': failed_trains,
            'detailed_results': results
        }
        
        summary_file = self.output_dir / f"batch_train_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"批量训练完成: 成功 {len(successful_trains)}, 失败 {len(failed_trains)}")
        
        return summary
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.training_pipeline:
                if hasattr(self.training_pipeline, 'cleanup'):
                    self.training_pipeline.cleanup()
            
            if self.sb3_trainer:
                if hasattr(self.sb3_trainer, 'cleanup'):
                    self.sb3_trainer.cleanup()
                    
            self.logger.info("训练器资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基础参数
    parser.add_argument('--symbol', '-s',
                       default='EURUSD',
                       help='外汇代码 (默认: EURUSD)')
    
    parser.add_argument('--data-dir',
                       help='预处理数据目录路径 (如果不指定，将自动查找最新的数据目录)')
    
    parser.add_argument('--feature-columns',
                       nargs='*',
                       help='要使用的特征列列表 (如果不指定，将根据数据集自动选择)')
    
    parser.add_argument('--use-all-features',
                       action='store_true',
                       help='使用数据集中的所有特征列 (覆盖feature-columns参数)')
    
    # 训练参数
    parser.add_argument('--iterations', '-i',
                       type=int,
                       default=DEFAULT_ITERATIONS,
                       help=f'训练迭代次数 (默认: {DEFAULT_ITERATIONS})')
    
    parser.add_argument('--checkpoint-freq',
                       type=int,
                       default=20,
                       help='检查点保存频率 (默认: 20)')
    
    parser.add_argument('--hyperparameter-search',
                       action='store_true',
                       help='启用超参数搜索')
    
    parser.add_argument('--resume-from',
                       help='从指定检查点恢复训练')
    
    # 自定义配置参数
    parser.add_argument('--target-reward',
                       type=float,
                       help='目标奖励值')
    
    parser.add_argument('--timesteps-total',
                       type=int,
                       help='总训练步数')
    
    # 输出参数
    parser.add_argument('--output-dir',
                       default='models',
                       help='模型输出目录 (默认: models)')
    
    parser.add_argument('--config', '-c',
                       help='配置文件路径')
    
    # 其他参数
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='详细输出')
    
    # GPU参数
    parser.add_argument('--use-gpu',
                       action='store_true',
                       default=True,
                       help='启用GPU训练 (默认启用，需要CUDA支持)')
    
    parser.add_argument('--no-gpu',
                       action='store_true',
                       help='禁用GPU训练，强制使用CPU')
    
    parser.add_argument('--num-gpus',
                       type=int,
                       default=1,
                       help='使用的GPU数量 (默认: 1)')
    
    # 多线程优化参数
    parser.add_argument('--n-envs',
                       type=int,
                       default=DEFAULT_N_ENVS,
                       help=f'并行环境数量，提升GPU训练速度 (默认: {DEFAULT_N_ENVS})')
    
    parser.add_argument('--num-threads',
                       type=int,
                       default=DEFAULT_NUM_THREADS,
                       help=f'数据加载线程数 (默认: {DEFAULT_NUM_THREADS})')
    
    parser.add_argument('--device',
                       choices=['auto', 'cpu', 'cuda'],
                       default=DEFAULT_DEVICE,
                       help=f'训练设备选择 (默认: {DEFAULT_DEVICE})')
    
    parser.add_argument('--tensorboard-log',
                       default=DEFAULT_TENSORBOARD_LOG,
                       help=f'TensorBoard日志目录 (默认: {DEFAULT_TENSORBOARD_LOG})')
    
    # 可视化参数
    parser.add_argument('--enable-visualization',
                       action='store_true',
                       default=DEFAULT_ENABLE_VISUALIZATION,
                       help=f'启用训练过程可视化 (默认: {DEFAULT_ENABLE_VISUALIZATION})')
    
    parser.add_argument('--disable-visualization',
                       action='store_true',
                       help='禁用训练过程可视化')
    
    parser.add_argument('--visualization-freq',
                       type=int,
                       default=DEFAULT_VISUALIZATION_FREQ,
                       help=f'可视化生成频率 (训练步数) (默认: {DEFAULT_VISUALIZATION_FREQ})')
    
    parser.add_argument('--visualization-output',
                       default=DEFAULT_VISUALIZATION_OUTPUT,
                       help=f'训练可视化输出目录 (默认: {DEFAULT_VISUALIZATION_OUTPUT})')
    
    parser.add_argument('--save-formats',
                       nargs='*',
                       default=DEFAULT_SAVE_FORMATS,
                       help=f'可视化保存格式 (默认: {" ".join(DEFAULT_SAVE_FORMATS)})')
    
    parser.add_argument('--generate-final-report',
                       action='store_true',
                       default=DEFAULT_GENERATE_FINAL_REPORT,
                       help=f'生成最终训练报告 (默认: {DEFAULT_GENERATE_FINAL_REPORT})')
    
    # 奖励函数参数
    parser.add_argument('--reward-type',
                       choices=['forex_optimized', 'forex_simple', 'risk_adjusted', 'simple_return', 'profit_loss', 'diversified', 'log_sharpe', 'return_drawdown', 'dynamic_sortino', 'regime_aware', 'expert_committee',
                               'uncertainty_aware', 'curiosity_driven', 'self_rewarding', 'causal_reward', 'llm_guided', 'curriculum_reward', 'federated_reward', 'meta_learning_reward',
                               'optimized_forex_reward', 'experiment_005', 'enhanced_forex', 'reward_return_consistent', 'stable_forex', 'correlation_fixed',  # Experiment #005 additions
                               'forex', 'fx', 'currency', 'pip_based', 'trend_following', 'forex_basic', 'fx_simple',
                               'default', 'sharpe', 'basic', 'simple', 'pnl', 'comprehensive', 'multi',
                               'differential_sharpe', 'dsr', 'log_dsr', 'calmar', 'return_dd', 'rdd', 'drawdown',
                               'dts', 'adaptive_sortino', 'time_varying_sortino', 'sortino',
                               'adaptive_expert', 'market_state', 'regime', 'state_aware',
                               'committee', 'multi_objective', 'morl', 'experts', 'pareto',
                               'uncertainty', 'bayesian', 'confidence', 'risk_sensitive', 'cvar',
                               'curiosity', 'intrinsic', 'exploration', 'self_improving', 'meta_ai',
                               'causal', 'causal_inference', 'confounding', 'llm', 'eureka', 'constitutional',
                               'curriculum', 'progressive', 'adaptive_difficulty',
                               'federated', 'distributed', 'collaborative', 'privacy_preserving', 'blockchain',
                               'meta_learning', 'maml', 'adaptive', 'meta_gradient', 'self_adapting', 'task_adaptive'],
                       default=DEFAULT_REWARD_TYPE,
                       help=f'奖励函数类型 (默认: {DEFAULT_REWARD_TYPE})')
    
    parser.add_argument('--risk-free-rate',
                       type=float,
                       help='无风险收益率，适用于风险调整和多指标奖励函数')
    
    parser.add_argument('--reward-window-size',
                       type=int,
                       help='奖励函数计算窗口大小')
    
    # LogSharpe奖励函数专用参数
    parser.add_argument('--eta',
                       type=float,
                       help='指数移动平均衰减率，适用于log_sharpe奖励函数 (默认: 0.01)')
    
    parser.add_argument('--scale-factor',
                       type=float,
                       help='奖励缩放因子，适用于log_sharpe奖励函数 (默认: 100.0)')
    
    parser.add_argument('--adaptive-eta',
                       action='store_true',
                       help='启用自适应学习率调整，适用于log_sharpe奖励函数')
    
    # ReturnDrawdown奖励函数专用参数
    parser.add_argument('--return-weight',
                       type=float,
                       help='收益权重，适用于return_drawdown奖励函数 (默认: 0.6)')
    
    parser.add_argument('--drawdown-weight',
                       type=float,
                       help='回撤惩罚权重，适用于return_drawdown奖励函数 (默认: 0.4)')
    
    parser.add_argument('--calmar-scale',
                       type=float,
                       help='Calmar比率缩放因子，适用于return_drawdown奖励函数 (默认: 10.0)')
    
    parser.add_argument('--drawdown-tolerance',
                       type=float,
                       help='回撤容忍度，适用于return_drawdown奖励函数 (默认: 0.05)')
    
    parser.add_argument('--adaptive-weights',
                       action='store_true',
                       help='启用自适应权重调整，适用于return_drawdown奖励函数')
    
    # DynamicSortino奖励函数专用参数
    parser.add_argument('--base-window-size',
                       type=int,
                       help='基础计算窗口大小，适用于dynamic_sortino奖励函数 (默认: 50)')
    
    parser.add_argument('--min-window-size',
                       type=int,
                       help='最小窗口大小，适用于dynamic_sortino奖励函数 (默认: 20)')
    
    parser.add_argument('--max-window-size',
                       type=int,
                       help='最大窗口大小，适用于dynamic_sortino奖励函数 (默认: 200)')
    
    parser.add_argument('--volatility-threshold',
                       type=float,
                       help='波动性阈值，适用于dynamic_sortino奖励函数 (默认: 0.02)')
    
    parser.add_argument('--trend-sensitivity',
                       type=float,
                       help='趋势敏感度，适用于dynamic_sortino奖励函数 (默认: 0.1)')
    
    parser.add_argument('--time-decay-factor',
                       type=float,
                       help='时间衰减因子，适用于dynamic_sortino奖励函数 (默认: 0.95)')
    
    # RegimeAware奖励函数专用参数
    parser.add_argument('--detection-window',
                       type=int,
                       help='市场状态检测窗口大小，适用于regime_aware奖励函数 (默认: 50)')
    
    parser.add_argument('--weight-adaptation-rate',
                       type=float,
                       help='权重自适应学习率，适用于regime_aware奖励函数 (默认: 0.1)')
    
    parser.add_argument('--expert-confidence-threshold',
                       type=float,
                       help='专家策略置信度阈值，适用于regime_aware奖励函数 (默认: 0.6)')
    
    parser.add_argument('--volatility-regime-threshold',
                       type=float,
                       help='波动性状态阈值，适用于regime_aware奖励函数 (默认: 0.02)')
    
    parser.add_argument('--trend-regime-threshold',
                       type=float,
                       help='趋势状态阈值，适用于regime_aware奖励函数 (默认: 0.05)')
    
    parser.add_argument('--regime-stability-threshold',
                       type=float,
                       help='市场状态稳定性阈值，适用于regime_aware奖励函数 (默认: 0.8)')
    
    # ExpertCommittee奖励函数专用参数
    parser.add_argument('--committee-weight-adaptation-rate',
                       type=float,
                       help='委员会权重自适应学习率，适用于expert_committee奖励函数 (默认: 0.1)')
    
    parser.add_argument('--tchebycheff-rho',
                       type=float,
                       help='Tchebycheff标量化参数，适用于expert_committee奖励函数 (默认: 0.1)')
    
    parser.add_argument('--committee-temperature',
                       type=float,
                       help='Softmax温度参数，适用于expert_committee奖励函数 (默认: 1.0)')
    
    parser.add_argument('--pareto-archive-size',
                       type=int,
                       help='Pareto前沿解集大小，适用于expert_committee奖励函数 (默认: 100)')
    
    parser.add_argument('--committee-update-frequency',
                       type=int,
                       help='权重更新频率，适用于expert_committee奖励函数 (默认: 50)')
    
    parser.add_argument('--initial-expert-weights',
                       type=str,
                       help='专家初始权重(JSON格式)，适用于expert_committee奖励函数 (默认: 均匀分布)')
    
    # UncertaintyAware奖励函数专用参数
    parser.add_argument('--uncertainty-lambda',
                       type=float,
                       help='不确定性惩罚权重，适用于uncertainty_aware奖励函数 (默认: 1.0)')
    
    parser.add_argument('--cvar-gamma',
                       type=float,
                       help='CVaR风险惩罚权重，适用于uncertainty_aware奖励函数 (默认: 0.5)')
    
    parser.add_argument('--cvar-alpha',
                       type=float,
                       help='CVaR置信水平，适用于uncertainty_aware奖励函数 (默认: 0.05)')
    
    parser.add_argument('--ensemble-size',
                       type=int,
                       help='集成模型数量，适用于uncertainty_aware奖励函数 (默认: 5)')
    
    parser.add_argument('--confidence-threshold',
                       type=float,
                       help='最低置信度阈值，适用于uncertainty_aware奖励函数 (默认: 0.3)')
    
    # CuriosityDriven奖励函数专用参数
    parser.add_argument('--alpha-extrinsic',
                       type=float,
                       help='外在奖励权重，适用于curiosity_driven奖励函数 (默认: 1.0)')
    
    parser.add_argument('--alpha-curiosity',
                       type=float,
                       help='好奇心奖励权重，适用于curiosity_driven奖励函数 (默认: 0.5)')
    
    parser.add_argument('--beta-progress',
                       type=float,
                       help='学习进度奖励权重，适用于curiosity_driven奖励函数 (默认: 0.3)')
    
    parser.add_argument('--gamma-hierarchical',
                       type=float,
                       help='层次化奖励权重，适用于curiosity_driven奖励函数 (默认: 0.2)')
    
    parser.add_argument('--forward-model-lr',
                       type=float,
                       help='前向模型学习率，适用于curiosity_driven奖励函数 (默认: 0.01)')
    
    parser.add_argument('--progress-window',
                       type=int,
                       help='学习进度计算窗口，适用于curiosity_driven奖励函数 (默认: 50)')
    
    parser.add_argument('--skill-discovery-threshold',
                       type=float,
                       help='技能发现阈值，适用于curiosity_driven奖励函数 (默认: 0.1)')
    
    parser.add_argument('--enable-dinat',
                       action='store_true',
                       help='启用DiNAT特征增强，适用于curiosity_driven奖励函数')
    
    # SelfRewardingReward 参数
    parser.add_argument('--base-reward-weight',
                       type=float,
                       help='基础奖励权重，适用于self_rewarding奖励函数 (默认: 0.6)')
    
    parser.add_argument('--self-evaluation-weight',
                       type=float,
                       help='自我评估权重，适用于self_rewarding奖励函数 (默认: 0.3)')
    
    parser.add_argument('--meta-evaluation-weight',
                       type=float,
                       help='元评估权重，适用于self_rewarding奖励函数 (默认: 0.1)')
    
    parser.add_argument('--dpo-beta',
                       type=float,
                       help='DPO温度参数，适用于self_rewarding奖励函数 (默认: 0.1)')
    
    parser.add_argument('--self-learning-rate',
                       type=float,
                       help='自我学习率，适用于self_rewarding奖励函数 (默认: 0.01)')
    
    parser.add_argument('--evaluation-window',
                       type=int,
                       help='评估窗口大小，适用于self_rewarding奖励函数 (默认: 20)')
    
    parser.add_argument('--bias-detection-threshold',
                       type=float,
                       help='偏差检测阈值，适用于self_rewarding奖励函数 (默认: 0.3)')
    
    parser.add_argument('--enable-meta-judge',
                       action='store_true',
                       help='启用元评判功能，适用于self_rewarding奖励函数')
    
    # CausalReward 参数
    parser.add_argument('--causal-reward-weight',
                       type=float,
                       help='因果奖励权重，适用于causal_reward奖励函数 (默认: 0.6)')
    
    parser.add_argument('--confounding-penalty',
                       type=float,
                       help='混淆惩罚权重，适用于causal_reward奖励函数 (默认: 0.3)')
    
    parser.add_argument('--intervention-bonus',
                       type=float,
                       help='干预奖励权重，适用于causal_reward奖励函数 (默认: 0.1)')
    
    parser.add_argument('--adjustment-method',
                       type=str,
                       choices=['backdoor', 'frontdoor', 'dovi', 'do_calculus'],
                       help='因果调整方法，适用于causal_reward奖励函数 (默认: backdoor)')
    
    parser.add_argument('--confounding-detection-threshold',
                       type=float,
                       help='混淆检测阈值，适用于causal_reward奖励函数 (默认: 0.2)')
    
    parser.add_argument('--temporal-window',
                       type=int,
                       help='时间窗口大小，适用于causal_reward奖励函数 (默认: 50)')
    
    parser.add_argument('--dovi-confidence-level',
                       type=float,
                       help='DOVI置信水平，适用于causal_reward奖励函数 (默认: 0.95)')
    
    # LLMGuidedReward 参数
    parser.add_argument('--natural-language-spec',
                       type=str,
                       help='自然语言奖励规范，适用于llm_guided奖励函数 (默认: "Maximize risk-adjusted returns with moderate risk tolerance")')
    
    parser.add_argument('--enable-iterative-improvement',
                       action='store_true',
                       help='启用迭代改进，适用于llm_guided奖励函数')
    
    parser.add_argument('--safety-level',
                       type=str,
                       choices=['low', 'medium', 'high'],
                       help='安全级别，适用于llm_guided奖励函数 (默认: high)')
    
    parser.add_argument('--explanation-detail',
                       type=str,
                       choices=['low', 'medium', 'high'],
                       help='解释详细程度，适用于llm_guided奖励函数 (默认: medium)')
    
    # CurriculumReward 参数
    parser.add_argument('--enable-auto-progression',
                       action='store_true',
                       help='启用自动阶段转换，适用于curriculum_reward奖励函数')
    
    parser.add_argument('--manual-stage',
                       type=str,
                       choices=['beginner', 'intermediate', 'advanced', 'expert'],
                       help='手动设置课程阶段，适用于curriculum_reward奖励函数')
    
    parser.add_argument('--progression-sensitivity',
                       type=float,
                       help='阶段转换敏感度，适用于curriculum_reward奖励函数 (默认: 1.0)')
    
    parser.add_argument('--performance-window',
                       type=int,
                       help='性能评估窗口大小，适用于curriculum_reward奖励函数 (默认: 50)')
    
    # FederatedReward 参数
    parser.add_argument('--num-clients',
                       type=int,
                       help='联邦学习客户端总数，适用于federated_reward奖励函数 (默认: 10)')
    
    parser.add_argument('--min-clients-per-round',
                       type=int,
                       help='每轮最小参与客户端数，适用于federated_reward奖励函数 (默认: 3)')
    
    parser.add_argument('--privacy-epsilon',
                       type=float,
                       help='差分隐私预算(ε)，适用于federated_reward奖励函数 (默认: 1.0)')
    
    parser.add_argument('--privacy-delta',
                       type=float,
                       help='差分隐私失败概率(δ)，适用于federated_reward奖励函数 (默认: 1e-5)')
    
    parser.add_argument('--enable-reputation',
                       action='store_true',
                       help='启用声誉系统，适用于federated_reward奖励函数')
    
    parser.add_argument('--enable-smart-contracts',
                       action='store_true',
                       help='启用智能合约，适用于federated_reward奖励函数')
    
    parser.add_argument('--aggregation-method',
                       type=str,
                       choices=['secure_avg', 'weighted_avg', 'median', 'trimmed_mean'],
                       help='聚合方法，适用于federated_reward奖励函数 (默认: secure_avg)')
    
    parser.add_argument('--collaboration-weight',
                       type=float,
                       help='协作奖励权重，适用于federated_reward奖励函数 (默认: 0.3)')
    
    # MetaLearningReward 参数
    parser.add_argument('--alpha',
                       type=float,
                       help='MAML内层学习率，适用于meta_learning_reward奖励函数 (默认: 0.01)')
    
    parser.add_argument('--beta',
                       type=float,
                       help='MAML外层学习率，适用于meta_learning_reward奖励函数 (默认: 0.001)')
    
    parser.add_argument('--adaptation-steps',
                       type=int,
                       help='任务适应步数，适用于meta_learning_reward奖励函数 (默认: 5)')
    
    parser.add_argument('--enable-self-rewarding',
                       action='store_true',
                       help='启用自我奖励机制，适用于meta_learning_reward奖励函数')
    
    parser.add_argument('--enable-memory-augmentation',
                       action='store_true',
                       help='启用记忆增强，适用于meta_learning_reward奖励函数')
    
    parser.add_argument('--meta-update-frequency',
                       type=int,
                       help='元更新频率，适用于meta_learning_reward奖励函数 (默认: 100)')
    
    parser.add_argument('--task-detection-window',
                       type=int,
                       help='任务检测窗口大小，适用于meta_learning_reward奖励函数 (默认: 50)')
    
    parser.add_argument('--adaptation-threshold',
                       type=float,
                       help='适应触发阈值，适用于meta_learning_reward奖励函数 (默认: 0.1)')
    
    # Forex专用奖励函数参数
    parser.add_argument('--pip-size',
                       type=float,
                       default=DEFAULT_PIP_SIZE,
                       help=f'汇率点大小，适用于forex奖励函数 (默认: {DEFAULT_PIP_SIZE})')
    
    parser.add_argument('--daily-target-pips',
                       type=float,
                       default=DEFAULT_DAILY_TARGET_PIPS,
                       help=f'日目标点数，适用于forex奖励函数 (默认: {DEFAULT_DAILY_TARGET_PIPS})')
    
    parser.add_argument('--trend-window',
                       type=int,
                       default=DEFAULT_TREND_WINDOW,
                       help=f'趋势判断窗口大小，适用于forex奖励函数 (默认: {DEFAULT_TREND_WINDOW})')
    
    parser.add_argument('--quality-window',
                       type=int,
                       default=DEFAULT_QUALITY_WINDOW,
                       help=f'交易质量评估窗口，适用于forex奖励函数 (默认: {DEFAULT_QUALITY_WINDOW})')
    
    # Experiment #005 OptimizedForexReward 专用参数
    parser.add_argument('--exp005-return-weight',
                       type=float,
                       default=1.0,
                       help='Experiment #005: 回报权重 (默认: 1.0)')
    
    parser.add_argument('--exp005-risk-penalty',
                       type=float,
                       default=0.1,
                       help='Experiment #005: 风险惩罚系数 (默认: 0.1)')
    
    parser.add_argument('--exp005-transaction-cost',
                       type=float,
                       default=0.0001,
                       help='Experiment #005: 交易成本 (默认: 0.0001)')
    
    parser.add_argument('--exp005-correlation-threshold',
                       type=float,
                       default=0.8,
                       help='Experiment #005: 奖励-回报相关性阈值 (默认: 0.8)')
    
    parser.add_argument('--exp005-stability-window',
                       type=int,
                       default=20,
                       help='Experiment #005: 稳定性窗口大小 (默认: 20)')
    
    parser.add_argument('--exp005-volatility-adjustment',
                       action='store_true',
                       help='Experiment #005: 启用波动率调整')
    
    parser.add_argument('--exp005-clip-range',
                       type=float,
                       nargs=2,
                       default=[-1.0, 1.0],
                       help='Experiment #005: 奖励范围限制 (默认: -1.0 1.0)')
    
    parser.add_argument('--list-reward-types',
                       action='store_true',
                       help='列出所有可用的奖励函数类型并退出')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("=" * 60)
    print("TensorTrade 模型训练脚本")
    print("基于预处理数据训练强化学习模型")
    print("=" * 60)
    
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 处理奖励函数类型列表请求
    if args.list_reward_types:
        print("\n可用的奖励函数类型:")
        print("=" * 50)
        try:
            from src.rewards import get_reward_function_info
            reward_info = get_reward_function_info()
            
            for reward_type, info in reward_info.items():
                print(f"\n{reward_type}")
                print(f"   名称: {info['name']}")
                print(f"   描述: {info['description']}")
                print(f"   类别: {info['category']}")
                if 'features' in info and info['features']:
                    print(f"   特性: {', '.join(info['features'][:3])}...")
                    
        except ImportError as e:
            print(f"无法加载奖励函数信息: {e}")
            print("可用类型: risk_adjusted, simple_return, profit_loss, diversified")
        
        print("\n" + "=" * 50)
        print("使用示例:")
        print("  python train_model.py --reward-type risk_adjusted")
        print("  python train_model.py --reward-type simple_return")
        print("  python train_model.py --reward-type profit_loss")
        print("  python train_model.py --reward-type diversified")
        print("\n训练框架:")
        print("  基于Stable-Baselines3")
        print("  支持PPO、SAC、DQN等现代RL算法")
        return 0
    
    # 使用默认值覆盖未指定的参数
    symbol = args.symbol or DEFAULT_SYMBOL
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    iterations = args.iterations if args.iterations != 200 else DEFAULT_ITERATIONS  
    checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq != 20 else DEFAULT_CHECKPOINT_FREQ
    hyperparameter_search = args.hyperparameter_search or DEFAULT_HYPERPARAMETER_SEARCH
    resume_from = args.resume_from or DEFAULT_RESUME_FROM
    target_reward = args.target_reward or DEFAULT_TARGET_REWARD
    timesteps_total = args.timesteps_total or DEFAULT_TIMESTEPS_TOTAL
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    config_path = args.config or DEFAULT_CONFIG
    verbose = args.verbose or DEFAULT_VERBOSE
    
    # 特征列参数处理
    feature_columns = None
    if args.use_all_features:
        feature_columns = "use_all"  # 特殊标记，表示使用所有特征
        print("特征选择: 使用数据集中的所有特征")
    elif args.feature_columns:
        feature_columns = args.feature_columns
        print(f"特征选择: 使用指定的 {len(feature_columns)} 个特征")
        if verbose:
            print(f"   指定特征: {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")
    else:
        print("特征选择: 自动选择（根据数据集特征数量决定）")
    
    # 奖励函数参数处理 - 实验#003汇率专用配置 + 实验#005增强
    reward_type = args.reward_type or DEFAULT_REWARD_TYPE
    pip_size = getattr(args, 'pip_size', DEFAULT_PIP_SIZE)
    daily_target_pips = getattr(args, 'daily_target_pips', DEFAULT_DAILY_TARGET_PIPS)
    trend_window = getattr(args, 'trend_window', DEFAULT_TREND_WINDOW)
    quality_window = getattr(args, 'quality_window', DEFAULT_QUALITY_WINDOW)
    
    # Experiment #005 OptimizedForexReward 参数
    exp005_return_weight = getattr(args, 'exp005_return_weight', 1.0)
    exp005_risk_penalty = getattr(args, 'exp005_risk_penalty', 0.1)
    exp005_transaction_cost = getattr(args, 'exp005_transaction_cost', 0.0001)
    exp005_correlation_threshold = getattr(args, 'exp005_correlation_threshold', 0.8)
    exp005_stability_window = getattr(args, 'exp005_stability_window', 20)
    exp005_volatility_adjustment = getattr(args, 'exp005_volatility_adjustment', False)
    exp005_clip_range = getattr(args, 'exp005_clip_range', [-1.0, 1.0])
    
    # GPU配置处理
    if args.no_gpu:
        use_gpu = False
        num_gpus = 0
        device = 'cpu'
    else:
        use_gpu = DEFAULT_USE_GPU
        num_gpus = args.num_gpus if args.num_gpus else DEFAULT_NUM_GPUS
        device = args.device if args.device != 'auto' else ('cuda' if use_gpu else 'cpu')
    
    # 多线程优化配置
    n_envs = args.n_envs or DEFAULT_N_ENVS
    num_threads = args.num_threads or DEFAULT_NUM_THREADS
    tensorboard_log = args.tensorboard_log or DEFAULT_TENSORBOARD_LOG
    
    # 可视化配置处理
    if args.disable_visualization:
        enable_visualization = False
    else:
        enable_visualization = args.enable_visualization or DEFAULT_ENABLE_VISUALIZATION
    
    visualization_freq = args.visualization_freq or DEFAULT_VISUALIZATION_FREQ
    
    # 可视化输出目录：如果未指定，则使用模型输出目录下的visualizations子目录
    if args.visualization_output and args.visualization_output != DEFAULT_VISUALIZATION_OUTPUT:
        # 用户明确指定了可视化输出目录
        visualization_output = args.visualization_output
    else:
        # 自动设置为模型输出目录下的visualizations子目录
        visualization_output = os.path.join(output_dir, "visualizations")
        print(f"可视化输出自动设置为: {visualization_output}")
    
    save_formats = args.save_formats or DEFAULT_SAVE_FORMATS
    generate_final_report = args.generate_final_report or DEFAULT_GENERATE_FINAL_REPORT
    
    try:
        # 创建模型训练器
        trainer = ModelTrainer(
            config_path=config_path,
            output_dir=output_dir
        )
        
        # 配置奖励函数参数 - 实验#003汇率专用配置
        trainer.config.reward.reward_type = reward_type
        if hasattr(trainer.config.reward, 'pip_size'):
            trainer.config.reward.pip_size = pip_size
        if hasattr(trainer.config.reward, 'daily_target_pips'):
            trainer.config.reward.daily_target_pips = daily_target_pips
        
        # 配置奖励函数（静默配置，仅记录异常）
        try:
            # 构建奖励函数参数 - 实验#003汇率专用配置 + 实验#005增强
            reward_kwargs = {
                'initial_balance': DEFAULT_INITIAL_BALANCE,  # 使用汇率优化的5万初始资金
                'pip_size': pip_size,                       # 汇率点大小
                'daily_target_pips': daily_target_pips,     # 日目标点数
                'trend_window': trend_window,               # 趋势判断窗口
                'quality_window': quality_window,           # 交易质量窗口
            }
            
            # Experiment #005 OptimizedForexReward 专用参数
            if reward_type in ['optimized_forex_reward', 'experiment_005', 'enhanced_forex', 'reward_return_consistent', 'stable_forex', 'correlation_fixed']:
                reward_kwargs.update({
                    'return_weight': exp005_return_weight,
                    'risk_penalty': exp005_risk_penalty,
                    'transaction_cost': exp005_transaction_cost,
                    'correlation_threshold': exp005_correlation_threshold,
                    'stability_window': exp005_stability_window,
                    'volatility_adjustment': exp005_volatility_adjustment,
                    'clip_range': tuple(exp005_clip_range),
                    'base_currency_pair': symbol  # 传递当前货币对
                })
            
            # 添加其他奖励函数特定参数
            if hasattr(args, 'eta') and args.eta is not None:
                reward_kwargs['eta'] = args.eta
            if hasattr(args, 'scale_factor') and args.scale_factor is not None:
                reward_kwargs['scale_factor'] = args.scale_factor
            if hasattr(args, 'return_weight') and args.return_weight is not None:
                reward_kwargs['return_weight'] = args.return_weight
            if hasattr(args, 'drawdown_weight') and args.drawdown_weight is not None:
                reward_kwargs['drawdown_weight'] = args.drawdown_weight
            
            # 创建奖励函数实例进行验证
            from src.rewards import create_reward_function
            reward_instance = create_reward_function(reward_type, **reward_kwargs)
            print(f"SUCCESS: 奖励函数配置成功: {reward_type}")
            
        except Exception as e:
            print(f"WARNING: 奖励函数配置验证失败: {e}")
            print("   将使用默认配置继续训练")
        
        # 自动查找数据目录 (如果未指定) - 使用统一的数据处理工具
        if not data_dir:
            data_processor = get_data_processor("AutoDataFinder")
            data_dir = data_processor.auto_find_data_directory(symbol)
            if not data_dir:
                print(f"未找到 {symbol} 的数据目录，请先运行 download_data.py 下载数据")
                return 1
            print(f"自动找到数据目录: {data_dir}")
        
        # 构建训练配置，包含用户指定的参数
        trainer_config = {
            'target_reward': target_reward,
            'total_timesteps': timesteps_total
        }
        
        # 传递其他训练参数到配置中 - 实验#003汇率专用配置
        custom_config = {
            'target_reward': 4.0,  # 汇率专用目标奖励
            'total_timesteps': 3000000,  # 300万步，充分学习汇率规律
            'iterations': iterations,
            'checkpoint_freq': checkpoint_freq,
            'reward_type': reward_type,
            'reward_kwargs': {
                'initial_balance': DEFAULT_INITIAL_BALANCE,     # 汇率优化的5万初始资金
                'pip_size': pip_size,                          # 汇率点大小
                'daily_target_pips': daily_target_pips,        # 日目标点数
                'trend_window': trend_window,                  # 趋势判断窗口
                'quality_window': quality_window,              # 交易质量窗口
                'max_risk_per_trade': 0.02,                   # 单笔最大风险2%
                'consistency_weight': 0.3,                    # 一致性权重
                # Experiment #005 参数
                'return_weight': exp005_return_weight,
                'risk_penalty': exp005_risk_penalty,
                'transaction_cost': exp005_transaction_cost,
                'correlation_threshold': exp005_correlation_threshold,
                'stability_window': exp005_stability_window,
                'volatility_adjustment': exp005_volatility_adjustment,
                'clip_range': tuple(exp005_clip_range)
            },
            # 添加汇率专用训练优化参数
            'learning_rate': DEFAULT_LEARNING_RATE,           # 5e-5 汇率优化学习率
            'clip_range': DEFAULT_CLIP_RANGE,                 # 0.08 保守裁剪范围
            'n_steps': DEFAULT_N_STEPS,                       # 2048 增大批次
            'batch_size': DEFAULT_BATCH_SIZE,                 # 128 适合汇率
            'n_epochs': DEFAULT_N_EPOCHS,                     # 6 轮训练
            'transaction_costs': DEFAULT_TRANSACTION_COSTS,   # 0.005% 汇率低成本
            'max_position_size': DEFAULT_MAX_POSITION_SIZE,   # 80% 最大仓位
            'stop_loss_threshold': DEFAULT_STOP_LOSS_THRESHOLD, # 5% 止损阈值
            # GPU多线程优化参数
            'n_envs': n_envs,                                 # 并行环境数量
            'num_threads': num_threads,                       # 数据加载线程数
            'device': device,                                 # 训练设备
            'tensorboard_log': tensorboard_log,               # TensorBoard日志
            # 添加可视化配置
            'enable_visualization': enable_visualization,     # 启用训练可视化
            'visualization_frequency': visualization_freq,    # 可视化生成频率
            'visualization_output_dir': visualization_output,  # 可视化输出目录
            'visualization_save_formats': save_formats,       # 保存格式
            'generate_final_report': generate_final_report,   # 生成最终报告
            'experiment_name': f"{symbol}_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # 实验名称
        }
        
        # 添加奖励函数特定参数到配置中
        if hasattr(args, 'eta') and args.eta is not None:
            custom_config['reward_kwargs']['eta'] = args.eta
        if hasattr(args, 'scale_factor') and args.scale_factor is not None:
            custom_config['reward_kwargs']['scale_factor'] = args.scale_factor
        if hasattr(args, 'return_weight') and args.return_weight is not None:
            custom_config['reward_kwargs']['return_weight'] = args.return_weight
        if hasattr(args, 'drawdown_weight') and args.drawdown_weight is not None:
            custom_config['reward_kwargs']['drawdown_weight'] = args.drawdown_weight
        if hasattr(args, 'base_window_size') and args.base_window_size is not None:
            custom_config['reward_kwargs']['base_window_size'] = args.base_window_size
        if hasattr(args, 'volatility_threshold') and args.volatility_threshold is not None:
            custom_config['reward_kwargs']['volatility_threshold'] = args.volatility_threshold
        
        if custom_config:
            trainer_config.update(custom_config)
        
        # Stable-Baselines3不需要单独的停止条件配置
        stop_config = None
        
        if use_gpu:
            print(f"GPU训练模式: 使用 {num_gpus} 个GPU ({device})")
            print(f"   - 并行环境数: {n_envs} (GPU加速)")
            print(f"   - 数据线程数: {num_threads}")
        else:
            print("CPU训练模式")
            print(f"   - 并行环境数: {n_envs}")
            print(f"   - 数据线程数: {num_threads}")
        
        print(f"\n训练框架: Stable-Baselines3")
        print(f"TensorBoard日志: {tensorboard_log}")
        
        # 可视化配置信息
        if enable_visualization:
            print(f"训练可视化: 启用")
            print(f"   - 生成频率: 每 {visualization_freq:,} 步")
            print(f"   - 输出目录: {visualization_output}")
            print(f"   - 保存格式: {', '.join(save_formats)}")
            print(f"   - 最终报告: {'启用' if generate_final_report else '禁用'}")
        else:
            print(f"训练可视化: 禁用 (可使用 --enable-visualization 启用)")
        
        # 开始训练
        result = trainer.train_model(
            symbol=symbol,
            data_dir=data_dir,
            iterations=iterations,
            checkpoint_freq=checkpoint_freq,
            hyperparameter_search=hyperparameter_search,
            resume_from=resume_from,
            custom_config=trainer_config if trainer_config else None,
            stop_config=None,  # Stable-Baselines3不需要Ray RLlib风格的停止条件
            feature_columns=feature_columns,
            n_envs=n_envs
        )
        
        # 输出结果
        if result['status'] == 'success':
            framework = result.get('framework', 'unknown')
            print(f"\n✅ {symbol} 模型训练成功! (框架: {framework})")
            print(f"模型保存至: {result['model_path']}")
            print(f"训练迭代: {result['iterations']}")
            
            if framework == 'stable-baselines3':
                total_timesteps = result.get('total_timesteps', 'unknown')
                print(f"总训练步数: {total_timesteps}")
            
            # 显示可视化信息
            if enable_visualization:
                training_result = result.get('training_result', {})
                if 'visualization_files' in training_result:
                    viz_files = training_result['visualization_files']
                    print(f"\n训练可视化:")
                    print(f"   生成图表: {len(viz_files)} 个")
                    print(f"   输出目录: {visualization_output}")
                    
                    if verbose:
                        print(f"   生成的图表类型:")
                        for chart_type in viz_files.keys():
                            print(f"     - {chart_type}")
                
                if 'final_report' in training_result:
                    print(f"   📄 最终训练报告: {training_result['final_report']}")
            
            if verbose and result.get('training_result'):
                print("\n训练结果详情:")
                print(json.dumps(result['training_result'], indent=2, ensure_ascii=False, default=str))
        else:
            print(f"\nFAILED: {symbol} 模型训练失败:")
            print(f"错误: {result.get('error', '未知错误')}")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        return 0
    
    except Exception as e:
        print(f"\n训练失败: {e}")
        return 1
    
    finally:
        # 清理资源
        try:
            trainer.cleanup()
        except:
            pass
    
    print(f"\n模型训练完成")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)