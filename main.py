#!/usr/bin/env python
"""
强化学习交易系统主程序
基于 Stable-Baselines3 的智能交易系统

主要功能:
1. 训练模式 - 使用Stable-Baselines3训练模型
2. 评估模式 - 评估训练好的模型性能
3. 回测模式 - 历史数据回测  
4. 实时交易模式 - 实时数据交易

使用示例:
  # 训练模式
  python main.py --mode train --symbol AAPL --period 2y --iterations 100

  # 评估模式
  python main.py --mode evaluate --symbol AAPL --period 6m --model-path models/best_model

  # 回测模式
  python main.py --mode backtest --symbol AAPL --period 1y --model-path models/best_model

  # 实时交易模式
  python main.py --mode live --symbol AAPL --model-path models/best_model
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    # 导入项目模块
    from src.data.data_manager import DataManager
    from src.features.feature_engineer import FeatureEngineer
    from src.utils.config import Config
    from src.utils.logger import setup_logger, get_default_log_file
    
    # 导入训练组件
    from src.training import ModernTrainer, TrainingPipeline, HyperparameterOptimizer
    
    # 实时交易模块(可选)
    try:
        from src.realtime.real_time_trading_system import RealTimeTradingSystem
        REALTIME_AVAILABLE = True
    except ImportError:
        REALTIME_AVAILABLE = False
    
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)


class TensorTradeSystem:
    """
    强化学习交易系统主控制器
    
    基于Stable-Baselines3 RL框架
    提供统一的接口和工作流程管理
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化交易系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = Config(config_file=config_path) if config_path else Config()
        
        # 确保必要目录存在
        self._create_directories()
        
        # 初始化日志系统
        self.logger = setup_logger(
            name="TensorTradeSystem",
            level="INFO",
            log_file=get_default_log_file("main")
        )
        
        # 初始化核心组件
        self.data_manager = None
        self.feature_engineer = None
        self.realtime_system = None
        
        # 初始化训练组件
        self.sb3_trainer = None
        self.training_pipeline = None
        self.hyperparameter_optimizer = None
        
        # 系统状态
        self.is_initialized = False
        
        self.logger.info("交易系统初始化开始")
        
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            "models",
            "logs", 
            "data_cache",
            "results",
            "reports",
            "configs"
        ]
        
        for directory in directories:
            dir_path = PROJECT_ROOT / directory
            dir_path.mkdir(exist_ok=True)
    
    def initialize_components(self):
        """初始化所有系统组件"""
        try:
            self.logger.info("开始初始化系统组件...")
            
            # 1. 数据管理器
            self.data_manager = DataManager(self.config)
            self.logger.info("✓ 数据管理器初始化完成")
            
            # 2. 特征工程器
            self.feature_engineer = FeatureEngineer(self.config)
            self.logger.info("✓ 特征工程器初始化完成")
            
            # 3. 训练组件
            self.sb3_trainer = ModernTrainer(self.config)
            self.training_pipeline = TrainingPipeline(self.config)
            self.hyperparameter_optimizer = HyperparameterOptimizer(self.config)
            self.logger.info("✓ 训练组件初始化完成")
            
            # 4. 实时交易系统(可选)
            if REALTIME_AVAILABLE:
                self.realtime_system = RealTimeTradingSystem(self.config)
                self.logger.info("✓ 实时交易系统初始化完成")
            
            self.is_initialized = True
            self.logger.info("系统组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def train_mode(
        self,
        symbol: str,
        period: str = "2y",
        iterations: int = 100,
        save_path: str = "models"
    ) -> Dict[str, Any]:
        """
        训练模式 - 使用Stable-Baselines3训练智能体
        
        Args:
            symbol: 股票代码
            period: 数据周期
            iterations: 训练迭代次数
            save_path: 模型保存路径
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        self.logger.info(f"开始训练: {symbol}, 周期: {period}, 迭代: {iterations}")
        
        try:
            # 1. 获取数据
            self.logger.info("获取和处理数据...")
            raw_data = self.data_manager.get_stock_data(symbol, period=period)
            features_data = self.feature_engineer.prepare_features(raw_data)
            
            self.logger.info(f"数据准备完成: {len(features_data)} 条记录, {len(features_data.columns)} 个特征")
            
            # 2. 创建训练流水线
            experiment_name = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.training_pipeline = TrainingPipeline(
                config=self.config,
                experiment_name=experiment_name,
                base_path=save_path
            )
            
            # 3. 计算训练步数
            total_timesteps = max(iterations * 2000, 100000)
            
            # 4. 执行完整训练流水线
            self.logger.info(f"开始训练流水线: {total_timesteps} 步")
            
            training_result = self.training_pipeline.run_complete_pipeline(
                symbol=symbol,
                period=period,
                algorithm='ppo',  # 默认使用PPO
                reward_type='risk_adjusted',
                optimization_trials=20,
                final_timesteps=total_timesteps
            )
            
            # 5. 获取模型路径
            model_path = self.training_pipeline.experiment_path
            
            self.logger.info(f"训练完成，模型保存至: {model_path}")
            
            return {
                'status': 'success',
                'model_path': str(model_path),
                'training_result': training_result,
                'symbol': symbol,
                'period': period,
                'iterations': iterations,
                'total_timesteps': total_timesteps,
                'framework': 'stable-baselines3'
            }
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol
            }
    
    
    def backtest_mode(
        self,
        symbol: str,
        period: str = "1y",
        model_path: str = None
    ) -> Dict[str, Any]:
        """
        回测模式 - 使用历史数据回测模型
        
        Args:
            symbol: 股票代码
            period: 回测周期
            model_path: 模型路径
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        self.logger.info(f"开始回测模式: {symbol}, 周期: {period}")
        
        try:
            # 1. 检查模型路径
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"无效的模型路径: {model_path}")
            
            # 2. 获取回测数据
            raw_data = self.data_manager.get_stock_data(symbol, period=period)
            features_data = self.feature_engineer.prepare_features(raw_data)
            
            self.logger.info(f"回测数据准备完成: {len(features_data)} 条记录")
            
            # 3. 加载模型
            agent = self.trading_agent.load_model(model_path)
            
            # 4. 创建交易环境
            self.trading_environment.create_environment(features_data)
            
            # 5. 执行回测
            backtest_results = []
            
            # 分段回测以获得更详细的结果
            segment_size = len(features_data) // 4  # 分成4段
            
            for i in range(4):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, len(features_data))
                segment_data = features_data.iloc[start_idx:end_idx]
                
                if len(segment_data) < 50:  # 跳过太短的段
                    continue
                
                self.trading_environment.create_environment(segment_data)
                
                # 运行回测
                segment_result = self._run_backtest_segment(agent, segment_data, i)
                backtest_results.append(segment_result)
            
            # 6. 汇总结果
            overall_result = self._aggregate_backtest_results(backtest_results)
            
            # 7. 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = PROJECT_ROOT / "results" / f"backtest_{symbol}_{timestamp}.json"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(overall_result, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"回测完成，结果保存至: {results_path}")
            
            return {
                'status': 'success',
                'backtest_result': overall_result,
                'symbol': symbol,
                'period': period,
                'model_path': model_path
            }
            
        except Exception as e:
            self.logger.error(f"回测模式失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol
            }
    
    def _run_backtest_segment(self, agent, data, segment_id):
        """运行单个回测段"""
        from src.environment.trading_environment import TradingEnvWrapper
        
        # 创建环境包装器
        env_config = {
            'features_data': data,
            'config': self.config,
            'initial_balance': self.config.trading.initial_balance,
            'window_size': 50,
            'max_allowed_loss': 0.3
        }
        
        env = TradingEnvWrapper(env_config)
        
        # 运行回测
        obs = env.reset()
        episode_rewards = []
        episode_actions = []
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < len(data) - 50:
            action = agent.compute_single_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action[0] if isinstance(action, (list, tuple)) else action)
            total_reward += reward
            step_count += 1
        
        # 获取性能指标
        performance_metrics = env.get_performance_metrics()
        
        env.close()
        
        return {
            'segment_id': segment_id,
            'total_reward': total_reward,
            'episode_rewards': episode_rewards,
            'episode_actions': episode_actions,
            'performance_metrics': performance_metrics,
            'steps': step_count
        }
    
    def _aggregate_backtest_results(self, results):
        """聚合回测结果"""
        if not results:
            return {'error': 'No backtest results'}
        
        # 合并所有奖励
        all_rewards = []
        all_actions = []
        all_metrics = []
        
        for result in results:
            all_rewards.extend(result['episode_rewards'])
            all_actions.extend(result['episode_actions'])
            all_metrics.append(result['performance_metrics'])
        
        # 计算综合指标
        import numpy as np
        
        aggregate_result = {
            'segments_count': len(results),
            'total_steps': sum(r['steps'] for r in results),
            'total_reward': sum(r['total_reward'] for r in results),
            'mean_reward': np.mean(all_rewards) if all_rewards else 0,
            'std_reward': np.std(all_rewards) if all_rewards else 0,
            'min_reward': np.min(all_rewards) if all_rewards else 0,
            'max_reward': np.max(all_rewards) if all_rewards else 0,
            'win_rate': np.mean(np.array(all_rewards) > 0) if all_rewards else 0,
            'segment_results': results
        }
        
        # 平均性能指标
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics if isinstance(m.get(key), (int, float))]
                if values:
                    avg_metrics[f'avg_{key}'] = np.mean(values)
            
            aggregate_result['average_performance_metrics'] = avg_metrics
        
        return aggregate_result
    
    def evaluate_mode(
        self,
        symbol: str,
        period: str = "6m",
        model_path: str = None
    ) -> Dict[str, Any]:
        """
        评估模式 - 评估Stable-Baselines3模型性能
        
        Args:
            symbol: 股票代码
            period: 评估周期
            model_path: 模型路径
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"开始评估: {symbol}, 周期: {period}")
        
        try:
            # 获取数据
            raw_data = self.data_manager.get_stock_data(symbol, period=period)
            features_data = self.feature_engineer.prepare_features(raw_data)
            
            # 使用训练器进行评估
            evaluation_result = self.sb3_trainer.evaluate(
                model_path=model_path,
                test_df=features_data,
                n_episodes=30,
                render=False
            )
            
            # 保存评估结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = PROJECT_ROOT / "results" / f"evaluation_{symbol}_{timestamp}.json"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                'status': 'success',
                'evaluation_result': evaluation_result,
                'symbol': symbol,
                'period': period,
                'model_path': model_path,
                'framework': 'stable-baselines3'
            }
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol
            }
    
    def live_mode(
        self,
        symbol: str,
        model_path: str = None,
        duration_hours: int = 8
    ) -> Dict[str, Any]:
        """
        实时交易模式
        
        Args:
            symbol: 股票代码
            model_path: 模型路径
            duration_hours: 运行时长(小时)
            
        Returns:
            Dict[str, Any]: 实时交易结果
        """
        if not REALTIME_AVAILABLE:
            return {
                'status': 'error',
                'error': '实时交易模块不可用'
            }
        
        self.logger.info(f"开始实时交易模式: {symbol}, 运行时长: {duration_hours}小时")
        
        try:
            # 初始化实时交易系统
            if not self.realtime_system:
                self.realtime_system = RealTimeTradingSystem(self.config)
            
            # 设置模型路径
            if model_path:
                self.realtime_system.model_inference_service.load_model(model_path)
            
            # 启动实时交易
            result = self.realtime_system.start_trading(
                symbols=[symbol],
                duration_hours=duration_hours
            )
            
            return {
                'status': 'success',
                'trading_result': result,
                'symbol': symbol,
                'duration_hours': duration_hours
            }
            
        except Exception as e:
            self.logger.error(f"实时交易模式失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol
            }
    
    def cleanup(self):
        """清理系统资源"""
        try:
            if self.training_pipeline:
                if hasattr(self.training_pipeline, 'cleanup'):
                    self.training_pipeline.cleanup()
            
            if self.sb3_trainer:
                if hasattr(self.sb3_trainer, 'cleanup'):
                    self.sb3_trainer.cleanup()
            
            if self.realtime_system:
                self.realtime_system.stop_trading()
            
            self.logger.info("系统资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TensorTrade强化学习交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基础参数
    parser.add_argument('--mode', '-m', 
                       choices=['train', 'backtest', 'evaluate', 'live'],
                       required=True,
                       help='运行模式 (train, backtest, evaluate, live)')
    
    parser.add_argument('--symbol', '-s',
                       required=True,
                       help='股票代码 (例如: AAPL, GOOGL)')
    
    parser.add_argument('--period', '-p',
                       default='1y',
                       help='数据周期 (例如: 1y, 2y, 6m, 3m)')
    
    parser.add_argument('--config', '-c',
                       help='配置文件路径')
    
    # 训练模式参数
    parser.add_argument('--iterations', '-i',
                       type=int,
                       default=100,
                       help='训练迭代次数 (训练模式)')
    
    parser.add_argument('--save-path',
                       default='models',
                       help='模型保存路径 (训练模式)')
    
    # 模型路径参数
    parser.add_argument('--model-path',
                       help='模型路径 (回测/评估/实时交易模式)')
    
    # 使用内置交叉验证
    
    # 实时交易参数
    parser.add_argument('--duration',
                       type=int,
                       default=8,
                       help='实时交易运行时长(小时)')
    
    # 其他参数
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='详细输出')
    
    parser.add_argument('--save-results',
                       action='store_true',
                       default=True,
                       help='保存结果到文件')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("=" * 60)
    print("强化学习交易系统")
    print("基于 Stable-Baselines3 的智能交易平台")
    print("=" * 60)
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建系统实例
    try:
        system = TensorTradeSystem(config_path=args.config)
        system.initialize_components()
        
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return 1
    
    # 根据模式执行相应操作
    try:
        result = None
        
        if args.mode == 'train':
            print(f"\n🚀 使用Stable-Baselines3训练...")
            result = system.train_mode(
                symbol=args.symbol,
                period=args.period,
                iterations=args.iterations,
                save_path=args.save_path
            )
            
        # 使用内置评估
            
        elif args.mode == 'backtest':
            if not args.model_path:
                print("回测模式需要指定 --model-path 参数")
                return 1
            
            result = system.backtest_mode(
                symbol=args.symbol,
                period=args.period,
                model_path=args.model_path
            )
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                print("评估模式需要指定 --model-path 参数")
                return 1
            
            result = system.evaluate_mode(
                symbol=args.symbol,
                period=args.period,
                model_path=args.model_path
            )
            
        elif args.mode == 'live':
            if not args.model_path:
                print("实时交易模式需要指定 --model-path 参数")
                return 1
            
            result = system.live_mode(
                symbol=args.symbol,
                model_path=args.model_path,
                duration_hours=args.duration
            )
        
        # 输出结果
        if result:
            if result['status'] == 'success':
                print(f"\n✓ {args.mode} 模式执行成功!")
                if args.verbose:
                    print("\n结果详情:")
                    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            else:
                print(f"\n✗ {args.mode} 模式执行失败:")
                print(f"错误: {result.get('error', '未知错误')}")
                return 1
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        return 0
        
    except Exception as e:
        print(f"\n执行失败: {e}")
        return 1
    
    finally:
        # 清理资源
        try:
            system.cleanup()
        except:
            pass
    
    print(f"\n{args.mode} 模式执行完成")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)