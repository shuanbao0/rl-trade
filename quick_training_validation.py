#!/usr/bin/env python3
"""
Quick Training Validation for Experiment 006
实验006快速训练验证

Purpose: 验证DirectPnLReward奖励函数在实际训练中的表现
使用较短的训练时间来快速验证系统是否正常工作
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.environment.rewards import create_reward_function
from src.environment.trading_environment import TradingEnvironment  
from src.data.data_manager import DataManager
from src.utils.config import Config
from src.utils.logger import setup_logger

class QuickTrainingValidator:
    """快速训练验证器"""
    
    def __init__(self):
        self.logger = setup_logger("QuickTrainingValidator")
        self.config = Config()
        self.results = {}
        
    def run_quick_validation(self) -> dict:
        """运行快速训练验证"""
        
        self.logger.info("=== 开始实验006快速训练验证 ===")
        
        try:
            # 1. 准备数据
            self.logger.info("1. 准备EURUSD数据...")
            dm = DataManager()
            raw_data = dm.get_stock_data('EURUSD=X', period='6mo')
            
            if len(raw_data) < 50:
                raise ValueError(f"数据量不足: {len(raw_data)}")
                
            self.logger.info(f"数据获取成功: {len(raw_data)}条记录")
            
            # 2. 特征工程 (简化版)
            self.logger.info("2. 创建特征数据...")
            features = raw_data[['open', 'high', 'low', 'close', 'volume']].copy()
            features = features.dropna()
            
            # 添加简单的技术指标
            features['sma_5'] = features['close'].rolling(5).mean()
            features['rsi'] = self._calculate_rsi(features['close'], 14)
            features['volatility'] = features['close'].rolling(10).std()
            
            # 移除NaN值
            features = features.dropna()
            self.logger.info(f"特征数据形状: {features.shape}")
            
            # 3. 数据分割
            train_size = int(len(features) * 0.8)
            train_data = features[:train_size]
            test_data = features[train_size:]
            
            self.logger.info(f"训练数据: {len(train_data)}条, 测试数据: {len(test_data)}条")
            
            # 4. 创建DirectPnL奖励函数
            self.logger.info("3. 创建DirectPnL奖励函数...")
            reward_fn = create_reward_function('direct_pnl_reward', initial_balance=10000)
            
            # 5. 创建训练环境
            self.logger.info("4. 创建训练环境...")
            train_env = TradingEnvironment(
                df=train_data,
                config=self.config,
                initial_balance=10000,
                window_size=5,
                reward_function=reward_fn,
                max_episode_steps=50  # 限制episode长度
            )
            
            # 创建验证环境
            test_env = TradingEnvironment(
                df=test_data,
                config=self.config,
                initial_balance=10000,
                window_size=5,
                reward_function=reward_fn,
                max_episode_steps=len(test_data)-10
            )
            
            # 包装环境
            train_env_vec = DummyVecEnv([lambda: train_env])
            test_env_vec = DummyVecEnv([lambda: test_env])
            
            self.logger.info("5. 开始PPO训练...")
            
            # 6. 创建PPO模型 - 使用较简单的配置
            model = PPO(
                "MlpPolicy",
                train_env_vec,
                verbose=1,
                learning_rate=0.001,
                n_steps=512,
                batch_size=64,
                n_epochs=5,
                gamma=0.99,
                device='cpu',  # 强制使用CPU避免CUDA问题
                tensorboard_log="./logs/tensorboard/"
            )
            
            # 7. 训练模型 (短时间)
            self.logger.info("开始训练 - 20K步快速验证...")
            
            start_time = datetime.now()
            model.learn(
                total_timesteps=20_000,  # 短时间训练
                callback=None,
                tb_log_name="quick_validation"
            )
            training_time = datetime.now() - start_time
            
            self.logger.info(f"训练完成，耗时: {training_time}")
            
            # 8. 评估模型
            self.logger.info("6. 评估训练结果...")
            eval_results = self._evaluate_model(model, test_env_vec, n_episodes=5)
            
            # 9. 验证奖励-回报相关性
            correlation_results = self._validate_reward_correlation(reward_fn)
            
            # 10. 汇总结果
            self.results = {
                "training_success": True,
                "training_time": str(training_time),
                "training_steps": 20_000,
                "data_size": len(features),
                "train_episodes": len(train_data) - 10,
                "evaluation_results": eval_results,
                "reward_correlation": correlation_results,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("=== 快速训练验证完成 ===")
            self._print_results()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"快速训练验证失败: {e}")
            import traceback
            traceback.print_exc()
            return {"training_success": False, "error": str(e)}
    
    def _calculate_rsi(self, prices, window=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _evaluate_model(self, model, test_env, n_episodes=5):
        """评估训练好的模型"""
        
        self.logger.info(f"在测试环境中评估模型 ({n_episodes}轮)...")
        
        episode_returns = []
        episode_lengths = []
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs = test_env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                if episode_length > 100:  # 防止无限循环
                    break
            
            # 获取最终回报
            final_return = info[0].get('total_return', 0.0)
            
            episode_returns.append(final_return)
            episode_lengths.append(episode_length)
            episode_rewards.append(episode_reward)
            
            self.logger.info(f"Episode {episode+1}: 长度={episode_length}, 奖励={episode_reward:.4f}, 回报={final_return:.2f}%")
        
        return {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_reward": np.mean(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "episodes_evaluated": n_episodes
        }
    
    def _validate_reward_correlation(self, reward_fn):
        """验证奖励函数的相关性"""
        
        # 模拟一些交易数据来验证相关性
        portfolio_values = [10000 + i*50 + np.random.normal(0, 20) for i in range(20)]
        actions = [np.random.uniform(-0.5, 0.5) for _ in range(20)]
        prices = [1.09 + i*0.001 + np.random.normal(0, 0.002) for i in range(20)]
        
        rewards = []
        returns = []
        
        for i in range(1, len(portfolio_values)):
            portfolio_info = {
                'total_value': portfolio_values[i],
                'cash': portfolio_values[i] * 0.5,
                'shares': portfolio_values[i] * 0.5 / prices[i],
                'return_pct': (portfolio_values[i] - 10000) / 10000 * 100
            }
            
            trade_info = {'executed': True, 'amount': 0, 'cost': 0}
            
            reward = reward_fn.calculate_reward(
                portfolio_value=portfolio_values[i],
                action=actions[i],
                price=prices[i],
                portfolio_info=portfolio_info,
                trade_info=trade_info,
                step=i
            )
            
            actual_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] * 100
            
            rewards.append(reward)
            returns.append(actual_return)
        
        # 计算相关性
        correlation = np.corrcoef(rewards, returns)[0, 1] if len(rewards) > 1 else 0.0
        
        return {
            "correlation": correlation,
            "correlation_target": 0.8,
            "correlation_status": "GOOD" if correlation >= 0.8 else "NEEDS_IMPROVEMENT",
            "sample_size": len(rewards)
        }
    
    def _print_results(self):
        """打印验证结果摘要"""
        
        print("\n" + "="*80)
        print("实验006快速训练验证结果")
        print("="*80)
        
        if self.results.get("training_success"):
            print("✅ 训练状态: 成功")
            print(f"📊 训练时间: {self.results['training_time']}")
            print(f"🎯 训练步数: {self.results['training_steps']:,}")
            print(f"📈 数据规模: {self.results['data_size']} 条记录")
            
            eval_results = self.results.get("evaluation_results", {})
            print(f"\n📊 评估结果:")
            print(f"   平均回报: {eval_results.get('mean_return', 0):.2f}%")
            print(f"   平均奖励: {eval_results.get('mean_reward', 0):.4f}")
            print(f"   平均Episode长度: {eval_results.get('mean_length', 0):.1f}")
            
            corr_results = self.results.get("reward_correlation", {})
            print(f"\n🎯 奖励函数验证:")
            print(f"   奖励-回报相关性: {corr_results.get('correlation', 0):.4f}")
            print(f"   相关性状态: {corr_results.get('correlation_status', 'UNKNOWN')}")
            
            if corr_results.get('correlation', 0) >= 0.8:
                print("✅ DirectPnLReward奖励函数验证通过")
            else:
                print("⚠️ DirectPnLReward奖励函数需要进一步调试")
                
        else:
            print("❌ 训练状态: 失败")
            print(f"错误: {self.results.get('error', 'Unknown error')}")
        
        print("="*80)


def main():
    """主执行函数"""
    
    validator = QuickTrainingValidator()
    results = validator.run_quick_validation()
    
    # 保存结果
    results_file = f"quick_training_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")
    
    return results


if __name__ == "__main__":
    main()