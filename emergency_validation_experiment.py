#!/usr/bin/env python
"""
Emergency Validation Experiment
基于实验5发现的关键问题，快速验证奖励函数修复

Purpose: Verify that simple_return reward function provides proper reward-return correlation
Timeline: 30-60 minutes rapid validation
Target: Reward-return correlation > 0.5 to prove system works

Critical Issue from Experiment 5:
- Reward: +1152.41 (model thinks it's doing great)
- Return: -63.76% (actually losing money badly)  
- Correlation: ~0 (complete disconnect)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import TensorTrade components
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.data_manager import DataManager
from src.features.feature_engineer import FeatureEngineer
from src.training.stable_baselines_trainer import StableBaselinesTrainer
from src.environment.rewards import create_reward_function

def main():
    """Run emergency validation experiment"""
    
    # Setup logging
    logger = setup_logger(
        "EmergencyValidation", 
        log_file="logs/emergency_validation_experiment.log",
        console_output=True
    )
    
    logger.info("=" * 80)
    logger.info("EMERGENCY VALIDATION EXPERIMENT - START")
    logger.info("=" * 80)
    logger.info("Goal: Verify reward-return correlation with simple_return")
    logger.info("Target: Correlation > 0.5 to validate system functionality")
    
    try:
        # Configuration for minimal, fast experiment
        config = Config()
        config.data.symbol = "EURUSD=X"
        config.data.period = "6mo"  # Reduced dataset for speed
        config.training.total_timesteps = 500_000  # Quick validation
        config.training.algorithm = "ppo"
        config.training.learning_rate = 3e-4
        
        logger.info(f"Configuration:")
        logger.info(f"  Symbol: {config.data.symbol}")
        logger.info(f"  Period: {config.data.period}")
        logger.info(f"  Timesteps: {config.training.total_timesteps:,}")
        logger.info(f"  Algorithm: {config.training.algorithm}")
        
        # Step 1: Load minimal data
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Loading data for validation")
        
        data_manager = DataManager(config)
        raw_data = data_manager.get_stock_data(
            symbol=config.data.symbol,
            period=config.data.period,
            interval="1d"
        )
        
        logger.info(f"Data loaded: {len(raw_data)} rows")
        logger.info(f"Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
        
        # Step 2: Create minimal features (3 basic features)
        logger.info("\n" + "="*50)
        logger.info("STEP 2: Creating minimal feature set")
        
        feature_engineer = FeatureEngineer(config)
        
        # Override to use only 3 basic features to isolate the issue
        basic_features = {
            'close_price': raw_data['Close'].pct_change().fillna(0),  # Price returns
            'volume': raw_data['Volume'].pct_change().fillna(0),      # Volume change  
            'price_position': (raw_data['Close'] - raw_data['Close'].rolling(20).min()) / (raw_data['Close'].rolling(20).max() - raw_data['Close'].rolling(20).min())  # Price position in range
        }
        
        features_df = pd.DataFrame(basic_features, index=raw_data.index).fillna(0)
        logger.info(f"Features created: {len(features_df.columns)} basic features")
        logger.info(f"Feature names: {list(features_df.columns)}")
        
        # Step 3: Create simple_return reward function
        logger.info("\n" + "="*50)
        logger.info("STEP 3: Creating simple_return reward function")
        
        reward_function = create_reward_function(
            'simple_return',
            initial_balance=10000.0
        )
        reward_info = reward_function.get_reward_info()
        logger.info(f"Reward function: {reward_info['name']}")
        logger.info(f"Expected behavior: Direct correlation with portfolio returns")
        
        # Step 4: Setup and run training
        logger.info("\n" + "="*50)
        logger.info("STEP 4: Running emergency validation training")
        
        trainer = StableBaselinesTrainer(config)
        
        # Setup training with minimal configuration
        trainer.setup_training(
            df=features_df,
            algorithm=config.training.algorithm,
            reward_function=reward_function,
            model_kwargs={
                'learning_rate': config.training.learning_rate,
                'batch_size': 64,
                'n_steps': 1024,
                'verbose': 1
            }
        )
        
        start_time = datetime.now()
        logger.info(f"Training started at: {start_time}")
        logger.info(f"Training for {config.training.total_timesteps:,} timesteps")
        
        # Train the model
        result = trainer.train(
            total_timesteps=config.training.total_timesteps,
            eval_freq=50_000,  # Evaluate every 50K steps
            save_path=f"models/emergency_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        logger.info(f"Training completed at: {end_time}")
        logger.info(f"Training duration: {training_duration:.0f} seconds ({training_duration/60:.1f} minutes)")
        
        # Step 5: Evaluate and check correlation
        logger.info("\n" + "="*50)
        logger.info("STEP 5: Emergency evaluation and correlation check")
        
        # Quick evaluation with 5 episodes
        evaluation_results = trainer.evaluate_model(
            num_episodes=5,
            render=False
        )
        
        # Extract key metrics
        mean_reward = evaluation_results.get('mean_reward', 0)
        std_reward = evaluation_results.get('std_reward', 0)
        episode_returns = evaluation_results.get('episode_returns', [])
        episode_rewards = evaluation_results.get('episode_rewards', [])
        
        # Calculate correlation
        if len(episode_returns) > 1 and len(episode_rewards) > 1:
            correlation = np.corrcoef(episode_rewards, episode_returns)[0, 1]
            correlation = correlation if not np.isnan(correlation) else 0.0
        else:
            correlation = 0.0
            
        # Calculate mean return
        mean_return = np.mean(episode_returns) if episode_returns else -100.0
        
        logger.info("="*60)
        logger.info("EMERGENCY VALIDATION RESULTS")
        logger.info("="*60)
        logger.info(f"Mean Reward: {mean_reward:.2f} (±{std_reward:.2f})")
        logger.info(f"Mean Return: {mean_return:.2f}%")
        logger.info(f"Reward-Return Correlation: {correlation:.3f}")
        logger.info(f"Episodes evaluated: {len(episode_returns)}")
        
        # Critical assessment
        logger.info("\n" + "="*50)
        logger.info("CRITICAL ASSESSMENT")
        
        success_criteria = {
            'correlation_threshold': 0.5,
            'reasonable_reward_range': (-100, 100),
            'system_stability': True
        }
        
        # Check correlation success
        correlation_success = abs(correlation) >= success_criteria['correlation_threshold']
        logger.info(f"Reward-Return Correlation: {correlation:.3f} {'✅ PASS' if correlation_success else '❌ FAIL'} (target: ±{success_criteria['correlation_threshold']})")
        
        # Check reward range reasonableness
        reward_reasonable = success_criteria['reasonable_reward_range'][0] <= mean_reward <= success_criteria['reasonable_reward_range'][1]
        logger.info(f"Reward Range Check: {mean_reward:.2f} {'✅ PASS' if reward_reasonable else '❌ FAIL'} (target: {success_criteria['reasonable_reward_range']})")
        
        # Overall assessment
        overall_success = correlation_success and reward_reasonable
        
        logger.info("\n" + "="*60)
        if overall_success:
            logger.info("🎉 EMERGENCY VALIDATION SUCCESS!")
            logger.info("✅ System can work with proper reward function")
            logger.info("✅ simple_return provides meaningful correlation")
            logger.info("✅ Ready to proceed with systematic fixes")
            logger.info("\nRECOMMENDATIONS:")
            logger.info("1. Use simple_return for Phase 1 of Experiment 5")
            logger.info("2. Investigate progressive_features reward bug")
            logger.info("3. Proceed with gradual feature addition")
        else:
            logger.info("🚨 EMERGENCY VALIDATION FAILED!")
            logger.info("❌ System issues persist even with simple reward")
            logger.info("❌ Deeper architectural problems likely")
            logger.info("\nRECOMMENDATIONS:")
            logger.info("1. Full system architecture review required")
            logger.info("2. Check trading environment implementation")
            logger.info("3. Verify data processing pipeline")
            
        logger.info("="*60)
        
        return {
            'success': overall_success,
            'correlation': correlation,
            'mean_reward': mean_reward,
            'mean_return': mean_return,
            'training_duration': training_duration,
            'evaluation_episodes': len(episode_returns)
        }
        
    except Exception as e:
        logger.error(f"Emergency validation failed with error: {e}")
        logger.error("Exception details:", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }
    
    finally:
        logger.info("=" * 80)
        logger.info("EMERGENCY VALIDATION EXPERIMENT - END")
        logger.info("=" * 80)

if __name__ == "__main__":
    print("Starting Emergency Validation Experiment...")
    print("This will quickly test if simple_return reward works properly.")
    print("Expected duration: 30-60 minutes")
    print()
    
    result = main()
    
    print("\nEmergency Validation Complete!")
    if result.get('success'):
        print("SUCCESS: System validation passed with simple_return reward")
        print(f"Correlation: {result.get('correlation', 0):.3f}")
    else:
        print("FAILED: System validation failed")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print("\nCheck logs/emergency_validation_experiment.log for detailed results")