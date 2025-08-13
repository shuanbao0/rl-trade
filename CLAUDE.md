# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment
Use 'D:/ProgramData/anaconda3/envs/tensortrade_modern/python.exe' to run any Python files. For uncertain solutions, refer to requirement.txt dependencies and consult official documentation.

## User Context and Workflow Requirements
- Always review `@progress.md` to understand completed work and `@project-status.md` for session context
- Update progress logs in `@progress.md` after completing tasks
- Generate session reports in `@project-status.md` at session end
- Use Chinese for user communication while keeping technical docs in English

## Project Overview
This is a reinforcement learning trading system (强化学习交易系统) that has been migrated from Ray RLlib to **Stable-Baselines3** for quantitative trading. The system combines data management, feature engineering, and modern RL algorithms for automated stock trading decisions.

## Commands

### Dependencies and Setup
```bash
# Install dependencies (Python 3.11+ environment: tensortrade_modern)
pip install -r requirement.txt

# Train model using Stable-Baselines3
python train_model.py --symbol AAPL --iterations 150 --reward-type risk_adjusted

# Run main system with training mode
python main.py --mode train --symbol AAPL --period 2y --iterations 100

# Run backtest mode
python main.py --mode backtest --symbol AAPL --period 1y --model-path models/AAPL_model

# Run evaluation mode  
python main.py --mode evaluate --symbol AAPL --period 6m --model-path models/AAPL_model

# Download and prepare data
python download_data.py --symbol AAPL --period 2y

# Evaluate trained model
python evaluate_model.py --symbol AAPL --model-path experiments/AAPL_*/final_model.zip
```

### Testing Commands
```bash
# Run all tests with correct Python path
"D:/ProgramData/anaconda3/envs/tensortrade_modern/python.exe" -m pytest test/ -v --tb=short

# Run specific module tests
"D:/ProgramData/anaconda3/envs/tensortrade_modern/python.exe" -m pytest test/utils/ -v
"D:/ProgramData/anaconda3/envs/tensortrade_modern/python.exe" -m pytest test/environment/rewards/ -v
"D:/ProgramData/anaconda3/envs/tensortrade_modern/python.exe" -m pytest test/training/ -v

# Run tests with coverage
"D:/ProgramData/anaconda3/envs/tensortrade_modern/python.exe" -m pytest test/ --cov=src --cov-report=html
```

### Development Commands
```bash
# Check system logs
ls logs/
tail -f logs/main.log

# View cached data
ls data_cache/

# Check training results
ls experiments/
ls models/

# Clean up cache and logs
rm -rf data_cache/* logs/*
```

## Architecture

### Core Framework Migration
**IMPORTANT**: The system has been fully migrated from Ray RLlib to Stable-Baselines3:
- **Training Framework**: `src/training/stable_baselines_trainer.py` (formerly modern_trainer.py)
- **Algorithms**: PPO, SAC, DQN via Stable-Baselines3
- **Environment**: Custom TradingEnvironment with Gymnasium compatibility
- **Rewards**: 17+ advanced reward functions with modern RL integration

### Core Components

**Training System (`src/training/`)**
- `StableBaselinesTrainer`: Main training class using Stable-Baselines3
- `TrainingPipeline`: Complete training workflow management
- `HyperparameterOptimizer`: Optuna-based parameter optimization
- `model_trainer.py`: Compatibility wrapper

**Data Management (`src/data/`)**
- `DataManager`: Singleton pattern for stock data fetching via yfinance API
- Dual caching system (memory + file) with automatic expiry
- Data validation and quality checks
- Retry mechanisms with exponential backoff

**Feature Engineering (`src/features/`)**
- `FeatureEngineer`: Technical indicators and statistical features
- Indicators: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV
- Statistical features: returns, rolling statistics, multi-timeframe
- MinMaxScaler integration for normalization

**Trading Environment (`src/environment/`)**
- `TradingEnvironment`: Gymnasium-compatible RL environment
- 17+ reward functions in `rewards/` including advanced AI-based rewards
- Continuous action space [-1, 1] for position management
- Custom reward functions: risk_adjusted, curiosity_driven, self_rewarding, etc.

**Utilities (`src/utils/`)**
- `Config`: Environment variables and JSON config file support
- `Logger`: Multi-level logging with console and file output
- `data_utils`: Unified data processing utilities

### Advanced Reward Functions System
The system includes 17 reward function types with 150+ aliases:
- **Basic**: simple_return, risk_adjusted, profit_loss, diversified
- **Advanced**: log_sharpe, dynamic_sortino, regime_aware, expert_committee
- **AI-Enhanced**: uncertainty_aware, curiosity_driven, self_rewarding
- **Modern RL**: causal_reward, llm_guided, curriculum_reward
- **Distributed**: federated_reward, meta_learning_reward

### Module Integration Pattern
1. **Initialize Config**: `config = Config()`
2. **Create DataManager**: `data_manager = DataManager(config)`
3. **Get Data**: `data = data_manager.get_stock_data('AAPL', period='1y')`
4. **Feature Engineering**: `feature_engineer = FeatureEngineer(config)`
5. **Process Features**: `features = feature_engineer.prepare_features(data)`
6. **Create Trainer**: `trainer = StableBaselinesTrainer(config)`
7. **Train Model**: `result = trainer.train(features, algorithm='ppo')`

### Key Design Patterns
- **Modern RL Framework**: Stable-Baselines3 with Gymnasium environments
- **Advanced Rewards**: 17 reward functions with AI/ML enhancements
- **Configuration Management**: Centralized config with environment variable override
- **Caching Strategy**: Memory + file caching with TTL expiry
- **Error Handling**: Retry mechanisms with detailed logging
- **Modular Design**: Each component can be used independently

### Data Flow
```
Raw Data (yfinance) → DataManager → Feature Engineering → TradingEnvironment → StableBaselinesTrainer
                         ↓
                    Cache System
                         ↓
                    Validation & Logging
```

## Testing
The project uses pytest with comprehensive test coverage:
- Core modules: `test/utils/`, `test/environment/`, `test/training/`
- Reward functions: `test/environment/rewards/`
- Integration tests for complete training pipeline
- Use the specific Python environment path for all test commands

## Configuration
Three-tier configuration approach:
1. **Default values** in dataclass definitions
2. **JSON config files** loaded via `Config(config_file='path')`
3. **Environment variables** that override other settings

Key configuration sections:
- `data`: Cache settings, timeouts, retry logic
- `feature`: Technical indicator parameters (periods, thresholds)
- `trading`: Initial balance, commission, position limits
- `reward`: Reward function type and parameters

## Important Notes
- **Python Environment**: Uses tensortrade_modern conda environment with Python 3.11+
- **Framework**: Fully migrated to Stable-Baselines3 (Ray RLlib removed)
- **Dependencies**: Modern ML stack with PyTorch 2.5.1, Stable-Baselines3 2.7.0
- **Training**: Supports PPO, SAC, DQN algorithms
- **Rewards**: 17 advanced reward functions with AI enhancements
- **Logging**: Automatic log file creation in `logs/` directory
- **Caching**: Cache files in `data_cache/` with automatic cleanup
- **Models**: Saved in `experiments/` and `models/` directories

## Development Status
- ✅ Data Management Module (100% complete)
- ✅ Feature Engineering Module (100% complete)
- ✅ Trading Environment Module (100% complete)
- ✅ Advanced Reward System (17 functions, 100% complete)
- ✅ Stable-Baselines3 Training System (100% complete)
- ✅ Utils Module (100% complete)
- ✅ Testing Framework (comprehensive coverage)
- ✅ Migration from Ray RLlib to Stable-Baselines3 (100% complete)