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

# Docker deployment commands
docker-compose up -d  # Start all services
docker-compose logs -f tensortrade_app  # View application logs
docker-compose ps  # Check service status
docker-compose down  # Stop all services

# API service
python api.py  # Start FastAPI web service
uvicorn api:app --host 0.0.0.0 --port 8000  # Alternative start method
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
- `TradingEnvironment`: Gymnasium-compatible RL environment replacing TensorTrade
- Modern portfolio management with simplified state tracking
- 17+ reward functions in `rewards/` including advanced AI-based rewards
- Continuous action space [-1, 1] for position management
- Custom reward functions: risk_adjusted, curiosity_driven, self_rewarding, etc.
- Advanced reward components: RLHF (Human Feedback), multimodal analysis

**Real-time Trading (`src/realtime/`)**
- `real_time_trading_system.py`: Complete real-time trading system
- `real_time_data_manager.py`: WebSocket live data feeds
- `order_manager.py`: Trade execution and order management  
- `broker_api.py`: Broker integration interface
- `model_inference_service.py`: Real-time model prediction service

**Risk Management (`src/risk/`)**
- `risk_manager.py`: Multi-layer risk control system
- Position sizing, drawdown, and loss limits
- Real-time risk monitoring and alerts

**Monitoring & Visualization (`src/monitoring/`, `src/visualization/`)**
- `system_monitor.py`: System performance monitoring
- `alert_manager.py`: Alert and notification system
- `correlation_monitor.py`: Market correlation tracking
- Comprehensive visualization suite for training and evaluation results

**Validation (`src/validation/`)**
- `walk_forward_validator.py`: Time-series forward validation
- `time_series_validator.py`: Time series specific validation methods
- Prevents data leakage with strict temporal splitting

**Data Sources (`src/data/sources/`)**
- Multiple data source adapters: yfinance, FXMinute, OANDA, TrueFX
- `factory.py`: Data source factory pattern
- `converter.py`: Unified data format conversion
- Extensible architecture for adding new data providers

**Utilities (`src/utils/`)**
- `Config`: Three-tier configuration management (defaults → JSON → env vars)
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
Multiple Data Sources (yfinance, FXMinute, OANDA) → Data Source Factory → DataManager
                                                                              ↓
                                                        Cache System (Memory + File)
                                                                              ↓
                                              Feature Engineering → TradingEnvironment
                                                                              ↓
                                            StableBaselinesTrainer ← Reward Functions (17 types)
                                                                              ↓
                                              Model Evaluation ← Risk Management
                                                                              ↓
                                               Real-time Trading ← Monitoring & Alerts
```

## Testing
The project uses pytest with comprehensive test coverage across all modules:
- **Core utilities**: `test/utils/` (config, logging, data utilities)
- **Trading environment**: `test/environment/` (trading environment, rewards)
- **Training system**: `test/training/` (Stable-Baselines3 trainer, optimization)
- **Data management**: `test/data/` (data sources, caching, validation)
- **Real-time components**: `test/realtime/` (trading systems, order management)
- **Risk management**: `test/risk/` (risk controls, monitoring)
- **Visualization**: `test/visualization/` (plotting, report generation)
- **Feature engineering**: `test/features/` (technical indicators, transformations)
- **Validation**: `test/validation/` (walk-forward, time-series validation)
- **Monitoring**: `test/monitoring/` (system monitoring, alerts)

Use the specific Python environment path for all test commands to ensure compatibility.

## Configuration
Three-tier configuration approach:
1. **Default values** in dataclass definitions
2. **JSON config files** loaded via `Config(config_file='path')`
3. **Environment variables** that override other settings

Key configuration sections:
- `data`: Cache settings, timeouts, retry logic, data source selection
- `feature`: Technical indicator parameters (periods, thresholds), Forex-specific configs
- `trading`: Initial balance, commission, position limits, action space settings
- `reward`: Reward function type and parameters (17 different reward types supported)
- `risk`: Position sizing, drawdown limits, loss controls
- `realtime`: WebSocket settings, broker API configurations
- `monitoring`: Alert thresholds, system monitoring parameters

## Important Notes
- **Python Environment**: Uses tensortrade_modern conda environment with Python 3.11+
- **Framework**: Fully migrated to Stable-Baselines3 (Ray RLlib removed)
- **Dependencies**: Modern ML stack with PyTorch 2.5.1+cu121, Stable-Baselines3 2.7.0
- **GPU Support**: CUDA 12.1+ with automatic GPU detection and usage
- **Training**: Supports PPO, SAC, DQN algorithms with hyperparameter optimization
- **Rewards**: 17 advanced reward functions including RLHF and multimodal components
- **Data Sources**: Multiple providers (yfinance, FXMinute, OANDA, TrueFX)
- **Logging**: Automatic log file creation in `logs/` directory with component-specific logs
- **Caching**: Intelligent dual-layer caching (memory + file) in `data_cache/`
- **Models**: Saved in timestamped directories under `experiments/` and `models/`
- **Docker**: Full containerization support with Docker Compose
- **API**: FastAPI web service for programmatic access
- **Monitoring**: Integrated Prometheus + Grafana monitoring stack

## Development Status
- ✅ **Data Management Module** (100% complete) - Multi-source data fetching with intelligent caching
- ✅ **Feature Engineering Module** (100% complete) - 35+ technical indicators with Forex optimization  
- ✅ **Trading Environment Module** (100% complete) - Modern Gymnasium environment replacing TensorTrade
- ✅ **Advanced Reward System** (100% complete) - 17 reward functions including RLHF and multimodal
- ✅ **Stable-Baselines3 Training System** (100% complete) - PPO/SAC/DQN with optimization
- ✅ **Real-time Trading System** (100% complete) - WebSocket feeds and order management
- ✅ **Risk Management System** (100% complete) - Multi-layer risk controls and monitoring
- ✅ **Validation Framework** (100% complete) - Walk-forward and time-series validation
- ✅ **Visualization Suite** (100% complete) - Comprehensive plotting and analysis tools
- ✅ **Monitoring & Alerting** (100% complete) - System monitoring with Prometheus/Grafana
- ✅ **Utils & Configuration** (100% complete) - Advanced configuration management
- ✅ **Testing Framework** (100% complete) - Comprehensive test coverage across all modules
- ✅ **Docker & API Services** (100% complete) - Full containerization and FastAPI web service
- ✅ **Migration from Ray RLlib to Stable-Baselines3** (100% complete)

## Common Development Workflows

### Adding New Reward Functions
1. Create new reward class in `src/environment/rewards/`
2. Inherit from `BaseReward` and implement required methods
3. Register in `reward_factory.py` with aliases
4. Add tests in `test/environment/rewards/`
5. Update documentation in `docs/`

### Adding New Data Sources
1. Create source adapter in `src/data/sources/`
2. Inherit from `BaseDataSource` 
3. Implement data fetching and conversion methods
4. Register in `factory.py`
5. Add configuration options in `config.py`
6. Add comprehensive tests

### Training New Models
1. Ensure data is available: `python download_data.py --symbol SYMBOL --period PERIOD`
2. Configure reward function and hyperparameters
3. Run training: `python main.py --mode train --symbol SYMBOL --iterations N`
4. Monitor training logs in `logs/`
5. Evaluate results: `python main.py --mode evaluate --model-path models/LATEST`

### Debugging Training Issues
1. Check logs in `logs/` directory for component-specific errors
2. Verify data quality: `python -c "from src.data import DataManager; dm = DataManager(); data = dm.get_stock_data('AAPL', '1y'); print(data.describe())"`
3. Test environment: `python -c "from src.environment import TradingEnvironment; env = TradingEnvironment(); env.reset()"`
4. Run single test: `"D:/ProgramData/anaconda3/envs/tensortrade_modern/python.exe" -m pytest test/training/test_stable_baselines_trainer.py -v`

## Key Architecture Patterns

### Dependency Injection Pattern
- `Config` class provides centralized configuration
- All major components accept `Config` instance in constructor
- Allows easy testing and configuration switching

### Factory Pattern
- `DataSourceFactory` for data source creation
- `RewardFactory` for reward function instantiation
- Enables runtime configuration and extensibility

### Observer Pattern
- Training callbacks provide progress monitoring
- Risk manager observes portfolio state changes
- Alert system observes system metrics

### Singleton Pattern
- `DataManager` implements singleton for cache management
- Prevents duplicate data downloads and memory usage

### Strategy Pattern
- Reward functions implement common interface but different strategies
- Data sources follow common interface with different implementations
- Allows easy switching between algorithms and providers