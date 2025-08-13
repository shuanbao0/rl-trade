"""
测试实时交易系统模块
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from src.realtime.real_time_trading_system import (
    RealTimeTradingSystem, TradingMode, TradingSignal, SystemState, PerformanceMetrics
)
from src.utils.config import Config


class TestTradingSignal:
    def test_trading_signal_creation(self):
        """测试交易信号创建"""
        signal = TradingSignal(
            symbol="AAPL",
            action=0.5,
            confidence=0.8,
            target_position=0.4,
            current_position=0.2,
            timestamp=datetime.now(),
            model_version="v1.0"
        )
        
        assert signal.symbol == "AAPL"
        assert signal.action == 0.5
        assert signal.confidence == 0.8
        assert signal.target_position == 0.4
        assert signal.current_position == 0.2
        assert signal.model_version == "v1.0"


class TestSystemState:
    def test_system_state_creation(self):
        """测试系统状态创建"""
        state = SystemState()
        
        assert state.is_running is False
        assert state.is_trading_active is False
        assert state.total_trades_today == 0
        assert state.current_pnl == 0.0
        assert state.daily_pnl == 0.0
    
    def test_system_state_with_values(self):
        """测试带值的系统状态创建"""
        state = SystemState(
            is_running=True,
            is_trading_active=True,
            data_connection_ok=True,
            total_trades_today=5,
            current_pnl=1500.0
        )
        
        assert state.is_running is True
        assert state.is_trading_active is True
        assert state.data_connection_ok is True
        assert state.total_trades_today == 5
        assert state.current_pnl == 1500.0


class TestPerformanceMetrics:
    def test_performance_metrics_creation(self):
        """测试性能指标创建"""
        metrics = PerformanceMetrics()
        
        assert metrics.data_latency_ms == 0.0
        assert metrics.inference_latency_ms == 0.0
        assert metrics.order_execution_latency_ms == 0.0
        assert metrics.signals_processed == 0
        assert metrics.orders_executed == 0
        assert metrics.orders_failed == 0
    
    def test_performance_metrics_with_values(self):
        """测试带值的性能指标创建"""
        metrics = PerformanceMetrics(
            data_latency_ms=50.5,
            inference_latency_ms=25.3,
            signals_processed=100,
            orders_executed=45
        )
        
        assert metrics.data_latency_ms == 50.5
        assert metrics.inference_latency_ms == 25.3
        assert metrics.signals_processed == 100
        assert metrics.orders_executed == 45


class TestRealTimeTradingSystem:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.trading_system = RealTimeTradingSystem(
            config=self.config,
            trading_mode=TradingMode.SIMULATION,
            initial_capital=100000.0
        )
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        if hasattr(self.trading_system, 'cleanup'):
            self.trading_system.cleanup()
    
    def test_trading_system_initialization(self):
        """测试交易系统初始化"""
        assert self.trading_system.config is not None
        assert self.trading_system.trading_mode == TradingMode.SIMULATION
        assert self.trading_system.initial_capital == 100000.0
        assert isinstance(self.trading_system.system_state, SystemState)
        assert isinstance(self.trading_system.performance_metrics, PerformanceMetrics)
        assert len(self.trading_system.current_positions) == 0
        assert len(self.trading_system.pending_orders) == 0
        assert len(self.trading_system.trade_history) == 0
    
    def test_trading_mode_enum(self):
        """测试交易模式枚举"""
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.SIMULATION.value == "simulation"
        assert TradingMode.PAPER.value == "paper"
    
    @patch('src.realtime.real_time_data_manager.RealTimeDataManager')
    @patch('src.realtime.model_inference_service.ModelInferenceService')
    @patch('src.risk.risk_manager.RiskManager')
    @patch('src.realtime.order_manager.OrderManager')
    @patch('src.realtime.broker_api.SimulatedBroker')
    def test_initialize_components(self, mock_broker, mock_order_manager, 
                                 mock_risk_manager, mock_model_service, mock_data_manager):
        """测试组件初始化"""
        # 设置模拟对象
        mock_data_manager.return_value = MagicMock()
        mock_model_service.return_value = MagicMock()
        mock_risk_manager.return_value = MagicMock()
        mock_order_manager.return_value = MagicMock()
        mock_broker.return_value = MagicMock()
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        model_path = "test_model.pkl"
        
        self.trading_system.initialize_components(symbols, model_path)
        
        # 验证组件已初始化
        assert self.trading_system.data_manager is not None
        assert self.trading_system.model_service is not None
        assert self.trading_system.risk_manager is not None
        assert self.trading_system.order_manager is not None
        assert self.trading_system.broker_api is not None
        
        # 验证持仓初始化
        for symbol in symbols:
            assert symbol in self.trading_system.current_positions
            assert self.trading_system.current_positions[symbol] == 0.0
    
    def test_system_state_management(self):
        """测试系统状态管理"""
        # 初始状态
        assert self.trading_system.system_state.is_running is False
        assert self.trading_system.system_state.is_trading_active is False
        
        # 启动交易
        with patch.object(self.trading_system, '_verify_system_ready', return_value=True):
            self.trading_system.start_trading()
            assert self.trading_system.system_state.is_trading_active is True
        
        # 停止交易
        self.trading_system.stop_trading()
        assert self.trading_system.system_state.is_trading_active is False
    
    def test_callback_registration(self):
        """测试回调函数注册"""
        signal_callback = MagicMock()
        trade_callback = MagicMock()
        error_callback = MagicMock()
        
        self.trading_system.add_signal_callback(signal_callback)
        self.trading_system.add_trade_callback(trade_callback)
        self.trading_system.add_error_callback(error_callback)
        
        assert signal_callback in self.trading_system.signal_callbacks
        assert trade_callback in self.trading_system.trade_callbacks
        assert error_callback in self.trading_system.error_callbacks
    
    def test_create_trading_signal(self):
        """测试交易信号创建"""
        prediction = {
            'action': 0.3,
            'confidence': 0.7,
            'model_version': 'test_v1'
        }
        
        from src.realtime.real_time_data_manager import MarketData
        market_data = MarketData(
            symbol="AAPL",
            price=150.0,
            volume=1000,
            timestamp=datetime.now(),
            bid=149.5,
            ask=150.5
        )
        
        # 设置当前仓位
        self.trading_system.current_positions["AAPL"] = 0.1
        
        signal = self.trading_system._create_trading_signal(prediction, market_data)
        
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.action == 0.3
        assert signal.confidence == 0.7
        assert signal.current_position == 0.1
    
    def test_weak_signal_filtering(self):
        """测试弱信号过滤"""
        prediction = {
            'action': 0.05,  # 弱信号
            'confidence': 0.3,  # 低置信度
            'model_version': 'test_v1'
        }
        
        from src.realtime.real_time_data_manager import MarketData
        market_data = MarketData(
            symbol="AAPL", price=150.0, volume=1000,
            timestamp=datetime.now(), bid=149.5, ask=150.5
        )
        
        signal = self.trading_system._create_trading_signal(prediction, market_data)
        
        # 弱信号应该被过滤掉
        assert signal is None
    
    @patch('src.realtime.real_time_data_manager.MarketData')
    def test_market_data_validation(self, mock_market_data):
        """测试市场数据验证"""
        # 有效数据
        valid_data = mock_market_data
        valid_data.price = 150.0
        valid_data.volume = 1000
        valid_data.symbol = "AAPL"
        
        self.trading_system.data_manager = MagicMock()
        self.trading_system.data_manager.latest_data = {}
        
        assert self.trading_system._is_valid_market_data(valid_data) is True
        
        # 无效价格
        invalid_data = mock_market_data
        invalid_data.price = -10.0
        invalid_data.volume = 1000
        
        assert self.trading_system._is_valid_market_data(invalid_data) is False
        
        # 负成交量
        invalid_volume_data = mock_market_data
        invalid_volume_data.price = 150.0
        invalid_volume_data.volume = -100
        
        assert self.trading_system._is_valid_market_data(invalid_volume_data) is False
    
    def test_portfolio_value_calculation(self):
        """测试投资组合价值计算"""
        # 模拟经纪商API
        mock_broker = MagicMock()
        mock_account_info = {'total_value': 105000.0}
        mock_broker.get_account_info.return_value = mock_account_info
        
        self.trading_system.broker_api = mock_broker
        
        value = self.trading_system._calculate_portfolio_value()
        assert value == 105000.0
    
    def test_portfolio_value_calculation_no_broker(self):
        """测试无经纪商时的投资组合价值计算"""
        self.trading_system.broker_api = None
        
        value = self.trading_system._calculate_portfolio_value()
        assert value == self.trading_system.initial_capital
    
    def test_system_status_reporting(self):
        """测试系统状态报告"""
        # 设置一些状态
        self.trading_system.system_state.is_running = True
        self.trading_system.system_state.total_trades_today = 5
        self.trading_system.performance_metrics.signals_processed = 100
        self.trading_system.performance_metrics.orders_executed = 25
        self.trading_system.current_positions["AAPL"] = 0.2
        
        status = self.trading_system.get_system_status()
        
        assert isinstance(status, dict)
        assert 'system_state' in status
        assert 'performance_metrics' in status
        assert 'trading_status' in status
        
        # 验证系统状态
        assert status['system_state']['is_running'] is True
        assert status['system_state']['total_trades_today'] == 5
        
        # 验证性能指标
        assert status['performance_metrics']['signals_processed'] == 100
        assert status['performance_metrics']['orders_executed'] == 25
        
        # 验证交易状态
        assert status['trading_status']['current_positions']['AAPL'] == 0.2
        assert status['trading_status']['trading_mode'] == TradingMode.SIMULATION.value
    
    @pytest.mark.asyncio
    async def test_start_system_initialization(self):
        """测试系统启动初始化"""
        # 模拟组件
        self.trading_system.model_service = MagicMock()
        self.trading_system.data_manager = MagicMock()
        self.trading_system.broker_api = AsyncMock()
        
        # 模拟验证方法
        with patch.object(self.trading_system, '_verify_system_ready', return_value=True), \
             patch.object(self.trading_system, '_start_signal_processor'), \
             patch.object(self.trading_system, '_start_heartbeat_monitor'):
            
            symbols = ["AAPL", "MSFT"]
            data_feed_config = {
                "test_feed": {
                    "url": "ws://test.com",
                    "reconnect_interval": 5
                }
            }
            
            await self.trading_system.start_system(symbols, data_feed_config)
            
            # 验证系统状态
            assert self.trading_system.system_state.is_running is True
            assert self.trading_system.system_state.system_start_time is not None
    
    @pytest.mark.asyncio
    async def test_stop_system(self):
        """测试系统停止"""
        # 设置运行状态
        self.trading_system.system_state.is_running = True
        self.trading_system.system_state.is_trading_active = True
        
        # 模拟组件
        self.trading_system.data_manager = MagicMock()
        self.trading_system.model_service = MagicMock()
        self.trading_system.broker_api = AsyncMock()
        
        await self.trading_system.stop_system()
        
        # 验证系统状态
        assert self.trading_system.system_state.is_running is False
        assert self.trading_system.system_state.is_trading_active is False
    
    def test_error_handling(self):
        """测试错误处理"""
        test_error = Exception("Test error")
        
        # 添加错误回调
        error_callback = MagicMock()
        self.trading_system.add_error_callback(error_callback)
        
        # 触发错误处理
        self.trading_system._handle_error("test_error", test_error)
        
        # 验证错误状态更新
        assert self.trading_system.system_state.last_error == "test_error: Test error"
        
        # 验证回调被调用
        error_callback.assert_called_once_with("test_error", test_error)
    
    def test_cleanup(self):
        """测试资源清理"""
        # 添加一些数据
        self.trading_system.current_positions["AAPL"] = 0.2
        self.trading_system.trade_history.append({"test": "data"})
        self.trading_system.signal_callbacks.append(lambda x: None)
        
        # 执行清理
        self.trading_system.cleanup()
        
        # 验证清理结果
        assert len(self.trading_system.current_positions) == 0
        assert len(self.trading_system.trade_history) == 0
        assert len(self.trading_system.signal_callbacks) == 0
    
    def test_system_ready_verification(self):
        """测试系统就绪验证"""
        # 设置所有组件
        self.trading_system.data_manager = MagicMock()
        self.trading_system.model_service = MagicMock()
        self.trading_system.risk_manager = MagicMock()
        self.trading_system.order_manager = MagicMock()
        self.trading_system.broker_api = MagicMock()
        
        # 设置系统状态
        self.trading_system.system_state.data_connection_ok = True
        self.trading_system.system_state.model_service_ok = True
        self.trading_system.system_state.broker_connection_ok = True
        
        result = self.trading_system._verify_system_ready()
        assert result is True
        
        # 测试未就绪状态
        self.trading_system.system_state.data_connection_ok = False
        result = self.trading_system._verify_system_ready()
        assert result is False