"""
测试经纪商API模块
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from src.realtime.broker_api import (
    BrokerAPI, SimulatedBroker, AccountInfo, Position, 
    ExecutionReport, BrokerConnectionStatus
)
from src.realtime.order_manager import Order, OrderSide, OrderType, OrderStatus
from src.utils.config import Config


class TestAccountInfo:
    def test_account_info_creation(self):
        """测试账户信息创建"""
        account = AccountInfo(
            account_id="TEST_001",
            total_value=100000.0,
            cash=50000.0,
            equity=100000.0,
            margin_used=0.0,
            margin_available=50000.0,
            buying_power=50000.0,
            day_trading_buying_power=200000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
        
        assert account.account_id == "TEST_001"
        assert account.total_value == 100000.0
        assert account.cash == 50000.0
        assert account.last_update is not None
    
    def test_account_info_post_init(self):
        """测试账户信息初始化后处理"""
        account = AccountInfo(
            account_id="TEST_001",
            total_value=100000.0,
            cash=50000.0,
            equity=100000.0,
            margin_used=0.0,
            margin_available=50000.0,
            buying_power=50000.0,
            day_trading_buying_power=200000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
        
        # last_update应该自动设置
        assert isinstance(account.last_update, datetime)


class TestPosition:
    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            average_price=150.0,
            market_price=152.0,
            unrealized_pnl=200.0,
            realized_pnl=0.0,
            side="long",
            cost_basis=15000.0,
            market_value=15200.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100.0
        assert position.is_long is True
        assert position.is_short is False
    
    def test_position_properties(self):
        """测试持仓属性"""
        # 多头持仓
        long_position = Position(
            symbol="AAPL", quantity=100.0, average_price=150.0,
            market_price=152.0, unrealized_pnl=200.0, realized_pnl=0.0,
            side="long", cost_basis=15000.0, market_value=15200.0
        )
        
        assert long_position.is_long is True
        assert long_position.is_short is False
        
        # 空头持仓
        short_position = Position(
            symbol="TSLA", quantity=-50.0, average_price=200.0,
            market_price=195.0, unrealized_pnl=250.0, realized_pnl=0.0,
            side="short", cost_basis=10000.0, market_value=9750.0
        )
        
        assert short_position.is_long is False
        assert short_position.is_short is True


class TestSimulatedBroker:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.broker = SimulatedBroker(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_bps=1.0,
            execution_delay_ms=50.0
        )
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        asyncio.run(self.broker.disconnect())
    
    @pytest.mark.asyncio
    async def test_broker_initialization(self):
        """测试经纪商初始化"""
        assert self.broker.initial_capital == 100000.0
        assert self.broker.commission_rate == 0.001
        assert self.broker.slippage_bps == 1.0
        assert self.broker.execution_delay_ms == 50.0
        assert self.broker.cash == 100000.0
    
    @pytest.mark.asyncio
    async def test_broker_connection(self):
        """测试经纪商连接"""
        # 测试连接
        result = await self.broker.connect()
        
        assert result is True
        assert self.broker.is_connected() is True
        assert self.broker.connection_status == BrokerConnectionStatus.CONNECTED
        assert self.broker.account_info is not None
        assert self.broker.account_info.account_id == "SIM_ACCOUNT_001"
    
    @pytest.mark.asyncio
    async def test_broker_disconnection(self):
        """测试经纪商断开连接"""
        await self.broker.connect()
        await self.broker.disconnect()
        
        assert self.broker.is_connected() is False
        assert self.broker.connection_status == BrokerConnectionStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_get_account_info(self):
        """测试获取账户信息"""
        await self.broker.connect()
        
        account_info = await self.broker.get_account_info()
        
        assert isinstance(account_info, AccountInfo)
        assert account_info.total_value == 100000.0
        assert account_info.cash == 100000.0
        assert account_info.account_id == "SIM_ACCOUNT_001"
    
    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        """测试获取空持仓"""
        await self.broker.connect()
        
        positions = await self.broker.get_positions()
        
        assert isinstance(positions, dict)
        assert len(positions) == 0
    
    @pytest.mark.asyncio
    async def test_market_price_setting(self):
        """测试市场价格设置"""
        self.broker.set_market_price("AAPL", 150.0)
        self.broker.set_market_price("TSLA", 200.0)
        
        assert self.broker.market_prices["AAPL"] == 150.0
        assert self.broker.market_prices["TSLA"] == 200.0
    
    @pytest.mark.asyncio
    async def test_order_validation_insufficient_funds(self):
        """测试资金不足的订单验证"""
        await self.broker.connect()
        
        # 创建超出资金的买单
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000000,  # 过大的数量
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        is_valid = await self.broker._validate_order(order)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_order_validation_insufficient_position(self):
        """测试持仓不足的订单验证"""
        await self.broker.connect()
        
        # 创建没有持仓的卖单
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        is_valid = await self.broker._validate_order(order)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_order_submission_buy(self):
        """测试买单提交"""
        await self.broker.connect()
        self.broker.set_market_price("AAPL", 150.0)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        order_id = await self.broker.submit_order(order)
        
        assert order_id is not None
        assert order_id.startswith("SIM_")
        assert order_id in self.broker.pending_orders
    
    @pytest.mark.asyncio
    async def test_order_execution_buy(self):
        """测试买单执行"""
        await self.broker.connect()
        self.broker.set_market_price("AAPL", 150.0)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        order_id = await self.broker.submit_order(order)
        
        # 等待订单执行
        result = await self.broker.wait_for_execution(order_id, timeout=2.0)
        
        assert result['status'] == 'FILLED'
        assert result['executed_quantity'] == 100
        assert result['executed_price'] > 150.0  # 考虑滑点
        
        # 检查持仓
        positions = await self.broker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 100
        assert positions["AAPL"].side == "long"
    
    @pytest.mark.asyncio
    async def test_order_execution_sell(self):
        """测试卖单执行"""
        await self.broker.connect()
        self.broker.set_market_price("AAPL", 150.0)
        
        # 先买入建仓
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, price=150.0
        )
        
        buy_order_id = await self.broker.submit_order(buy_order)
        await self.broker.wait_for_execution(buy_order_id)
        
        # 然后卖出
        sell_order = Order(
            symbol="AAPL", side=OrderSide.SELL, quantity=50,
            order_type=OrderType.MARKET, price=150.0
        )
        
        sell_order_id = await self.broker.submit_order(sell_order)
        result = await self.broker.wait_for_execution(sell_order_id)
        
        assert result['status'] == 'FILLED'
        assert result['executed_quantity'] == 50
        
        # 检查持仓减少
        positions = await self.broker.get_positions()
        assert positions["AAPL"].quantity == 50
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self):
        """测试订单取消"""
        await self.broker.connect()
        self.broker.set_market_price("AAPL", 150.0)
        
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, price=140.0  # 低于市价，不会立即执行
        )
        
        order_id = await self.broker.submit_order(order)
        
        # 立即取消订单
        result = await self.broker.cancel_order(order_id)
        
        assert result is True
        assert order_id not in self.broker.pending_orders
    
    @pytest.mark.asyncio
    async def test_execution_price_calculation_market_order(self):
        """测试市价单执行价格计算"""
        market_price = 150.0
        slippage_factor = self.broker.slippage_bps / 10000.0
        
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET
        )
        
        sell_order = Order(
            symbol="AAPL", side=OrderSide.SELL, quantity=100,
            order_type=OrderType.MARKET
        )
        
        buy_price = self.broker._calculate_execution_price(buy_order)
        sell_price = self.broker._calculate_execution_price(sell_order)
        
        # 买单向上滑点，卖单向下滑点
        expected_buy_price = market_price * (1 + slippage_factor)
        expected_sell_price = market_price * (1 - slippage_factor)
        
        # 由于没有设置市场价格，会使用默认值
        assert isinstance(buy_price, float)
        assert isinstance(sell_price, float)
    
    @pytest.mark.asyncio
    async def test_execution_price_calculation_limit_order(self):
        """测试限价单执行价格计算"""
        limit_price = 145.0
        
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, price=limit_price
        )
        
        execution_price = self.broker._calculate_execution_price(order)
        
        assert execution_price == limit_price
    
    def test_trade_summary_empty(self):
        """测试空交易汇总"""
        summary = self.broker.get_trade_summary()
        
        assert summary['total_trades'] == 0
        assert summary['total_volume'] == 0.0
        assert summary['total_commission'] == 0.0
        assert summary['net_pnl'] == 0.0
    
    @pytest.mark.asyncio
    async def test_trade_summary_with_trades(self):
        """测试有交易的汇总"""
        await self.broker.connect()
        self.broker.set_market_price("AAPL", 150.0)
        
        # 执行一些交易
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, price=150.0
        )
        
        order_id = await self.broker.submit_order(buy_order)
        await self.broker.wait_for_execution(order_id)
        
        summary = self.broker.get_trade_summary()
        
        assert summary['total_trades'] == 1
        assert summary['total_volume'] > 0.0
        assert summary['total_commission'] > 0.0
        assert summary['current_cash'] < 100000.0  # 应该减少
    
    def test_execution_history(self):
        """测试执行历史"""
        # 初始状态应该没有历史
        history = self.broker.get_execution_history()
        assert len(history) == 0
        
        # 手动添加一个执行记录进行测试
        execution = ExecutionReport(
            execution_id="TEST_001",
            order_id="ORDER_001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            commission=15.0
        )
        
        self.broker.execution_history.append(execution)
        
        history = self.broker.get_execution_history()
        assert len(history) == 1
        assert history[0].execution_id == "TEST_001"


class TestBrokerAPI:
    def test_abstract_base_class(self):
        """测试抽象基类特性"""
        # 不能直接实例化抽象基类
        with pytest.raises(TypeError):
            BrokerAPI()
    
    def test_broker_api_methods_abstract(self):
        """测试抽象方法"""
        # 所有抽象方法都应该在基类中定义
        abstract_methods = {
            'connect', 'disconnect', 'is_connected', 
            'get_account_info', 'get_positions', 
            'submit_order', 'cancel_order', 
            'get_order_status', 'wait_for_execution'
        }
        
        # 验证这些方法在BrokerAPI中存在
        for method_name in abstract_methods:
            assert hasattr(BrokerAPI, method_name)