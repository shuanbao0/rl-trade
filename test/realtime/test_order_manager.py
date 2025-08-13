"""
测试订单管理器模块
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.realtime.order_manager import OrderManager, Order, OrderSide, OrderType, OrderStatus
from src.utils.config import Config


class TestOrder:
    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.price == 150.0
        assert order.status == OrderStatus.PENDING


class TestOrderManager:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = Config()
        self.order_manager = OrderManager(config=self.config)
    
    def test_order_manager_initialization(self):
        """测试订单管理器初始化"""
        assert self.order_manager.config is not None
        assert len(self.order_manager.active_orders) == 0
        assert len(self.order_manager.order_history) == 0
    
    def test_create_order(self):
        """测试创建订单"""
        order = self.order_manager.create_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            price=150.0
        )
        
        assert isinstance(order, Order)
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_id is not None
    
    def test_validate_order_valid(self):
        """测试有效订单验证"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        is_valid = self.order_manager.validate_order(order)
        assert is_valid is True
    
    def test_validate_order_invalid_quantity(self):
        """测试无效数量订单验证"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,  # 无效数量
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        is_valid = self.order_manager.validate_order(order)
        assert is_valid is False
    
    def test_validate_order_invalid_price(self):
        """测试无效价格订单验证"""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            price=-10.0  # 无效价格
        )
        
        is_valid = self.order_manager.validate_order(order)
        assert is_valid is False
    
    def test_submit_order(self):
        """测试提交订单"""
        order = self.order_manager.create_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            price=150.0
        )
        
        result = self.order_manager.submit_order(order)
        
        assert result is True
        assert order.order_id in self.order_manager.active_orders
        assert order.status == OrderStatus.SUBMITTED
    
    def test_cancel_order(self):
        """测试取消订单"""
        # 先创建并提交订单
        order = self.order_manager.create_order(
            symbol="AAPL",
            side="BUY", 
            quantity=100,
            order_type="LIMIT",
            price=145.0  # 低于市价，不会立即执行
        )
        
        self.order_manager.submit_order(order)
        
        # 取消订单
        result = self.order_manager.cancel_order(order.order_id)
        
        assert result is True
        assert order.status == OrderStatus.CANCELLED
        assert order.order_id not in self.order_manager.active_orders
    
    def test_get_order_status(self):
        """测试获取订单状态"""
        order = self.order_manager.create_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            price=150.0
        )
        
        # 提交前
        status = self.order_manager.get_order_status(order.order_id)
        assert status == OrderStatus.PENDING
        
        # 提交后
        self.order_manager.submit_order(order)
        status = self.order_manager.get_order_status(order.order_id)
        assert status == OrderStatus.SUBMITTED
    
    def test_get_active_orders(self):
        """测试获取活跃订单"""
        # 创建几个订单
        order1 = self.order_manager.create_order("AAPL", "BUY", 100, "MARKET", 150.0)
        order2 = self.order_manager.create_order("MSFT", "SELL", 50, "LIMIT", 200.0)
        
        self.order_manager.submit_order(order1)
        self.order_manager.submit_order(order2)
        
        active_orders = self.order_manager.get_active_orders()
        
        assert len(active_orders) == 2
        assert order1.order_id in active_orders
        assert order2.order_id in active_orders
    
    def test_get_order_history(self):
        """测试获取订单历史"""
        order = self.order_manager.create_order("AAPL", "BUY", 100, "MARKET", 150.0)
        self.order_manager.submit_order(order)
        
        # 模拟订单完成
        order.status = OrderStatus.FILLED
        order.filled_quantity = 100
        order.average_price = 150.5
        self.order_manager._move_to_history(order)
        
        history = self.order_manager.get_order_history()
        
        assert len(history) == 1
        assert history[0].order_id == order.order_id
        assert history[0].status == OrderStatus.FILLED
    
    def test_order_statistics(self):
        """测试订单统计"""
        # 创建多个不同状态的订单
        order1 = self.order_manager.create_order("AAPL", "BUY", 100, "MARKET", 150.0)
        order2 = self.order_manager.create_order("MSFT", "SELL", 50, "LIMIT", 200.0)
        
        self.order_manager.submit_order(order1)
        self.order_manager.submit_order(order2)
        
        # 模拟一个订单完成
        order1.status = OrderStatus.FILLED
        self.order_manager._move_to_history(order1)
        
        stats = self.order_manager.get_order_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_orders' in stats
        assert 'active_orders' in stats
        assert 'filled_orders' in stats