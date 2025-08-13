"""
经纪商接口抽象
定义统一的经纪商API接口，支持多种经纪商实现

主要功能:
1. 抽象基类定义
2. 模拟经纪商实现
3. 账户信息管理
4. 订单执行接口
5. 持仓管理
6. 风险控制集成
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config
from .order_manager import Order, OrderStatus, OrderSide, OrderType


@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str
    total_value: float
    cash: float
    equity: float
    margin_used: float
    margin_available: float
    buying_power: float
    day_trading_buying_power: float
    unrealized_pnl: float
    realized_pnl: float
    currency: str = "USD"
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    average_price: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # "long" or "short"
    cost_basis: float
    market_value: float
    percentage_of_portfolio: float = 0.0
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()
    
    @property
    def is_long(self) -> bool:
        return self.side == "long" and self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.side == "short" and self.quantity < 0


@dataclass
class ExecutionReport:
    """成交报告"""
    execution_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    fees: float = 0.0
    exchange: str = ""
    execution_type: str = "trade"  # trade, partial, cancel, etc.


class BrokerConnectionStatus(Enum):
    """经纪商连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class BrokerAPI(ABC):
    """
    经纪商API抽象基类
    
    定义所有经纪商实现必须提供的接口:
    - 连接管理
    - 账户信息
    - 订单管理
    - 持仓查询
    - 市场数据(可选)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化经纪商API
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.logger = setup_logger(
            name=f"{self.__class__.__name__}",
            level="INFO",
            log_file=get_default_log_file("broker_api")
        )
        
        self.connection_status = BrokerConnectionStatus.DISCONNECTED
        self.account_info: Optional[AccountInfo] = None
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """连接到经纪商"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """获取持仓信息"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """提交订单"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """获取订单状态"""
        pass
    
    @abstractmethod
    async def wait_for_execution(self, order_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """等待订单执行"""
        pass


class SimulatedBroker(BrokerAPI):
    """
    模拟经纪商实现
    
    用于回测和模拟交易:
    - 模拟订单执行
    - 虚拟账户管理
    - 滑点和手续费模拟
    - 延迟模拟
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 1.0,
        execution_delay_ms: float = 50.0,
        config: Optional[Config] = None
    ):
        """
        初始化模拟经纪商
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_bps: 滑点(基点)
            execution_delay_ms: 执行延迟(毫秒)
            config: 配置对象
        """
        super().__init__(config)
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.execution_delay_ms = execution_delay_ms
        
        # 账户状态
        self.cash = initial_capital
        self.equity = initial_capital
        self.total_value = initial_capital
        
        # 执行历史
        self.execution_history: List[ExecutionReport] = []
        self.order_executions: Dict[str, List[ExecutionReport]] = {}
        
        # 价格缓存(用于模拟市场价格)
        self.market_prices: Dict[str, float] = {}
        
        self.logger.info(f"模拟经纪商初始化 - 初始资金: ${initial_capital:,.2f}")
    
    async def connect(self) -> bool:
        """连接到模拟经纪商"""
        try:
            self.connection_status = BrokerConnectionStatus.CONNECTING
            
            # 模拟连接延迟
            await asyncio.sleep(0.1)
            
            # 初始化账户信息
            self.account_info = AccountInfo(
                account_id="SIM_ACCOUNT_001",
                total_value=self.total_value,
                cash=self.cash,
                equity=self.equity,
                margin_used=0.0,
                margin_available=self.cash,
                buying_power=self.cash,
                day_trading_buying_power=self.cash * 4,  # 4倍日内交易杠杆
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
            
            self.connection_status = BrokerConnectionStatus.CONNECTED
            self.logger.info("模拟经纪商连接成功")
            
            return True
            
        except Exception as e:
            self.connection_status = BrokerConnectionStatus.ERROR
            self.logger.error(f"模拟经纪商连接失败: {e}")
            return False
    
    async def disconnect(self) -> None:
        """断开连接"""
        self.connection_status = BrokerConnectionStatus.DISCONNECTED
        self.logger.info("模拟经纪商连接已断开")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connection_status == BrokerConnectionStatus.CONNECTED
    
    async def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        if not self.is_connected():
            raise RuntimeError("未连接到经纪商")
        
        # 更新账户价值
        await self._update_account_value()
        
        return self.account_info
    
    async def get_positions(self) -> Dict[str, Position]:
        """获取持仓信息"""
        if not self.is_connected():
            raise RuntimeError("未连接到经纪商")
        
        # 更新持仓市值
        for symbol, position in self.positions.items():
            if symbol in self.market_prices:
                position.market_price = self.market_prices[symbol]
                position.market_value = position.quantity * position.market_price
                position.unrealized_pnl = position.market_value - position.cost_basis
                position.last_update = datetime.now()
        
        return self.positions.copy()
    
    async def submit_order(self, order: Order) -> str:
        """提交订单"""
        if not self.is_connected():
            raise RuntimeError("未连接到经纪商")
        
        # 生成经纪商订单ID
        broker_order_id = f"SIM_{uuid.uuid4().hex[:8].upper()}"
        
        # 验证订单
        if not await self._validate_order(order):
            raise ValueError(f"订单验证失败: {order.order_id}")
        
        # 添加到待处理订单
        order.client_order_id = order.order_id
        order.order_id = broker_order_id
        self.pending_orders[broker_order_id] = order
        
        # 异步执行订单
        asyncio.create_task(self._execute_order_async(order))
        
        self.logger.info(f"订单提交成功: {broker_order_id}")
        
        return broker_order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if not self.is_connected():
            raise RuntimeError("未连接到经纪商")
        
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            
            self.logger.info(f"订单取消成功: {order_id}")
            return True
        else:
            self.logger.warning(f"未找到待取消的订单: {order_id}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """获取订单状态"""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id].status
        
        # 检查执行历史
        if order_id in self.order_executions:
            return OrderStatus.FILLED
        
        return OrderStatus.CANCELLED
    
    async def wait_for_execution(self, order_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """等待订单执行"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.get_order_status(order_id)
            
            if status == OrderStatus.FILLED:
                executions = self.order_executions.get(order_id, [])
                if executions:
                    exec_report = executions[-1]  # 最后一次执行
                    return {
                        'status': 'FILLED',
                        'executed_quantity': exec_report.quantity,
                        'executed_price': exec_report.price,
                        'commission': exec_report.commission,
                        'execution_time': exec_report.timestamp.isoformat()
                    }
            
            elif status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return {
                    'status': status.value,
                    'executed_quantity': 0.0,
                    'executed_price': 0.0,
                    'commission': 0.0
                }
            
            await asyncio.sleep(0.1)  # 100ms检查间隔
        
        # 超时
        return {
            'status': 'TIMEOUT',
            'executed_quantity': 0.0,
            'executed_price': 0.0,
            'commission': 0.0
        }
    
    def set_market_price(self, symbol: str, price: float) -> None:
        """设置市场价格(用于模拟)"""
        self.market_prices[symbol] = price
        self.logger.debug(f"设置市场价格: {symbol} = ${price:.2f}")
    
    async def _validate_order(self, order: Order) -> bool:
        """验证订单"""
        # 检查资金充足性
        if order.side == OrderSide.BUY:
            required_cash = order.quantity * (order.price or self.market_prices.get(order.symbol, 0))
            commission = required_cash * self.commission_rate
            
            if required_cash + commission > self.cash:
                self.logger.error(f"资金不足: 需要 ${required_cash + commission:.2f}, 可用 ${self.cash:.2f}")
                return False
        
        elif order.side == OrderSide.SELL:
            # 检查持仓充足性
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                available_qty = position.quantity if position else 0
                self.logger.error(f"持仓不足: 需要 {order.quantity}, 可用 {available_qty}")
                return False
        
        return True
    
    async def _execute_order_async(self, order: Order) -> None:
        """异步执行订单"""
        try:
            # 模拟执行延迟
            await asyncio.sleep(self.execution_delay_ms / 1000.0)
            
            # 获取执行价格
            execution_price = self._calculate_execution_price(order)
            
            # 计算手续费
            trade_value = order.quantity * execution_price
            commission = trade_value * self.commission_rate
            
            # 创建执行报告
            execution_report = ExecutionReport(
                execution_id=f"EXEC_{uuid.uuid4().hex[:8].upper()}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=execution_price,
                timestamp=datetime.now(),
                commission=commission
            )
            
            # 更新账户和持仓
            await self._process_execution(execution_report)
            
            # 记录执行
            self.execution_history.append(execution_report)
            if order.order_id not in self.order_executions:
                self.order_executions[order.order_id] = []
            self.order_executions[order.order_id].append(execution_report)
            
            # 更新订单状态
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_price = execution_price
            
            # 从待处理订单中移除
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            
            self.logger.info(f"订单执行完成: {order.order_id} {order.side.value} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"订单执行失败: {e}")
            order.status = OrderStatus.REJECTED
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
    
    def _calculate_execution_price(self, order: Order) -> float:
        """计算执行价格"""
        if order.order_type == OrderType.MARKET:
            # 市价单：使用当前市场价格加滑点
            market_price = self.market_prices.get(order.symbol, order.price or 100.0)
            slippage_factor = self.slippage_bps / 10000.0
            
            if order.side == OrderSide.BUY:
                # 买单：向上滑点
                return market_price * (1 + slippage_factor)
            else:
                # 卖单：向下滑点
                return market_price * (1 - slippage_factor)
        
        elif order.order_type == OrderType.LIMIT:
            # 限价单：使用指定价格
            return order.price
        
        else:
            # 其他订单类型暂时使用市场价格
            return self.market_prices.get(order.symbol, order.price or 100.0)
    
    async def _process_execution(self, execution: ExecutionReport) -> None:
        """处理订单执行"""
        symbol = execution.symbol
        quantity = execution.quantity
        price = execution.price
        commission = execution.commission
        
        if execution.side == "BUY":
            # 买入订单
            cost = quantity * price + commission
            self.cash -= cost
            
            # 更新持仓
            if symbol in self.positions:
                position = self.positions[symbol]
                total_quantity = position.quantity + quantity
                total_cost = position.cost_basis + (quantity * price)
                
                position.quantity = total_quantity
                position.average_price = total_cost / total_quantity
                position.cost_basis = total_cost
                position.side = "long"
                position.last_update = datetime.now()
            else:
                # 新建持仓
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=price,
                    market_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    side="long",
                    cost_basis=quantity * price,
                    market_value=quantity * price
                )
        
        else:  # SELL
            # 卖出订单
            proceeds = quantity * price - commission
            self.cash += proceeds
            
            # 更新持仓
            if symbol in self.positions:
                position = self.positions[symbol]
                
                if position.quantity >= quantity:
                    # 部分或全部平仓
                    avg_cost_per_share = position.cost_basis / position.quantity
                    realized_pnl = quantity * (price - avg_cost_per_share) - commission
                    
                    position.quantity -= quantity
                    position.cost_basis -= quantity * avg_cost_per_share
                    position.realized_pnl += realized_pnl
                    position.last_update = datetime.now()
                    
                    # 如果持仓为0，删除记录
                    if position.quantity <= 0:
                        del self.positions[symbol]
                
                else:
                    self.logger.warning(f"卖出数量超过持仓: {symbol}")
        
        # 更新账户总价值
        await self._update_account_value()
    
    async def _update_account_value(self) -> None:
        """更新账户价值"""
        # 计算持仓总市值
        total_position_value = 0.0
        total_unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in self.market_prices:
                position.market_price = self.market_prices[symbol]
                position.market_value = position.quantity * position.market_price
                position.unrealized_pnl = position.market_value - position.cost_basis
                
                total_position_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl
        
        # 更新账户信息
        if self.account_info:
            self.account_info.total_value = self.cash + total_position_value
            self.account_info.cash = self.cash
            self.account_info.equity = self.cash + total_position_value
            self.account_info.unrealized_pnl = total_unrealized_pnl
            self.account_info.buying_power = self.cash  # 简化计算
            self.account_info.margin_available = self.cash
            self.account_info.last_update = datetime.now()
            
            # 更新持仓占比
            for position in self.positions.values():
                if self.account_info.total_value > 0:
                    position.percentage_of_portfolio = (
                        position.market_value / self.account_info.total_value * 100
                    )
    
    def get_execution_history(self) -> List[ExecutionReport]:
        """获取执行历史"""
        return self.execution_history.copy()
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """获取交易汇总"""
        if not self.execution_history:
            return {
                'total_trades': 0,
                'total_volume': 0.0,
                'total_commission': 0.0,
                'net_pnl': 0.0
            }
        
        total_trades = len(self.execution_history)
        total_volume = sum(exec.quantity * exec.price for exec in self.execution_history)
        total_commission = sum(exec.commission for exec in self.execution_history)
        
        # 计算净盈亏
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        net_pnl = realized_pnl + unrealized_pnl
        
        return {
            'total_trades': total_trades,
            'total_volume': total_volume,
            'total_commission': total_commission,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'net_pnl': net_pnl,
            'current_cash': self.cash,
            'current_equity': self.account_info.equity if self.account_info else 0.0,
            'return_percentage': (net_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0.0
        }


class LiveBrokerTemplate(BrokerAPI):
    """
    实盘经纪商模板
    
    提供实盘经纪商实现的框架结构，
    具体实现需要根据不同经纪商的API进行定制
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        sandbox: bool = True,
        config: Optional[Config] = None
    ):
        """
        初始化实盘经纪商
        
        Args:
            api_key: API密钥
            api_secret: API密码
            sandbox: 是否使用沙盒环境
            config: 配置对象
        """
        super().__init__(config)
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        
        # 连接参数
        self.base_url = "https://api-sandbox.example.com" if sandbox else "https://api.example.com"
        self.websocket_url = "wss://stream-sandbox.example.com" if sandbox else "wss://stream.example.com"
        
        self.logger.info(f"实盘经纪商初始化 - 沙盒模式: {sandbox}")
    
    async def connect(self) -> bool:
        """连接到实盘经纪商"""
        # TODO: 实现具体的认证和连接逻辑
        self.logger.info("实盘经纪商连接功能待实现")
        return False
    
    async def disconnect(self) -> None:
        """断开连接"""
        # TODO: 实现断开连接逻辑
        pass
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        # TODO: 实现连接状态检查
        return False
    
    async def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        # TODO: 实现账户信息获取
        raise NotImplementedError("实盘经纪商功能待实现")
    
    async def get_positions(self) -> Dict[str, Position]:
        """获取持仓信息"""
        # TODO: 实现持仓信息获取
        raise NotImplementedError("实盘经纪商功能待实现")
    
    async def submit_order(self, order: Order) -> str:
        """提交订单"""
        # TODO: 实现订单提交
        raise NotImplementedError("实盘经纪商功能待实现")
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        # TODO: 实现订单取消
        raise NotImplementedError("实盘经纪商功能待实现")
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """获取订单状态"""
        # TODO: 实现订单状态查询
        raise NotImplementedError("实盘经纪商功能待实现")
    
    async def wait_for_execution(self, order_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """等待订单执行"""
        # TODO: 实现订单执行等待
        raise NotImplementedError("实盘经纪商功能待实现")