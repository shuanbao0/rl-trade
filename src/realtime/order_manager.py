"""
订单管理组件
处理订单创建、验证、跟踪和生命周期管理

主要功能:
1. 订单创建和验证
2. 订单状态跟踪
3. 订单历史管理
4. 风险检查和限制
5. 订单统计和分析
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"      # 市价单
    LIMIT = "LIMIT"        # 限价单
    STOP = "STOP"          # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 止损限价单


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"        # 待提交
    SUBMITTED = "SUBMITTED"    # 已提交
    PARTIAL_FILLED = "PARTIAL_FILLED"  # 部分成交
    FILLED = "FILLED"          # 完全成交
    CANCELLED = "CANCELLED"    # 已取消
    REJECTED = "REJECTED"      # 已拒绝
    EXPIRED = "EXPIRED"        # 已过期


class TimeInForce(Enum):
    """订单有效期"""
    GTC = "GTC"    # Good Till Cancel
    IOC = "IOC"    # Immediate or Cancel
    FOK = "FOK"    # Fill or Kill
    DAY = "DAY"    # 当日有效


@dataclass
class Order:
    """订单对象"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    
    # 状态信息
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_price: Optional[float] = None
    
    # 时间信息
    create_time: Optional[datetime] = None
    submit_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    
    # 元数据
    client_order_id: Optional[str] = None
    tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.create_time is None:
            self.create_time = datetime.now()
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def is_buy(self) -> bool:
        """是否为买单"""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """是否为卖单"""
        return self.side == OrderSide.SELL
    
    @property
    def is_pending(self) -> bool:
        """是否为待处理状态"""
        return self.status == OrderStatus.PENDING
    
    @property
    def is_active(self) -> bool:
        """是否为活跃状态"""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]
    
    @property
    def is_finished(self) -> bool:
        """是否已完成"""
        return self.status in [
            OrderStatus.FILLED, 
            OrderStatus.CANCELLED, 
            OrderStatus.REJECTED, 
            OrderStatus.EXPIRED
        ]
    
    @property
    def fill_ratio(self) -> float:
        """成交比例"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_price': self.average_price,
            'create_time': self.create_time.isoformat() if self.create_time else None,
            'submit_time': self.submit_time.isoformat() if self.submit_time else None,
            'update_time': self.update_time.isoformat() if self.update_time else None,
            'client_order_id': self.client_order_id,
            'tag': self.tag,
            'metadata': self.metadata,
            'fill_ratio': self.fill_ratio
        }


@dataclass
class OrderLimits:
    """订单限制配置"""
    max_order_size: float = 1000000.0      # 最大订单金额
    max_quantity: float = 100000.0         # 最大订单数量
    min_quantity: float = 1.0              # 最小订单数量
    max_orders_per_symbol: int = 10        # 每个标的最大订单数
    max_orders_per_day: int = 1000         # 每日最大订单数
    max_order_value_ratio: float = 0.1     # 最大订单价值比例
    price_deviation_limit: float = 0.05    # 价格偏离限制
    
    def validate(self) -> None:
        """验证配置参数"""
        if self.max_order_size <= 0:
            raise ValueError("max_order_size must be positive")
        if self.max_quantity <= 0:
            raise ValueError("max_quantity must be positive")
        if self.min_quantity <= 0:
            raise ValueError("min_quantity must be positive")
        if self.min_quantity > self.max_quantity:
            raise ValueError("min_quantity cannot be greater than max_quantity")


@dataclass
class OrderStatistics:
    """订单统计"""
    total_orders: int = 0
    buy_orders: int = 0
    sell_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    total_volume: float = 0.0
    total_value: float = 0.0
    average_fill_time_seconds: float = 0.0
    success_rate: float = 0.0
    last_reset_time: Optional[datetime] = None
    
    def reset(self):
        """重置统计"""
        self.total_orders = 0
        self.buy_orders = 0
        self.sell_orders = 0
        self.filled_orders = 0
        self.cancelled_orders = 0
        self.rejected_orders = 0
        self.total_volume = 0.0
        self.total_value = 0.0
        self.average_fill_time_seconds = 0.0
        self.success_rate = 0.0
        self.last_reset_time = datetime.now()


class OrderManager:
    """
    订单管理器
    
    提供订单生命周期管理:
    - 订单创建和验证
    - 订单状态跟踪
    - 风险检查
    - 统计分析
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        trading_mode: str = "simulation",
        order_limits: Optional[OrderLimits] = None
    ):
        """
        初始化订单管理器
        
        Args:
            config: 配置对象
            trading_mode: 交易模式
            order_limits: 订单限制配置
        """
        self.config = config or Config()
        self.trading_mode = trading_mode
        self.order_limits = order_limits or OrderLimits()
        self.order_limits.validate()
        
        # 初始化日志
        self.logger = setup_logger(
            name="OrderManager",
            level="INFO",
            log_file=get_default_log_file("order_manager")
        )
        
        # 订单存储
        self.orders: Dict[str, Order] = {}           # order_id -> Order
        self.active_orders: Dict[str, Order] = {}    # order_id -> Order
        self.symbol_orders: Dict[str, List[str]] = {}  # symbol -> [order_ids]
        
        # 统计信息
        self.statistics = OrderStatistics()
        self.statistics.last_reset_time = datetime.now()
        
        # 当日统计
        self.daily_order_count = 0
        self.daily_reset_date = datetime.now().date()
        
        self.logger.info(f"OrderManager 初始化完成 - 模式: {trading_mode}")
        self.logger.info(f"订单限制: {self.order_limits}")
    
    def create_order(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: float,
        order_type: Union[str, OrderType] = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Union[str, TimeInForce] = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        创建订单
        
        Args:
            symbol: 交易标的
            side: 买卖方向
            quantity: 数量
            order_type: 订单类型
            price: 价格（限价单需要）
            stop_price: 止损价格
            time_in_force: 有效期
            client_order_id: 客户端订单ID
            tag: 标签
            metadata: 元数据
            
        Returns:
            Order: 创建的订单对象
        """
        try:
            # 参数类型转换
            if isinstance(side, str):
                side = OrderSide(side.upper())
            if isinstance(order_type, str):
                order_type = OrderType(order_type.upper())
            if isinstance(time_in_force, str):
                time_in_force = TimeInForce(time_in_force.upper())
            
            # 生成订单ID
            order_id = self._generate_order_id()
            
            # 创建订单对象
            order = Order(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                client_order_id=client_order_id,
                tag=tag,
                metadata=metadata or {}
            )
            
            # 验证订单
            validation_result = self.validate_order(order)
            if not validation_result:
                raise ValueError("订单验证失败")
            
            # 存储订单
            self.orders[order_id] = order
            
            # 更新符号订单映射
            if symbol not in self.symbol_orders:
                self.symbol_orders[symbol] = []
            self.symbol_orders[symbol].append(order_id)
            
            # 更新统计
            self._update_creation_statistics(order)
            
            self.logger.info(f"订单创建成功: {order_id} {side.value} {quantity} {symbol}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"创建订单失败: {e}")
            raise
    
    def _generate_order_id(self) -> str:
        """生成唯一订单ID"""
        timestamp = int(time.time() * 1000000)  # 微秒时间戳
        random_part = str(uuid.uuid4())[:8]
        return f"ORD_{timestamp}_{random_part}"
    
    def validate_order(self, order: Order) -> bool:
        """
        验证订单
        
        Args:
            order: 待验证的订单
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 基础参数验证
            if order.quantity <= 0:
                self.logger.error(f"订单数量无效: {order.quantity}")
                return False
            
            if order.quantity < self.order_limits.min_quantity:
                self.logger.error(f"订单数量过小: {order.quantity} < {self.order_limits.min_quantity}")
                return False
            
            if order.quantity > self.order_limits.max_quantity:
                self.logger.error(f"订单数量过大: {order.quantity} > {self.order_limits.max_quantity}")
                return False
            
            # 价格验证
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order.price is None or order.price <= 0:
                    self.logger.error(f"限价单价格无效: {order.price}")
                    return False
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order.stop_price is None or order.stop_price <= 0:
                    self.logger.error(f"止损价格无效: {order.stop_price}")
                    return False
            
            # 订单金额验证
            if order.price:
                order_value = order.quantity * order.price
                if order_value > self.order_limits.max_order_size:
                    self.logger.error(f"订单金额过大: {order_value} > {self.order_limits.max_order_size}")
                    return False
            
            # 单个标的订单数量限制
            if order.symbol in self.symbol_orders:
                active_count = len([
                    oid for oid in self.symbol_orders[order.symbol]
                    if oid in self.active_orders
                ])
                if active_count >= self.order_limits.max_orders_per_symbol:
                    self.logger.error(f"标的 {order.symbol} 活跃订单数过多: {active_count}")
                    return False
            
            # 每日订单数量限制
            self._check_daily_limit()
            if self.daily_order_count >= self.order_limits.max_orders_per_day:
                self.logger.error(f"每日订单数量超限: {self.daily_order_count}")
                return False
            
            # 价格合理性检查（如果有参考价格）
            if order.price and hasattr(self, 'current_prices'):
                reference_price = self.current_prices.get(order.symbol)
                if reference_price:
                    price_deviation = abs(order.price - reference_price) / reference_price
                    if price_deviation > self.order_limits.price_deviation_limit:
                        self.logger.error(f"价格偏离过大: {price_deviation:.2%}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"订单验证异常: {e}")
            return False
    
    def _check_daily_limit(self) -> None:
        """检查每日限制"""
        current_date = datetime.now().date()
        if current_date != self.daily_reset_date:
            # 新的一天，重置计数
            self.daily_order_count = 0
            self.daily_reset_date = current_date
    
    def submit_order(self, order_id: str) -> bool:
        """
        提交订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            bool: 提交是否成功
        """
        try:
            if order_id not in self.orders:
                self.logger.error(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            if order.status != OrderStatus.PENDING:
                self.logger.error(f"订单状态不允许提交: {order.status}")
                return False
            
            # 更新订单状态
            order.status = OrderStatus.SUBMITTED
            order.submit_time = datetime.now()
            order.update_time = datetime.now()
            
            # 添加到活跃订单
            self.active_orders[order_id] = order
            
            # 更新当日订单计数
            self.daily_order_count += 1
            
            self.logger.info(f"订单提交成功: {order_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"提交订单失败: {e}")
            return False
    
    def update_order_status(
        self,
        order_id: str,
        status: Union[str, OrderStatus],
        filled_quantity: Optional[float] = None,
        average_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新订单状态
        
        Args:
            order_id: 订单ID
            status: 新状态
            filled_quantity: 成交数量
            average_price: 平均成交价格
            metadata: 附加元数据
            
        Returns:
            bool: 更新是否成功
        """
        try:
            if order_id not in self.orders:
                self.logger.error(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            # 状态类型转换
            if isinstance(status, str):
                status = OrderStatus(status.upper())
            
            old_status = order.status
            order.status = status
            order.update_time = datetime.now()
            
            # 更新成交信息
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
                order.remaining_quantity = order.quantity - filled_quantity
            
            if average_price is not None:
                order.average_price = average_price
            
            # 更新元数据
            if metadata:
                order.metadata.update(metadata)
            
            # 处理状态变化
            if status.is_finished and order_id in self.active_orders:
                # 移除活跃订单
                del self.active_orders[order_id]
                
                # 更新统计
                self._update_completion_statistics(order, old_status)
            
            self.logger.info(f"订单状态更新: {order_id} {old_status.value} -> {status.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新订单状态失败: {e}")
            return False
    
    def cancel_order(self, order_id: str, reason: str = "用户取消") -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            reason: 取消原因
            
        Returns:
            bool: 取消是否成功
        """
        try:
            if order_id not in self.orders:
                self.logger.error(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            if not order.is_active:
                self.logger.error(f"订单状态不允许取消: {order.status}")
                return False
            
            # 更新状态
            success = self.update_order_status(
                order_id,
                OrderStatus.CANCELLED,
                metadata={'cancel_reason': reason}
            )
            
            if success:
                self.logger.info(f"订单取消成功: {order_id}, 原因: {reason}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"取消订单失败: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        获取订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            Optional[Order]: 订单对象
        """
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        获取指定标的的所有订单
        
        Args:
            symbol: 交易标的
            
        Returns:
            List[Order]: 订单列表
        """
        symbol = symbol.upper()
        if symbol not in self.symbol_orders:
            return []
        
        return [self.orders[oid] for oid in self.symbol_orders[symbol] if oid in self.orders]
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        获取活跃订单
        
        Args:
            symbol: 可选的交易标的过滤
            
        Returns:
            List[Order]: 活跃订单列表
        """
        orders = list(self.active_orders.values())
        
        if symbol:
            symbol = symbol.upper()
            orders = [order for order in orders if order.symbol == symbol]
        
        return orders
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """
        获取订单历史
        
        Args:
            symbol: 可选的交易标的过滤
            start_time: 开始时间
            end_time: 结束时间
            status: 状态过滤
            limit: 返回数量限制
            
        Returns:
            List[Order]: 订单历史列表
        """
        orders = list(self.orders.values())
        
        # 应用过滤条件
        if symbol:
            symbol = symbol.upper()
            orders = [order for order in orders if order.symbol == symbol]
        
        if start_time:
            orders = [order for order in orders if order.create_time >= start_time]
        
        if end_time:
            orders = [order for order in orders if order.create_time <= end_time]
        
        if status:
            orders = [order for order in orders if order.status == status]
        
        # 按创建时间降序排序
        orders.sort(key=lambda x: x.create_time, reverse=True)
        
        return orders[:limit]
    
    def _update_creation_statistics(self, order: Order) -> None:
        """更新创建统计"""
        self.statistics.total_orders += 1
        
        if order.is_buy:
            self.statistics.buy_orders += 1
        else:
            self.statistics.sell_orders += 1
    
    def _update_completion_statistics(self, order: Order, old_status: OrderStatus) -> None:
        """更新完成统计"""
        if order.status == OrderStatus.FILLED:
            self.statistics.filled_orders += 1
            
            # 更新交易量和价值
            self.statistics.total_volume += order.filled_quantity
            if order.average_price:
                self.statistics.total_value += order.filled_quantity * order.average_price
            
            # 更新平均成交时间
            if order.submit_time and order.update_time:
                fill_time = (order.update_time - order.submit_time).total_seconds()
                if self.statistics.average_fill_time_seconds == 0:
                    self.statistics.average_fill_time_seconds = fill_time
                else:
                    # 指数移动平均
                    alpha = 0.1
                    self.statistics.average_fill_time_seconds = (
                        (1 - alpha) * self.statistics.average_fill_time_seconds + 
                        alpha * fill_time
                    )
        
        elif order.status == OrderStatus.CANCELLED:
            self.statistics.cancelled_orders += 1
        
        elif order.status == OrderStatus.REJECTED:
            self.statistics.rejected_orders += 1
        
        # 更新成功率
        total_processed = (
            self.statistics.filled_orders + 
            self.statistics.cancelled_orders + 
            self.statistics.rejected_orders
        )
        if total_processed > 0:
            self.statistics.success_rate = self.statistics.filled_orders / total_processed
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_orders': self.statistics.total_orders,
            'buy_orders': self.statistics.buy_orders,
            'sell_orders': self.statistics.sell_orders,
            'filled_orders': self.statistics.filled_orders,
            'cancelled_orders': self.statistics.cancelled_orders,
            'rejected_orders': self.statistics.rejected_orders,
            'active_orders': len(self.active_orders),
            'total_volume': self.statistics.total_volume,
            'total_value': self.statistics.total_value,
            'average_fill_time_seconds': self.statistics.average_fill_time_seconds,
            'success_rate': self.statistics.success_rate,
            'daily_order_count': self.daily_order_count,
            'symbols_count': len(self.symbol_orders),
            'last_reset_time': self.statistics.last_reset_time.isoformat() if self.statistics.last_reset_time else None
        }
    
    def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """获取指定标的的统计信息"""
        symbol = symbol.upper()
        orders = self.get_orders_by_symbol(symbol)
        
        if not orders:
            return {'symbol': symbol, 'total_orders': 0}
        
        buy_orders = len([o for o in orders if o.is_buy])
        sell_orders = len([o for o in orders if o.is_sell])
        filled_orders = len([o for o in orders if o.status == OrderStatus.FILLED])
        active_orders = len([o for o in orders if o.is_active])
        
        total_volume = sum(o.filled_quantity for o in orders if o.status == OrderStatus.FILLED)
        total_value = sum(
            o.filled_quantity * (o.average_price or 0) 
            for o in orders 
            if o.status == OrderStatus.FILLED and o.average_price
        )
        
        return {
            'symbol': symbol,
            'total_orders': len(orders),
            'buy_orders': buy_orders,
            'sell_orders': sell_orders,
            'filled_orders': filled_orders,
            'active_orders': active_orders,
            'total_volume': total_volume,
            'total_value': total_value,
            'fill_rate': filled_orders / len(orders) if orders else 0
        }
    
    def cleanup_old_orders(self, days: int = 30) -> int:
        """
        清理旧订单
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的订单数量
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            if order.is_finished and order.create_time < cutoff_time:
                orders_to_remove.append(order_id)
        
        # 移除旧订单
        removed_count = 0
        for order_id in orders_to_remove:
            if order_id in self.orders:
                order = self.orders[order_id]
                del self.orders[order_id]
                
                # 从符号映射中移除
                if order.symbol in self.symbol_orders:
                    if order_id in self.symbol_orders[order.symbol]:
                        self.symbol_orders[order.symbol].remove(order_id)
                
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"清理了 {removed_count} 个旧订单")
        
        return removed_count
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.statistics.reset()
        self.daily_order_count = 0
        self.daily_reset_date = datetime.now().date()
        
        self.logger.info("订单统计信息已重置")
    
    def export_orders_to_csv(self, filename: str, symbol: Optional[str] = None) -> None:
        """
        导出订单到CSV文件
        
        Args:
            filename: 文件名
            symbol: 可选的标的过滤
        """
        try:
            orders = list(self.orders.values())
            
            if symbol:
                symbol = symbol.upper()
                orders = [order for order in orders if order.symbol == symbol]
            
            if not orders:
                self.logger.warning("没有订单可导出")
                return
            
            # 转换为DataFrame
            data = [order.to_dict() for order in orders]
            df = pd.DataFrame(data)
            
            # 导出CSV
            df.to_csv(filename, index=False, encoding='utf-8')
            
            self.logger.info(f"订单已导出到: {filename}, 共 {len(orders)} 条记录")
            
        except Exception as e:
            self.logger.error(f"导出订单失败: {e}")
            raise