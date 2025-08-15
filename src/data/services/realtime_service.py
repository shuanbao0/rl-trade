#!/usr/bin/env python
"""
实时数据传输服务

提供实时数据流、WebSocket连接、数据订阅等功能
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

from .download_models import RealtimeRequest, RealtimeStream
from ..sources.base import DataSource, DataInterval
# DataManager 将在需要时动态导入以避免循环依赖
from ...utils.config import Config
from ...utils.logger import setup_logger, get_default_log_file


class StreamStatus(Enum):
    """流状态枚举"""
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class StreamSubscription:
    """流订阅"""
    stream_id: str
    callback: Callable
    filters: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_time: datetime = field(default_factory=datetime.now)


@dataclass
class DataPoint:
    """数据点"""
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source
        }


class RealtimeService:
    """实时数据传输服务"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化实时数据服务
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.logger = setup_logger(
            name="RealtimeService",
            level="INFO",
            log_file=get_default_log_file("realtime_service")
        )
        
        # 流管理
        self._streams: Dict[str, RealtimeStream] = {}
        self._stream_status: Dict[str, StreamStatus] = {}
        self._stream_threads: Dict[str, threading.Thread] = {}
        self._subscriptions: Dict[str, List[StreamSubscription]] = {}
        
        # 数据缓存
        self._data_buffer: Dict[str, List[DataPoint]] = {}
        self._buffer_lock = threading.RLock()
        
        # 连接管理
        self._data_managers = {}
        
        # 运行状态
        self._running = True
        self._cleanup_thread = None
        
        # 启动清理线程
        self._start_cleanup_thread()
        
        self.logger.info("实时数据服务初始化完成")
    
    def create_stream(self, request: RealtimeRequest) -> RealtimeStream:
        """
        创建实时数据流
        
        Args:
            request: 实时数据请求
            
        Returns:
            RealtimeStream: 实时数据流
        """
        stream_id = f"{request.symbol}_{request.data_source.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        stream = RealtimeStream(
            symbol=request.symbol,
            request=request,
            stream_id=stream_id,
            buffer_size=request.buffer_size
        )
        
        self._streams[stream_id] = stream
        self._stream_status[stream_id] = StreamStatus.IDLE
        self._subscriptions[stream_id] = []
        self._data_buffer[stream_id] = []
        
        self.logger.info(f"创建实时数据流: {request.symbol}, stream_id: {stream_id}")
        
        return stream
    
    def start_stream(self, stream_id: str) -> bool:
        """
        启动实时数据流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功启动
        """
        if stream_id not in self._streams:
            self.logger.error(f"流不存在: {stream_id}")
            return False
        
        if self._stream_status[stream_id] == StreamStatus.ACTIVE:
            self.logger.warning(f"流已经在运行: {stream_id}")
            return True
        
        try:
            stream = self._streams[stream_id]
            
            # 设置状态
            self._stream_status[stream_id] = StreamStatus.CONNECTING
            stream.is_active = True
            stream.start_time = datetime.now()
            
            # 创建数据管理器
            data_manager = self._create_data_manager(stream.request)
            self._data_managers[stream_id] = data_manager
            
            # 启动数据流线程
            thread = threading.Thread(
                target=self._stream_worker,
                args=(stream_id,),
                daemon=True,
                name=f"RealtimeStream-{stream.symbol}"
            )
            thread.start()
            self._stream_threads[stream_id] = thread
            
            self.logger.info(f"实时数据流已启动: {stream.symbol}, stream_id: {stream_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"启动实时数据流失败: {stream_id}, 错误: {e}")
            self._stream_status[stream_id] = StreamStatus.ERROR
            return False
    
    def stop_stream(self, stream_id: str) -> bool:
        """
        停止实时数据流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功停止
        """
        if stream_id not in self._streams:
            self.logger.error(f"流不存在: {stream_id}")
            return False
        
        try:
            # 设置状态
            stream = self._streams[stream_id]
            stream.is_active = False
            self._stream_status[stream_id] = StreamStatus.STOPPED
            
            # 等待线程结束
            if stream_id in self._stream_threads:
                thread = self._stream_threads[stream_id]
                if thread.is_alive():
                    thread.join(timeout=5.0)
                del self._stream_threads[stream_id]
            
            # 清理资源
            if stream_id in self._data_managers:
                del self._data_managers[stream_id]
            
            self.logger.info(f"实时数据流已停止: {stream.symbol}, stream_id: {stream_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"停止实时数据流失败: {stream_id}, 错误: {e}")
            return False
    
    def pause_stream(self, stream_id: str) -> bool:
        """
        暂停实时数据流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功暂停
        """
        if stream_id not in self._streams:
            return False
        
        if self._stream_status[stream_id] == StreamStatus.ACTIVE:
            self._stream_status[stream_id] = StreamStatus.PAUSED
            self.logger.info(f"实时数据流已暂停: {stream_id}")
            return True
        
        return False
    
    def resume_stream(self, stream_id: str) -> bool:
        """
        恢复实时数据流
        
        Args:
            stream_id: 流ID
            
        Returns:
            bool: 是否成功恢复
        """
        if stream_id not in self._streams:
            return False
        
        if self._stream_status[stream_id] == StreamStatus.PAUSED:
            self._stream_status[stream_id] = StreamStatus.ACTIVE
            self.logger.info(f"实时数据流已恢复: {stream_id}")
            return True
        
        return False
    
    def subscribe_stream(self, stream_id: str, callback: Callable, 
                        filters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        订阅实时数据流
        
        Args:
            stream_id: 流ID
            callback: 回调函数
            filters: 数据过滤器
            
        Returns:
            Optional[str]: 订阅ID
        """
        if stream_id not in self._streams:
            return None
        
        subscription_id = f"sub_{stream_id}_{datetime.now().strftime('%H%M%S_%f')}"
        
        subscription = StreamSubscription(
            stream_id=subscription_id,
            callback=callback,
            filters=filters or {}
        )
        
        self._subscriptions[stream_id].append(subscription)
        
        self.logger.info(f"已订阅数据流: {stream_id}, 订阅ID: {subscription_id}")
        return subscription_id
    
    def unsubscribe_stream(self, stream_id: str, subscription_id: str) -> bool:
        """
        取消订阅实时数据流
        
        Args:
            stream_id: 流ID
            subscription_id: 订阅ID
            
        Returns:
            bool: 是否成功取消订阅
        """
        if stream_id not in self._subscriptions:
            return False
        
        subscriptions = self._subscriptions[stream_id]
        for i, sub in enumerate(subscriptions):
            if sub.stream_id == subscription_id:
                del subscriptions[i]
                self.logger.info(f"已取消订阅: {stream_id}, 订阅ID: {subscription_id}")
                return True
        
        return False
    
    def get_stream(self, stream_id: str) -> Optional[RealtimeStream]:
        """
        获取实时数据流
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[RealtimeStream]: 实时数据流
        """
        return self._streams.get(stream_id)
    
    def get_stream_status(self, stream_id: str) -> Optional[StreamStatus]:
        """
        获取流状态
        
        Args:
            stream_id: 流ID
            
        Returns:
            Optional[StreamStatus]: 流状态
        """
        return self._stream_status.get(stream_id)
    
    def list_streams(self) -> List[Dict[str, Any]]:
        """
        列出所有实时数据流
        
        Returns:
            List[Dict[str, Any]]: 流信息列表
        """
        streams_info = []
        
        for stream_id, stream in self._streams.items():
            status = self._stream_status.get(stream_id, StreamStatus.IDLE)
            
            info = {
                'stream_id': stream_id,
                'symbol': stream.symbol,
                'status': status.value,
                'is_active': stream.is_active,
                'start_time': stream.start_time.isoformat() if stream.start_time else None,
                'last_update': stream.last_update.isoformat() if stream.last_update else None,
                'total_updates': stream.total_updates,
                'error_count': stream.error_count,
                'buffer_size': len(stream.data_buffer),
                'subscribers': len(self._subscriptions.get(stream_id, []))
            }
            streams_info.append(info)
        
        return streams_info
    
    def get_latest_data(self, stream_id: str, count: int = 100) -> List[Dict[str, Any]]:
        """
        获取最新数据
        
        Args:
            stream_id: 流ID
            count: 数据条数
            
        Returns:
            List[Dict[str, Any]]: 最新数据列表
        """
        if stream_id not in self._streams:
            return []
        
        stream = self._streams[stream_id]
        return stream.get_latest_data(count)
    
    def get_historical_buffer(self, stream_id: str, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[DataPoint]:
        """
        获取历史缓冲数据
        
        Args:
            stream_id: 流ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[DataPoint]: 历史数据点
        """
        if stream_id not in self._data_buffer:
            return []
        
        with self._buffer_lock:
            buffer_data = self._data_buffer[stream_id].copy()
        
        # 时间过滤
        if start_time or end_time:
            filtered_data = []
            for point in buffer_data:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered_data.append(point)
            return filtered_data
        
        return buffer_data
    
    def export_stream_data(self, stream_id: str, file_path: str, 
                          file_format: str = "csv") -> bool:
        """
        导出流数据
        
        Args:
            stream_id: 流ID
            file_path: 文件路径
            file_format: 文件格式 ("csv", "json", "parquet")
            
        Returns:
            bool: 是否成功导出
        """
        try:
            if stream_id not in self._data_buffer:
                return False
            
            with self._buffer_lock:
                buffer_data = self._data_buffer[stream_id].copy()
            
            if not buffer_data:
                return False
            
            # 转换为DataFrame
            data_list = [point.to_dict() for point in buffer_data]
            df = pd.DataFrame(data_list)
            
            # 保存文件
            if file_format == "csv":
                df.to_csv(file_path, index=False)
            elif file_format == "json":
                df.to_json(file_path, orient='records', indent=2)
            elif file_format == "parquet":
                df.to_parquet(file_path, index=False)
            else:
                return False
            
            self.logger.info(f"流数据已导出: {stream_id} -> {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出流数据失败: {stream_id}, 错误: {e}")
            return False
    
    def shutdown(self):
        """关闭实时数据服务"""
        self.logger.info("正在关闭实时数据服务...")
        
        self._running = False
        
        # 停止所有流
        stream_ids = list(self._streams.keys())
        for stream_id in stream_ids:
            self.stop_stream(stream_id)
        
        # 等待清理线程结束
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        self.logger.info("实时数据服务已关闭")
    
    # 私有方法
    
    def _create_data_manager(self, request: RealtimeRequest):
        """创建数据管理器"""
        data_source_config = self._get_data_source_config(request.data_source)
        
        # 设置代理
        if request.use_proxy:
            self._setup_proxy(request.proxy_host, request.proxy_port)
        
        from ..core.data_manager import DataManager
        return DataManager(
            config=self.config,
            data_source_type=request.data_source,
            data_source_config=data_source_config
        )
    
    def _get_data_source_config(self, data_source: DataSource) -> Dict:
        """获取数据源配置"""
        if data_source == DataSource.FXMINUTE:
            from pathlib import Path
            return {
                'data_directory': str(Path(__file__).parent.parent.parent.parent / 'local_data' / 'FX-1-Minute-Data'),
                'auto_extract': True,
                'cache_extracted': True,
                'extracted_cache_dir': str(Path(__file__).parent.parent.parent.parent / 'fx_minute_cache')
            }
        return {}
    
    def _setup_proxy(self, proxy_host: str, proxy_port: str):
        """设置代理"""
        import os
        proxy_url = f"socks5://{proxy_host}:{proxy_port}"
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
    
    def _stream_worker(self, stream_id: str):
        """流工作线程"""
        stream = self._streams[stream_id]
        data_manager = self._data_managers[stream_id]
        
        self.logger.info(f"流工作线程启动: {stream_id}")
        
        try:
            self._stream_status[stream_id] = StreamStatus.ACTIVE
            
            while stream.is_active and self._running:
                # 检查是否暂停
                if self._stream_status[stream_id] == StreamStatus.PAUSED:
                    time.sleep(1)
                    continue
                
                try:
                    # 获取最新数据（这里是模拟实现）
                    data_point = self._fetch_realtime_data(stream, data_manager)
                    
                    if data_point:
                        # 添加到流缓冲区
                        stream.add_data_point(data_point.to_dict())
                        
                        # 添加到服务缓冲区
                        with self._buffer_lock:
                            buffer = self._data_buffer[stream_id]
                            buffer.append(data_point)
                            
                            # 限制缓冲区大小
                            max_buffer_size = stream.request.buffer_size * 2
                            if len(buffer) > max_buffer_size:
                                buffer[:] = buffer[-max_buffer_size:]
                        
                        # 通知订阅者
                        self._notify_subscribers(stream_id, data_point)
                    
                    # 自动保存
                    if stream.request.auto_save:
                        self._auto_save_check(stream_id)
                    
                    # 等待下次更新
                    time.sleep(stream.request.update_frequency)
                    
                except Exception as e:
                    stream.error_count += 1
                    stream.last_error = str(e)
                    self.logger.error(f"流数据获取失败: {stream_id}, 错误: {e}")
                    
                    # 错误太多时暂停
                    if stream.error_count > 10:
                        self._stream_status[stream_id] = StreamStatus.ERROR
                        break
                    
                    time.sleep(5)  # 错误后等待更长时间
            
        except Exception as e:
            self.logger.error(f"流工作线程异常: {stream_id}, 错误: {e}")
            self._stream_status[stream_id] = StreamStatus.ERROR
        finally:
            self.logger.info(f"流工作线程结束: {stream_id}")
    
    def _fetch_realtime_data(self, stream: RealtimeStream, data_manager) -> Optional[DataPoint]:
        """获取实时数据（模拟实现）"""
        try:
            # 这里是模拟实现，实际应该调用数据源的实时接口
            # 对于演示，我们获取最新的一条历史数据
            symbol = stream.symbol
            interval = stream.request.interval
            
            # 获取最近的数据
            recent_data = data_manager.get_stock_data(
                symbol=symbol,
                period="1d",
                interval=interval.value if hasattr(interval, 'value') else str(interval)
            )
            
            if recent_data is not None and len(recent_data) > 0:
                # 取最后一行数据
                latest_row = recent_data.iloc[-1]
                
                data_dict = {}
                for col, value in latest_row.items():
                    if pd.notna(value):
                        data_dict[col] = float(value) if isinstance(value, (int, float)) else str(value)
                
                return DataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    data=data_dict,
                    source=stream.request.data_source.value
                )
            
        except Exception as e:
            self.logger.error(f"获取实时数据失败: {stream.symbol}, 错误: {e}")
        
        return None
    
    def _notify_subscribers(self, stream_id: str, data_point: DataPoint):
        """通知订阅者"""
        if stream_id not in self._subscriptions:
            return
        
        subscriptions = self._subscriptions[stream_id]
        
        for subscription in subscriptions[:]:  # 创建副本以避免修改时出错
            if not subscription.active:
                continue
            
            try:
                # 应用过滤器
                if self._apply_filters(data_point, subscription.filters):
                    subscription.callback(data_point)
            except Exception as e:
                self.logger.error(f"订阅者回调失败: {subscription.stream_id}, 错误: {e}")
                subscription.active = False
    
    def _apply_filters(self, data_point: DataPoint, filters: Dict[str, Any]) -> bool:
        """应用数据过滤器"""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            if key == 'symbol' and data_point.symbol != expected_value:
                return False
            elif key == 'source' and data_point.source != expected_value:
                return False
            elif key in data_point.data:
                actual_value = data_point.data[key]
                if isinstance(expected_value, dict):
                    # 范围过滤器 {'min': 100, 'max': 200}
                    if 'min' in expected_value and actual_value < expected_value['min']:
                        return False
                    if 'max' in expected_value and actual_value > expected_value['max']:
                        return False
                else:
                    # 精确匹配
                    if actual_value != expected_value:
                        return False
        
        return True
    
    def _auto_save_check(self, stream_id: str):
        """自动保存检查"""
        stream = self._streams[stream_id]
        
        if not hasattr(stream, '_last_save_time'):
            stream._last_save_time = datetime.now()
            return
        
        time_since_save = datetime.now() - stream._last_save_time
        if time_since_save.total_seconds() >= stream.request.save_interval:
            # 执行自动保存
            if stream.request.output_dir:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = f"{stream.request.output_dir}/realtime_{stream.symbol}_{timestamp}.csv"
                
                if self.export_stream_data(stream_id, file_path, "csv"):
                    stream._last_save_time = datetime.now()
                    self.logger.info(f"自动保存完成: {stream_id} -> {file_path}")
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_worker():
            while self._running:
                try:
                    # 清理停止的流
                    stopped_streams = []
                    for stream_id, status in self._stream_status.items():
                        if status == StreamStatus.STOPPED:
                            stopped_streams.append(stream_id)
                    
                    for stream_id in stopped_streams:
                        if stream_id in self._streams:
                            del self._streams[stream_id]
                        if stream_id in self._stream_status:
                            del self._stream_status[stream_id]
                        if stream_id in self._subscriptions:
                            del self._subscriptions[stream_id]
                        if stream_id in self._data_buffer:
                            del self._data_buffer[stream_id]
                    
                    # 清理过期数据
                    cutoff_time = datetime.now() - timedelta(hours=24)  # 保留24小时数据
                    
                    with self._buffer_lock:
                        for stream_id, buffer in self._data_buffer.items():
                            # 移除过期数据点
                            self._data_buffer[stream_id] = [
                                point for point in buffer 
                                if point.timestamp > cutoff_time
                            ]
                    
                    time.sleep(300)  # 每5分钟清理一次
                    
                except Exception as e:
                    self.logger.error(f"清理线程异常: {e}")
                    time.sleep(60)
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True, name="RealtimeCleanup")
        self._cleanup_thread.start()


# 全局实时服务实例
_service_instance = None

def get_realtime_service(config: Optional[Config] = None) -> RealtimeService:
    """获取实时服务单例"""
    global _service_instance
    if _service_instance is None:
        _service_instance = RealtimeService(config)
    return _service_instance