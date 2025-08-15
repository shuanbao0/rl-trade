#!/usr/bin/env python
"""
数据下载服务的数据模型

定义了下载请求、响应和配置的数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

from ..sources.base import DataSource, DataPeriod, DataInterval


@dataclass
class DownloadRequest:
    """单个数据下载请求"""
    symbol: str
    data_source: DataSource = DataSource.YFINANCE
    period: Union[str, DataPeriod] = DataPeriod.MONTH_1
    interval: Union[str, DataInterval] = DataInterval.DAY_1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # 处理选项
    include_features: bool = True
    split_datasets: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # 输出选项
    output_dir: Optional[str] = None
    save_data: bool = True
    file_formats: List[str] = field(default_factory=lambda: ['csv', 'pkl'])
    
    # 代理设置
    use_proxy: bool = False
    proxy_host: str = "127.0.0.1"
    proxy_port: str = "7891"
    
    def __post_init__(self):
        """验证请求参数"""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("训练、验证、测试集比例之和必须等于1.0")
        
        if self.start_date and self.end_date:
            try:
                start = datetime.strptime(self.start_date, '%Y-%m-%d')
                end = datetime.strptime(self.end_date, '%Y-%m-%d')
                if start >= end:
                    raise ValueError("开始日期必须早于结束日期")
            except ValueError as e:
                if "time data" in str(e):
                    raise ValueError("日期格式必须为YYYY-MM-DD")
                raise


@dataclass
class DownloadResult:
    """单个数据下载结果"""
    status: str  # 'success' | 'error' | 'partial'
    symbol: str
    request: DownloadRequest
    
    # 数据结果
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[Dict[str, pd.DataFrame]] = None  # {'train': df, 'val': df, 'test': df}
    features_data: Optional[pd.DataFrame] = None
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    quality_report: Optional[Dict[str, Any]] = None
    feature_stats: Optional[Dict[str, Any]] = None
    
    # 文件路径
    saved_files: Optional[Dict[str, str]] = None
    output_dir: Optional[str] = None
    
    # 执行信息
    processing_time: float = 0.0
    data_points: int = 0
    features_count: int = 0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def is_successful(self) -> bool:
        """检查是否成功"""
        return self.status == 'success'
    
    def has_data(self) -> bool:
        """检查是否有数据"""
        return self.raw_data is not None and len(self.raw_data) > 0


@dataclass
class MultiDownloadRequest:
    """多个数据下载请求"""
    symbols: List[str]
    data_source: DataSource = DataSource.YFINANCE
    period: Union[str, DataPeriod] = DataPeriod.MONTH_1
    interval: Union[str, DataInterval] = DataInterval.DAY_1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # 处理选项
    include_features: bool = True
    split_datasets: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # 执行选项
    concurrent: bool = True  # 并发下载
    max_workers: int = 4     # 最大并发数
    
    # 输出选项
    output_dir: Optional[str] = None
    save_data: bool = True
    file_formats: List[str] = field(default_factory=lambda: ['csv', 'pkl'])
    
    # 代理设置
    use_proxy: bool = False
    proxy_host: str = "127.0.0.1"
    proxy_port: str = "7891"
    
    def to_single_requests(self) -> List[DownloadRequest]:
        """转换为单个下载请求列表"""
        requests = []
        for symbol in self.symbols:
            request = DownloadRequest(
                symbol=symbol,
                data_source=self.data_source,
                period=self.period,
                interval=self.interval,
                start_date=self.start_date,
                end_date=self.end_date,
                include_features=self.include_features,
                split_datasets=self.split_datasets,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                output_dir=self.output_dir,
                save_data=self.save_data,
                file_formats=self.file_formats,
                use_proxy=self.use_proxy,
                proxy_host=self.proxy_host,
                proxy_port=self.proxy_port
            )
            requests.append(request)
        return requests


@dataclass
class MultiDownloadResult:
    """多个数据下载结果"""
    status: str  # 'success' | 'error' | 'partial'
    total_symbols: int
    successful_count: int
    failed_count: int
    
    # 结果详情
    results: Dict[str, DownloadResult] = field(default_factory=dict)
    successful_symbols: List[str] = field(default_factory=list)
    failed_symbols: List[str] = field(default_factory=list)
    
    # 执行信息
    total_processing_time: float = 0.0
    average_time_per_symbol: float = 0.0
    
    # 摘要文件
    summary_file: Optional[str] = None
    
    def success_rate(self) -> float:
        """成功率"""
        if self.total_symbols == 0:
            return 0.0
        return self.successful_count / self.total_symbols
    
    def is_fully_successful(self) -> bool:
        """是否全部成功"""
        return self.failed_count == 0 and self.successful_count > 0


@dataclass
class BatchDownloadRequest:
    """批量数据下载请求（大数据分批处理）"""
    symbol: str
    data_source: DataSource = DataSource.YFINANCE
    period: Union[str, DataPeriod] = DataPeriod.MAX  # 通常是大数据量
    interval: Union[str, DataInterval] = DataInterval.MINUTE_1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # 批次配置
    enable_batch: bool = True
    batch_threshold_days: int = 365     # 超过多少天启用批次下载
    batch_size_days: Optional[int] = None  # 批次大小（天）
    resume_download: bool = True        # 断点续传
    
    # 处理选项
    include_features: bool = True
    split_datasets: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # 输出选项
    output_dir: Optional[str] = None
    save_data: bool = True
    file_formats: List[str] = field(default_factory=lambda: ['csv', 'pkl'])
    
    # 代理设置
    use_proxy: bool = False
    proxy_host: str = "127.0.0.1"
    proxy_port: str = "7891"


@dataclass
class BatchDownloadResult:
    """批量数据下载结果"""
    status: str  # 'success' | 'error' | 'partial'
    symbol: str
    request: BatchDownloadRequest
    
    # 批次信息
    total_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    
    # 数据结果
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[Dict[str, pd.DataFrame]] = None
    features_data: Optional[pd.DataFrame] = None
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    quality_report: Optional[Dict[str, Any]] = None
    feature_stats: Optional[Dict[str, Any]] = None
    
    # 文件路径
    saved_files: Optional[Dict[str, str]] = None
    output_dir: Optional[str] = None
    
    # 执行信息
    total_processing_time: float = 0.0
    estimated_time_minutes: float = 0.0
    data_points: int = 0
    features_count: int = 0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def completion_rate(self) -> float:
        """完成率"""
        if self.total_batches == 0:
            return 0.0
        return self.completed_batches / self.total_batches
    
    def is_successful(self) -> bool:
        """检查是否成功"""
        return self.status == 'success'


@dataclass
class RealtimeRequest:
    """实时数据请求"""
    symbol: str
    data_source: DataSource = DataSource.YFINANCE
    interval: Union[str, DataInterval] = DataInterval.MINUTE_1
    
    # 实时配置
    buffer_size: int = 1000       # 缓冲区大小
    update_frequency: int = 60    # 更新频率（秒）
    auto_save: bool = False       # 自动保存
    save_interval: int = 300      # 保存间隔（秒）
    
    # 处理选项
    include_features: bool = False  # 实时通常不需要复杂特征
    
    # 输出选项
    output_dir: Optional[str] = None
    
    # 代理设置
    use_proxy: bool = False
    proxy_host: str = "127.0.0.1"
    proxy_port: str = "7891"


@dataclass
class RealtimeStream:
    """实时数据流"""
    symbol: str
    request: RealtimeRequest
    stream_id: str
    
    # 流状态
    is_active: bool = False
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # 数据缓冲
    data_buffer: List[Dict] = field(default_factory=list)
    buffer_size: int = 1000
    
    # 统计信息
    total_updates: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    
    def add_data_point(self, data_point: Dict):
        """添加数据点"""
        self.data_buffer.append(data_point)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)  # 移除最老的数据点
        
        self.total_updates += 1
        self.last_update = datetime.now()
    
    def get_latest_data(self, count: int = 100) -> List[Dict]:
        """获取最新数据"""
        return self.data_buffer[-count:] if count <= len(self.data_buffer) else self.data_buffer.copy()
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.data_buffer.clear()