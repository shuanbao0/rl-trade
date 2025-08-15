"""
分批次数据下载器
支持基于日期范围的智能批次下载
"""

from typing import Dict, List, Optional, Tuple, Union, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import time
import pickle
from pathlib import Path

from ..sources.base import DataSource, DataPeriod
from ...utils.date_range_utils import DateRangeUtils, DateRange
from ...utils.config import Config
from ...utils.logger import get_logger


@dataclass
class BatchInfo:
    """批次信息"""
    batch_id: str
    start_date: datetime
    end_date: datetime
    status: str  # pending, downloading, completed, failed
    data_points: Optional[int] = None
    error_message: Optional[str] = None
    download_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class BatchDownloadConfig:
    """批次下载配置"""
    # 基本配置
    batch_size_days: int = 30  # 每批次天数
    max_batches_parallel: int = 1  # 最大并行批次数
    
    # 重试配置
    max_retries_per_batch: int = 3
    retry_delay_base: float = 2.0
    retry_delay_max: float = 60.0
    
    # 性能配置
    batch_delay_seconds: float = 1.0  # 批次间延迟
    memory_limit_mb: int = 512  # 内存限制
    progress_save_interval: int = 5  # 每N个批次保存一次进度
    
    # 自动批次启用条件
    auto_enable_threshold_days: int = 365
    auto_enable_min_records: int = 10000
    high_freq_threshold_days: int = 30
    high_freq_intervals: List[str] = None
    
    def __post_init__(self):
        if self.high_freq_intervals is None:
            self.high_freq_intervals = ['1m', '5m', '15m', '30m', '1h']


class BatchDownloader:
    """分批次数据下载器"""
    
    def __init__(self, config: Optional[Config] = None, data_manager=None):
        """
        初始化分批次下载器
        
        Args:
            config: 配置对象
            data_manager: 数据管理器实例
        """
        self.config = config or Config()
        self.data_manager = data_manager
        self.logger = get_logger(__name__)
        
        # 初始化下载配置
        self.download_config = self._create_download_config()
        
        # 初始化缓存目录
        self.cache_dir = Path(self.config.data.cache_dir)
        self.batch_cache_dir = self.cache_dir / "batch_downloads"
        self.batch_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 内部状态
        self.current_download_id = None
        self.batch_progress = {}
    
    def _create_download_config(self) -> BatchDownloadConfig:
        """创建下载配置"""
        config = BatchDownloadConfig()
        
        # 从配置文件获取值
        if hasattr(self.config.data, 'batch_size_days'):
            config.batch_size_days = self.config.data.batch_size_days
            
        if hasattr(self.config.data, 'auto_batch_threshold_days'):
            config.auto_enable_threshold_days = self.config.data.auto_batch_threshold_days
            
        return config
    
    def should_use_batch_download(
        self, 
        date_range: DateRange, 
        interval: str,
        estimated_records: Optional[int] = None
    ) -> bool:
        """
        判断是否应该使用批次下载
        
        Args:
            date_range: 日期范围
            interval: 数据间隔
            estimated_records: 估算记录数
            
        Returns:
            是否需要批次下载
        """
        period_days = date_range.duration_days
        config = self.download_config
        
        # 估算记录数（如果未提供）
        if estimated_records is None:
            estimated_records = self._estimate_record_count(period_days, interval)
        
        # 判断条件
        conditions = [
            # 1. 时间跨度超过阈值
            period_days > config.auto_enable_threshold_days,
            
            # 2. 估算记录数超过阈值
            estimated_records > config.auto_enable_min_records,
            
            # 3. 高频数据且时间跨度较长
            (interval in config.high_freq_intervals and 
             period_days > config.high_freq_threshold_days)
        ]
        
        return any(conditions)
    
    def download_in_batches(
        self,
        symbol: str,
        date_range: DateRange,
        interval: str = "1d",
        resume: bool = True,
        batch_size_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        分批次下载数据
        
        Args:
            symbol: 交易标的
            date_range: 日期范围
            interval: 时间间隔
            resume: 是否启用断点续传
            batch_size_days: 自定义批次大小
            
        Returns:
            完整的数据DataFrame
        """
        if not self.data_manager:
            raise ValueError("DataManager未设置，无法进行批次下载")
        
        # 生成下载ID
        download_id = self._generate_download_id(symbol, date_range, interval)
        self.current_download_id = download_id
        
        start_time = time.time()
        self.logger.info(f"开始分批次下载: {symbol}")
        self.logger.info(f"时间范围: {date_range.start_date.date()} 到 {date_range.end_date.date()}")
        self.logger.info(f"下载ID: {download_id}")
        
        try:
            # 生成批次计划
            batches = self._generate_batch_plan(date_range, batch_size_days)
            total_batches = len(batches)
            
            self.logger.info(f"生成 {total_batches} 个批次，每批次约 {batches[0].end_date - batches[0].start_date} 天")
            
            # 检查已有进度
            if resume:
                progress_data = self._load_progress(download_id)
                if progress_data:
                    self.batch_progress = progress_data
                    completed_count = len([b for b in batches if b.batch_id in progress_data and progress_data[b.batch_id]['status'] == 'completed'])
                    self.logger.info(f"发现已完成批次: {completed_count}/{total_batches}")
            
            # 执行批次下载
            all_data = []
            successful_batches = 0
            failed_batches = []
            
            for i, batch in enumerate(batches, 1):
                self.logger.info(f"处理批次 {i}/{total_batches}: {batch.batch_id}")
                
                # 检查是否已完成
                if batch.batch_id in self.batch_progress and self.batch_progress[batch.batch_id]['status'] == 'completed':
                    batch_data = self.batch_progress[batch.batch_id]['data']
                    all_data.append(batch_data)
                    successful_batches += 1
                    self.logger.info(f"批次 {i} 已完成，跳过")
                    continue
                
                # 下载批次数据
                try:
                    batch_data = self._download_single_batch(
                        symbol, batch, interval, i, total_batches
                    )
                    
                    if batch_data is not None and not batch_data.empty:
                        all_data.append(batch_data)
                        successful_batches += 1
                        
                        # 保存批次进度
                        self.batch_progress[batch.batch_id] = {
                            'status': 'completed',
                            'data': batch_data,
                            'data_points': len(batch_data),
                            'download_time': time.time() - start_time
                        }
                        
                        self.logger.info(f"批次 {i} 完成: {len(batch_data)} 条记录")
                    else:
                        failed_batches.append((i, batch.batch_id, "空数据"))
                        self.logger.warning(f"批次 {i} 返回空数据")
                        
                except Exception as e:
                    failed_batches.append((i, batch.batch_id, str(e)))
                    self.logger.error(f"批次 {i} 下载失败: {e}")
                
                # 定期保存进度
                if i % self.download_config.progress_save_interval == 0:
                    self._save_progress(download_id, self.batch_progress)
                    self.logger.info(f"进度已保存 ({successful_batches}/{total_batches})")
                
                # 批次间延迟
                if i < total_batches:
                    time.sleep(self.download_config.batch_delay_seconds)
            
            # 最终保存进度
            self._save_progress(download_id, self.batch_progress)
            
            # 合并所有数据
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_index()  # 按时间排序
                combined_data = combined_data.drop_duplicates()  # 去重
                
                total_time = time.time() - start_time
                self.logger.info(f"分批次下载完成:")
                self.logger.info(f"  - 成功批次: {successful_batches}/{total_batches}")
                self.logger.info(f"  - 失败批次: {len(failed_batches)}")
                self.logger.info(f"  - 总记录数: {len(combined_data):,}")
                self.logger.info(f"  - 总耗时: {total_time:.1f}秒")
                
                return combined_data
            else:
                raise ValueError("所有批次下载均失败")
                
        except Exception as e:
            self.logger.error(f"分批次下载失败: {e}")
            raise
        finally:
            self.current_download_id = None
    
    def _generate_batch_plan(
        self, 
        date_range: DateRange, 
        batch_size_days: Optional[int] = None
    ) -> List[BatchInfo]:
        """
        生成批次计划
        
        Args:
            date_range: 日期范围
            batch_size_days: 批次大小（天）
            
        Returns:
            批次信息列表
        """
        if batch_size_days is None:
            batch_size_days = self.download_config.batch_size_days
        
        batches = []
        current_start = date_range.start_date
        batch_index = 1
        
        while current_start < date_range.end_date:
            # 计算当前批次的结束时间
            current_end = min(
                current_start + timedelta(days=batch_size_days),
                date_range.end_date
            )
            
            # 创建批次信息
            batch_id = f"batch_{batch_index:04d}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
            
            batch = BatchInfo(
                batch_id=batch_id,
                start_date=current_start,
                end_date=current_end,
                status="pending"
            )
            
            batches.append(batch)
            
            # 准备下一个批次
            current_start = current_end + timedelta(days=1)
            batch_index += 1
        
        return batches
    
    def _download_single_batch(
        self,
        symbol: str,
        batch: BatchInfo,
        interval: str,
        batch_num: int,
        total_batches: int
    ) -> Optional[pd.DataFrame]:
        """
        下载单个批次的数据
        
        Args:
            symbol: 交易标的
            batch: 批次信息
            interval: 时间间隔
            batch_num: 批次编号
            total_batches: 总批次数
            
        Returns:
            批次数据DataFrame
        """
        max_retries = self.download_config.max_retries_per_batch
        base_delay = self.download_config.retry_delay_base
        max_delay = self.download_config.retry_delay_max
        
        batch_range = DateRange(start_date=batch.start_date, end_date=batch.end_date)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"批次 {batch_num} 尝试 {attempt + 1}/{max_retries}")
                
                # 使用DataManager获取数据
                data = self.data_manager._fetch_from_data_source_by_range(
                    symbol, batch_range, interval
                )
                
                if data is not None and not data.empty:
                    batch.status = "completed"
                    batch.data_points = len(data)
                    return data
                else:
                    self.logger.warning(f"批次 {batch_num} 返回空数据")
                    
            except Exception as e:
                error_msg = str(e).lower()
                batch.error_message = str(e)
                batch.retry_count = attempt + 1
                
                self.logger.warning(f"批次 {batch_num} 尝试 {attempt + 1} 失败: {e}")
                
                if attempt < max_retries - 1:
                    # 计算重试延迟
                    if "rate" in error_msg or "limit" in error_msg:
                        delay = min(base_delay * (3 ** attempt), max_delay)
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    self.logger.info(f"批次 {batch_num} 等待 {delay:.1f}秒 后重试...")
                    time.sleep(delay)
        
        # 所有重试都失败了
        batch.status = "failed"
        self.logger.error(f"批次 {batch_num} 所有重试均失败")
        return None
    
    def _generate_download_id(self, symbol: str, date_range: DateRange, interval: str) -> str:
        """生成下载ID"""
        start_str = date_range.start_date.strftime('%Y%m%d')
        end_str = date_range.end_date.strftime('%Y%m%d')
        return f"{symbol}_{start_str}_{end_str}_{interval}"
    
    def _load_progress(self, download_id: str) -> Optional[Dict]:
        """加载下载进度"""
        progress_file = self.batch_cache_dir / f"{download_id}_progress.pkl"
        
        if not progress_file.exists():
            return None
        
        try:
            with open(progress_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"加载进度文件失败: {e}")
            return None
    
    def _save_progress(self, download_id: str, progress_data: Dict) -> None:
        """保存下载进度"""
        progress_file = self.batch_cache_dir / f"{download_id}_progress.pkl"
        
        try:
            with open(progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
        except Exception as e:
            self.logger.warning(f"保存进度文件失败: {e}")
    
    def _estimate_record_count(self, period_days: int, interval: str) -> int:
        """估算记录数量"""
        # 每天的记录数估算
        daily_records = {
            '1m': 1440,      # 分钟级
            '5m': 288,
            '15m': 96,
            '30m': 48,
            '1h': 24,        # 小时级
            '4h': 6,
            '1d': 1,         # 日级
            '1wk': 1/7,      # 周级
            '1mo': 1/30      # 月级
        }
        
        multiplier = daily_records.get(interval, 1)
        return int(period_days * multiplier)
    
    def get_download_status(self, download_id: Optional[str] = None) -> Dict:
        """
        获取下载状态
        
        Args:
            download_id: 下载ID，None表示当前下载
            
        Returns:
            下载状态信息
        """
        if download_id is None:
            download_id = self.current_download_id
        
        if not download_id:
            return {"status": "no_active_download"}
        
        progress_data = self._load_progress(download_id)
        if not progress_data:
            return {"status": "no_progress_data", "download_id": download_id}
        
        # 统计状态
        total_batches = len(progress_data)
        completed = len([b for b in progress_data.values() if b['status'] == 'completed'])
        failed = len([b for b in progress_data.values() if b['status'] == 'failed'])
        
        return {
            "download_id": download_id,
            "status": "in_progress" if completed < total_batches else "completed",
            "total_batches": total_batches,
            "completed_batches": completed,
            "failed_batches": failed,
            "progress_percentage": (completed / total_batches * 100) if total_batches > 0 else 0,
            "total_records": sum(b.get('data_points', 0) for b in progress_data.values() if b['status'] == 'completed')
        }
    
    def cleanup_old_progress(self, days_old: int = 7) -> None:
        """
        清理旧的进度文件
        
        Args:
            days_old: 清理多少天前的文件
        """
        cutoff_time = time.time() - (days_old * 24 * 3600)
        cleaned_count = 0
        
        try:
            for progress_file in self.batch_cache_dir.glob("*_progress.pkl"):
                if progress_file.stat().st_mtime < cutoff_time:
                    progress_file.unlink()
                    cleaned_count += 1
                    
            self.logger.info(f"清理了 {cleaned_count} 个旧进度文件")
        except Exception as e:
            self.logger.warning(f"清理进度文件失败: {e}")