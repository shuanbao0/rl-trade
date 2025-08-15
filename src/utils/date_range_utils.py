"""
时间范围工具类

提供日期计算、验证和格式化功能，支持时间范围下载
"""

from typing import Tuple, Optional, Union, Dict, Any, List
from datetime import datetime, timedelta, date
from dataclasses import dataclass
import re
import logging
from enum import Enum

from ..data.sources.base import DataSource, DataPeriod


class DateFormat(Enum):
    """支持的日期格式"""
    YYYY_MM_DD = "%Y-%m-%d"           # 2024-01-01
    YYYY_MM_DD_HH_MM = "%Y-%m-%d %H:%M"  # 2024-01-01 09:30
    YYYY_MM_DD_HH_MM_SS = "%Y-%m-%d %H:%M:%S"  # 2024-01-01 09:30:00
    ISO_8601 = "%Y-%m-%dT%H:%M:%S"    # 2024-01-01T09:30:00
    US_FORMAT = "%m/%d/%Y"            # 01/01/2024
    EUROPEAN_FORMAT = "%d/%m/%Y"      # 01/01/2024


@dataclass
class DateRange:
    """日期范围数据类"""
    start_date: datetime
    end_date: datetime
    
    def __post_init__(self):
        """验证日期范围的合理性"""
        if self.start_date >= self.end_date:
            raise ValueError(f"开始日期 ({self.start_date}) 必须早于结束日期 ({self.end_date})")
    
    @property
    def duration_days(self) -> int:
        """获取时间范围的天数"""
        return (self.end_date - self.start_date).days
    
    @property
    def duration_weeks(self) -> float:
        """获取时间范围的周数"""
        return self.duration_days / 7
    
    @property
    def duration_months(self) -> float:
        """获取时间范围的月数（近似）"""
        return self.duration_days / 30.44  # 平均每月天数
    
    @property
    def duration_years(self) -> float:
        """获取时间范围的年数（近似）"""
        return self.duration_days / 365.25  # 考虑闰年
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'duration_days': self.duration_days
        }
    
    def contains_date(self, check_date: datetime) -> bool:
        """检查指定日期是否在范围内"""
        return self.start_date <= check_date <= self.end_date
    
    def overlaps_with(self, other: 'DateRange') -> bool:
        """检查是否与另一个日期范围重叠"""
        return not (self.end_date <= other.start_date or self.start_date >= other.end_date)
    
    def get_equivalent_period(self) -> Optional[DataPeriod]:
        """获取等价的DataPeriod枚举（如果存在）"""
        days = self.duration_days
        
        # 允许一些误差（±5天）
        tolerance = 5
        
        if abs(days - 1) <= tolerance:
            return DataPeriod.DAYS_1
        elif abs(days - 7) <= tolerance:
            return DataPeriod.DAYS_7
        elif abs(days - 30) <= tolerance:
            return DataPeriod.DAYS_30
        elif abs(days - 60) <= tolerance:
            return DataPeriod.DAYS_60
        elif abs(days - 90) <= tolerance:
            return DataPeriod.DAYS_90
        elif abs(days - 180) <= tolerance:
            return DataPeriod.MONTH_6
        elif abs(days - 365) <= tolerance:
            return DataPeriod.YEAR_1
        elif abs(days - 730) <= tolerance:
            return DataPeriod.YEAR_2
        elif abs(days - 1825) <= tolerance:
            return DataPeriod.YEAR_5
        elif abs(days - 3650) <= tolerance:
            return DataPeriod.YEAR_10
        
        return None


class DateRangeUtils:
    """时间范围工具类"""
    
    @staticmethod
    def parse_date_string(date_str: str) -> datetime:
        """
        解析日期字符串为datetime对象
        
        Args:
            date_str: 日期字符串
            
        Returns:
            datetime: 解析后的日期对象
            
        Raises:
            ValueError: 如果无法解析日期字符串
        """
        # 标准化输入
        date_str = str(date_str).strip()
        
        # 支持的日期格式列表
        formats = [
            "%Y-%m-%d",           # 2024-01-01
            "%Y/%m/%d",           # 2024/01/01  
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 09:30:00
            "%Y-%m-%d %H:%M",     # 2024-01-01 09:30
            "%Y-%m-%dT%H:%M:%S",  # 2024-01-01T09:30:00
            "%m/%d/%Y",           # 01/01/2024 (US format)
            "%d/%m/%Y",           # 01/01/2024 (European format, 有歧义)
            "%Y%m%d",             # 20240101
            "%d-%m-%Y",           # 01-01-2024
        ]
        
        # 尝试解析每种格式
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # 尝试相对日期（如"today", "yesterday"）
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if date_str.lower() in ['today', 'now', '今天']:
            return today
        elif date_str.lower() in ['yesterday', '昨天']:
            return today - timedelta(days=1)
        elif date_str.lower() in ['tomorrow', '明天']:
            return today + timedelta(days=1)
        
        # 尝试相对时间表达式（如"3 days ago", "2 weeks ago"）
        relative_pattern = r'^(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*ago$'
        match = re.match(relative_pattern, date_str.lower())
        
        if match:
            number, unit = match.groups()
            number = int(number)
            
            if unit in ['day', 'days']:
                return today - timedelta(days=number)
            elif unit in ['week', 'weeks']:
                return today - timedelta(weeks=number)
            elif unit in ['month', 'months']:
                return today - timedelta(days=number * 30)  # 近似
            elif unit in ['year', 'years']:
                return today - timedelta(days=number * 365)  # 近似
        
        raise ValueError(f"无法解析日期字符串: {date_str}")
    
    @staticmethod 
    def convert_period_to_date_range(period: Union[str, DataPeriod]) -> DateRange:
        """
        将周期转换为日期范围
        
        Args:
            period: 数据周期
            
        Returns:
            DateRange: 对应的日期范围
        """
        return DateRangeUtils.create_date_range(period=period)
    
    @staticmethod
    def create_date_range(
        start_date: Union[str, datetime, None] = None,
        end_date: Union[str, datetime, None] = None,
        period: Union[str, DataPeriod, None] = None
    ) -> DateRange:
        """
        创建日期范围
        
        优先级: start_date + end_date > period
        
        Args:
            start_date: 开始日期（字符串或datetime）
            end_date: 结束日期（字符串或datetime）
            period: 时间周期（如果未指定start_date和end_date）
            
        Returns:
            DateRange: 日期范围对象
            
        Raises:
            ValueError: 如果参数无效或冲突
        """
        now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 解析输入参数
        parsed_start = None
        parsed_end = None
        
        if start_date is not None:
            if isinstance(start_date, str):
                parsed_start = DateRangeUtils.parse_date_string(start_date)
            elif isinstance(start_date, datetime):
                parsed_start = start_date
            else:
                raise ValueError(f"start_date必须是字符串或datetime，得到: {type(start_date)}")
        
        if end_date is not None:
            if isinstance(end_date, str):
                parsed_end = DateRangeUtils.parse_date_string(end_date)
            elif isinstance(end_date, datetime):
                parsed_end = end_date
            else:
                raise ValueError(f"end_date必须是字符串或datetime，得到: {type(end_date)}")
        
        # 情况1: 同时指定了开始和结束日期
        if parsed_start is not None and parsed_end is not None:
            return DateRange(parsed_start, parsed_end)
        
        # 情况2: 只指定了开始日期，使用今天作为结束日期
        if parsed_start is not None and parsed_end is None:
            return DateRange(parsed_start, now)
        
        # 情况3: 只指定了结束日期，根据period推断开始日期
        if parsed_start is None and parsed_end is not None:
            if period is None:
                # 默认使用1年周期
                period = DataPeriod.YEAR_1
            
            if isinstance(period, str):
                period = DataPeriod.from_string(period)
            
            period_days = period.to_days()
            parsed_start = parsed_end - timedelta(days=period_days)
            return DateRange(parsed_start, parsed_end)
        
        # 情况4: 都未指定，使用period创建相对于今天的范围
        if period is not None:
            if isinstance(period, str):
                period = DataPeriod.from_string(period)
            
            period_days = period.to_days()
            parsed_start = now - timedelta(days=period_days)
            parsed_end = now
            return DateRange(parsed_start, parsed_end)
        
        # 默认情况：使用1年期
        default_period = DataPeriod.YEAR_1
        period_days = default_period.to_days()
        parsed_start = now - timedelta(days=period_days)
        parsed_end = now
        return DateRange(parsed_start, parsed_end)
    
    @staticmethod
    def validate_date_range(
        date_range: DateRange,
        data_source: Optional[DataSource] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证日期范围的合理性
        
        Args:
            date_range: 要验证的日期范围
            data_source: 数据源（用于特定验证规则）
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误消息列表)
        """
        errors = []
        
        # 基本验证
        if date_range.start_date >= date_range.end_date:
            errors.append("开始日期必须早于结束日期")
        
        now = datetime.now()
        
        # 检查日期是否在合理范围内
        min_date = datetime(1900, 1, 1)  # 最早支持的日期
        max_date = now + timedelta(days=1)  # 最晚到明天
        
        if date_range.start_date < min_date:
            errors.append(f"开始日期不能早于 {min_date.strftime('%Y-%m-%d')}")
        
        if date_range.end_date > max_date:
            errors.append(f"结束日期不能晚于 {max_date.strftime('%Y-%m-%d')}")
        
        # 检查时间范围是否过长
        if date_range.duration_days > 365 * 25:  # 25年
            errors.append("时间范围不能超过25年")
        
        # 数据源特定验证
        if data_source:
            if data_source == DataSource.YFINANCE:
                # YFinance的限制
                if date_range.duration_days <= 7 and date_range.duration_days > 0:
                    # 7天内的数据，检查间隔限制
                    pass  # 1分钟数据可用
                elif date_range.duration_days <= 60:
                    pass  # 5分钟数据可用
                elif date_range.duration_days <= 730:
                    pass  # 1小时数据可用
                # 超过2年只能用日线数据
            
            elif data_source == DataSource.FXMINUTE:
                # FXMinute的限制：2000-2024年
                fx_min_date = datetime(2000, 1, 1)
                fx_max_date = datetime(2024, 12, 31)
                
                if date_range.start_date < fx_min_date:
                    errors.append(f"FXMinute数据源最早支持 {fx_min_date.strftime('%Y-%m-%d')}")
                
                if date_range.end_date > fx_max_date:
                    errors.append(f"FXMinute数据源最晚支持 {fx_max_date.strftime('%Y-%m-%d')}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def suggest_optimal_interval(
        date_range: DateRange,
        data_source: DataSource
    ) -> str:
        """
        根据日期范围和数据源推荐最佳时间间隔
        
        Args:
            date_range: 日期范围
            data_source: 数据源
            
        Returns:
            str: 推荐的时间间隔
        """
        days = date_range.duration_days
        
        if data_source == DataSource.YFINANCE:
            if days <= 7:
                return "1m"      # 7天内用1分钟
            elif days <= 60:
                return "5m"      # 60天内用5分钟
            elif days <= 730:
                return "1h"      # 2年内用1小时
            else:
                return "1d"      # 超过2年用日线
                
        elif data_source == DataSource.FXMINUTE:
            return "1m"          # FXMinute只支持1分钟
            
        elif data_source in [DataSource.TRUEFX, DataSource.OANDA]:
            if days <= 7:
                return "1m"
            elif days <= 30: 
                return "5m"
            elif days <= 180:
                return "1h"
            else:
                return "1d"
        else:
            # 默认推荐
            if days <= 7:
                return "1m"
            elif days <= 30:
                return "5m" 
            elif days <= 365:
                return "1h"
            else:
                return "1d"
    
    @staticmethod
    def format_duration(days: int) -> str:
        """
        格式化时长显示
        
        Args:
            days: 天数
            
        Returns:
            str: 格式化的时长字符串
        """
        if days == 1:
            return "1天"
        elif days < 30:
            return f"{days}天"
        elif days < 365:
            months = round(days / 30.44, 1)
            return f"{months}个月" if months != int(months) else f"{int(months)}个月"
        else:
            years = round(days / 365.25, 1)
            return f"{years}年" if years != int(years) else f"{int(years)}年"
    
    @staticmethod
    def split_date_range_for_batching(
        date_range: DateRange,
        batch_size_days: int = 30
    ) -> List[DateRange]:
        """
        将大的日期范围拆分为适合批处理的小范围
        
        Args:
            date_range: 原始日期范围
            batch_size_days: 批次大小（天）
            
        Returns:
            List[DateRange]: 拆分后的日期范围列表
        """
        ranges = []
        current_start = date_range.start_date
        
        while current_start < date_range.end_date:
            current_end = min(
                current_start + timedelta(days=batch_size_days),
                date_range.end_date
            )
            
            ranges.append(DateRange(current_start, current_end))
            current_start = current_end
        
        return ranges