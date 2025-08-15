"""
智能时间范围建议器
根据数据源、交易标的、间隔等因素提供最优时间范围建议
"""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..sources.base import DataSource, DataPeriod
from .time_compatibility_checker import get_time_compatibility_checker
from ...utils.date_range_utils import DateRangeUtils, DateRange


@dataclass
class TimeRangeSuggestion:
    """时间范围建议"""
    # 建议的时间范围
    period: Optional[DataPeriod]
    start_date: Optional[str]  # YYYY-MM-DD格式
    end_date: Optional[str]    # YYYY-MM-DD格式
    interval: str
    
    # 建议理由和信息
    reason: str
    confidence: float  # 0.0-1.0，建议的置信度
    data_source: DataSource
    estimated_data_points: int
    
    # 性能预期
    performance_impact: str  # low, medium, high
    download_time_estimate: str  # 预估下载时间
    
    # 替代建议
    alternatives: List[Dict] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class SmartTimeAdvisor:
    """智能时间范围建议器"""
    
    def __init__(self):
        """初始化建议器"""
        self.compatibility_checker = get_time_compatibility_checker()
        self.date_utils = DateRangeUtils()
        
        # 预定义建议模板
        self.suggestion_templates = self._initialize_suggestion_templates()
    
    def _initialize_suggestion_templates(self) -> Dict:
        """初始化建议模板"""
        return {
            # 不同用例的推荐时间范围
            "backtesting": {
                "min_period": DataPeriod.YEAR_1,
                "recommended_period": DataPeriod.YEAR_2,
                "max_period": DataPeriod.YEAR_5,
                "preferred_interval": "1d"
            },
            "model_training": {
                "min_period": DataPeriod.YEAR_2,
                "recommended_period": DataPeriod.YEAR_5,
                "max_period": DataPeriod.MAX,
                "preferred_interval": "1d"
            },
            "quick_analysis": {
                "min_period": DataPeriod.MONTH_3,
                "recommended_period": DataPeriod.MONTH_6,
                "max_period": DataPeriod.YEAR_1,
                "preferred_interval": "1d"
            },
            "intraday_analysis": {
                "min_period": DataPeriod.DAYS_7,
                "recommended_period": DataPeriod.MONTH_1,
                "max_period": DataPeriod.MONTH_3,
                "preferred_interval": "1h"
            },
            "long_term_research": {
                "min_period": DataPeriod.YEAR_5,
                "recommended_period": DataPeriod.YEAR_10,
                "max_period": DataPeriod.MAX,
                "preferred_interval": "1d"
            }
        }
    
    def suggest_optimal_time_range(
        self,
        symbol: str,
        data_source: Optional[Union[DataSource, str]] = None,
        use_case: str = "backtesting",
        interval: Optional[str] = None,
        max_data_points: Optional[int] = None,
        prefer_recent_data: bool = True
    ) -> TimeRangeSuggestion:
        """
        建议最优时间范围
        
        Args:
            symbol: 交易标的
            data_source: 数据源（None表示自动选择）
            use_case: 使用场景（backtesting, model_training, quick_analysis等）
            interval: 时间间隔（None表示使用推荐间隔）
            max_data_points: 最大数据点数量限制
            prefer_recent_data: 是否偏好最新数据
            
        Returns:
            TimeRangeSuggestion: 时间范围建议
        """
        # 1. 选择最优数据源
        if data_source is None:
            optimal_sources = self._select_optimal_data_source(symbol, use_case)
            if not optimal_sources:
                return self._create_fallback_suggestion("无可用数据源")
            data_source = optimal_sources[0][0]
        elif isinstance(data_source, str):
            try:
                data_source = DataSource.from_string(data_source)
            except ValueError:
                return self._create_fallback_suggestion(f"无效的数据源: {data_source}")
        
        # 2. 获取使用场景模板
        template = self.suggestion_templates.get(use_case, self.suggestion_templates["backtesting"])
        
        # 3. 确定推荐间隔
        if interval is None:
            interval = template["preferred_interval"]
        
        # 4. 根据数据源能力调整建议
        source_info = self.compatibility_checker.get_source_info(data_source)
        if not source_info:
            return self._create_fallback_suggestion(f"不支持的数据源: {data_source}")
        
        # 5. 计算推荐时间范围
        suggested_period = self._calculate_optimal_period(
            template, source_info, max_data_points, interval
        )
        
        # 6. 生成具体日期范围
        date_range = self._generate_date_range(
            suggested_period, prefer_recent_data, source_info
        )
        
        # 7. 验证兼容性
        compatibility_result = self.compatibility_checker.check_compatibility(
            data_source, symbol, 
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            interval=interval
        )
        
        # 8. 根据兼容性结果调整建议
        if not compatibility_result.get("compatible", False):
            return self._adjust_for_compatibility_issues(
                symbol, data_source, compatibility_result, template, interval
            )
        
        # 9. 计算性能指标
        estimated_points = self._estimate_data_points(date_range, interval)
        performance_impact = self._assess_performance_impact(estimated_points)
        download_time = self._estimate_download_time(estimated_points, data_source)
        
        # 10. 生成替代建议
        alternatives = self._generate_alternatives(
            symbol, data_source, template, interval, date_range
        )
        
        # 11. 计算置信度
        confidence = self._calculate_confidence(
            compatibility_result, source_info, estimated_points, max_data_points
        )
        
        return TimeRangeSuggestion(
            period=None,  # 使用具体日期范围
            start_date=date_range.start_date.strftime("%Y-%m-%d"),
            end_date=date_range.end_date.strftime("%Y-%m-%d"),
            interval=interval,
            reason=self._generate_reason(use_case, data_source, date_range, estimated_points),
            confidence=confidence,
            data_source=data_source,
            estimated_data_points=estimated_points,
            alternatives=alternatives,
            performance_impact=performance_impact,
            download_time_estimate=download_time
        )
    
    def suggest_for_period_constraint(
        self,
        symbol: str,
        min_period: Union[str, DataPeriod],
        max_period: Union[str, DataPeriod],
        data_source: Optional[Union[DataSource, str]] = None,
        interval: str = "1d"
    ) -> TimeRangeSuggestion:
        """
        在指定周期约束下建议时间范围
        
        Args:
            symbol: 交易标的
            min_period: 最小周期
            max_period: 最大周期
            data_source: 数据源
            interval: 时间间隔
            
        Returns:
            TimeRangeSuggestion: 时间范围建议
        """
        # 转换周期为DataPeriod
        if isinstance(min_period, str):
            try:
                min_period = DataPeriod.from_string(min_period)
            except ValueError:
                return self._create_fallback_suggestion(f"无效的最小周期: {min_period}")
        
        if isinstance(max_period, str):
            try:
                max_period = DataPeriod.from_string(max_period)
            except ValueError:
                return self._create_fallback_suggestion(f"无效的最大周期: {max_period}")
        
        # 选择数据源
        if data_source is None:
            optimal_sources = self._select_optimal_data_source(symbol, "backtesting")
            if not optimal_sources:
                return self._create_fallback_suggestion("无可用数据源")
            data_source = optimal_sources[0][0]
        elif isinstance(data_source, str):
            try:
                data_source = DataSource.from_string(data_source)
            except ValueError:
                return self._create_fallback_suggestion(f"无效的数据源: {data_source}")
        
        # 在约束范围内选择最优周期
        optimal_period = self._select_period_in_range(min_period, max_period)
        
        # 生成日期范围
        date_range = self.date_utils.convert_period_to_date_range(optimal_period)
        
        # 检查兼容性
        compatibility_result = self.compatibility_checker.check_compatibility(
            data_source, symbol,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            interval=interval
        )
        
        if not compatibility_result.get("compatible", False):
            # 尝试调整到兼容范围
            adjusted_range = self._adjust_range_for_source_limits(
                data_source, date_range, min_period, max_period
            )
            if adjusted_range:
                date_range = adjusted_range
            else:
                return self._create_fallback_suggestion("无法在指定约束下找到兼容的时间范围")
        
        # 计算性能指标
        estimated_points = self._estimate_data_points(date_range, interval)
        performance_impact = self._assess_performance_impact(estimated_points)
        download_time = self._estimate_download_time(estimated_points, data_source)
        
        return TimeRangeSuggestion(
            period=optimal_period,
            start_date=date_range.start_date.strftime("%Y-%m-%d"),
            end_date=date_range.end_date.strftime("%Y-%m-%d"),
            interval=interval,
            reason=f"在约束范围({min_period.value}-{max_period.value})内选择的最优周期",
            confidence=0.8,
            data_source=data_source,
            estimated_data_points=estimated_points,
            performance_impact=performance_impact,
            download_time_estimate=download_time
        )
    
    def suggest_for_performance_constraint(
        self,
        symbol: str,
        max_data_points: int,
        data_source: Optional[Union[DataSource, str]] = None,
        prefer_longer_history: bool = True
    ) -> TimeRangeSuggestion:
        """
        在性能约束下建议时间范围
        
        Args:
            symbol: 交易标的
            max_data_points: 最大数据点数量
            data_source: 数据源
            prefer_longer_history: 是否偏好更长历史数据
            
        Returns:
            TimeRangeSuggestion: 时间范围建议
        """
        # 选择数据源
        if data_source is None:
            optimal_sources = self._select_optimal_data_source(symbol, "quick_analysis")
            if not optimal_sources:
                return self._create_fallback_suggestion("无可用数据源")
            data_source = optimal_sources[0][0]
        elif isinstance(data_source, str):
            try:
                data_source = DataSource.from_string(data_source)
            except ValueError:
                return self._create_fallback_suggestion(f"无效的数据源: {data_source}")
        
        # 根据性能约束选择间隔和时间范围
        optimal_config = self._optimize_for_performance(
            max_data_points, prefer_longer_history
        )
        
        # 生成日期范围
        if prefer_longer_history:
            # 优先更长历史，使用较大间隔
            date_range = self.date_utils.convert_period_to_date_range(optimal_config["period"])
        else:
            # 优先精细粒度，使用较短时间范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=optimal_config["days"])
            date_range = DateRange(start_date=start_date, end_date=end_date)
        
        # 验证兼容性
        compatibility_result = self.compatibility_checker.check_compatibility(
            data_source, symbol,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            interval=optimal_config["interval"]
        )
        
        if not compatibility_result.get("compatible", False):
            return self._create_fallback_suggestion("在性能约束下无法找到兼容的配置")
        
        estimated_points = self._estimate_data_points(date_range, optimal_config["interval"])
        
        return TimeRangeSuggestion(
            period=optimal_config.get("period"),
            start_date=date_range.start_date.strftime("%Y-%m-%d"),
            end_date=date_range.end_date.strftime("%Y-%m-%d"),
            interval=optimal_config["interval"],
            reason=f"在{max_data_points:,}数据点约束下的最优配置",
            confidence=0.9,
            data_source=data_source,
            estimated_data_points=estimated_points,
            performance_impact="low",
            download_time_estimate=self._estimate_download_time(estimated_points, data_source)
        )
    
    def _select_optimal_data_source(self, symbol: str, use_case: str) -> List[Tuple[DataSource, float]]:
        """选择最优数据源"""
        template = self.suggestion_templates.get(use_case, self.suggestion_templates["backtesting"])
        
        # 使用推荐周期获取最优数据源
        return self.compatibility_checker.get_optimal_sources(
            symbol=symbol,
            period=template["recommended_period"],
            interval=template["preferred_interval"]
        )
    
    def _calculate_optimal_period(
        self, 
        template: Dict, 
        source_info, 
        max_data_points: Optional[int], 
        interval: str
    ) -> DataPeriod:
        """计算最优周期"""
        recommended = template["recommended_period"]
        
        # 如果有数据点限制，检查是否需要调整
        if max_data_points:
            estimated_points = self._estimate_data_points_for_period(recommended, interval)
            if estimated_points > max_data_points:
                # 降级到较小的周期
                if template["min_period"]:
                    return template["min_period"]
        
        # 检查数据源历史限制
        if source_info and source_info.max_historical_years:
            max_years = source_info.max_historical_years
            period_years = self._period_to_years(recommended)
            if period_years > max_years:
                # 降级到数据源支持的最大周期
                return self._years_to_period(max_years)
        
        return recommended
    
    def _generate_date_range(
        self, 
        period: DataPeriod, 
        prefer_recent: bool, 
        source_info
    ) -> DateRange:
        """生成具体日期范围"""
        date_range = self.date_utils.convert_period_to_date_range(period)
        
        # 检查数据源最早可用日期
        if source_info and source_info.min_date:
            min_available = datetime.strptime(source_info.min_date, "%Y-%m-%d")
            if date_range.start_date < min_available:
                # 调整开始日期
                date_range = DateRange(
                    start_date=min_available,
                    end_date=date_range.end_date
                )
        
        return date_range
    
    def _adjust_for_compatibility_issues(
        self,
        symbol: str,
        data_source: DataSource,
        compatibility_result: Dict,
        template: Dict,
        interval: str
    ) -> TimeRangeSuggestion:
        """根据兼容性问题调整建议"""
        errors = compatibility_result.get("errors", [])
        
        # 尝试使用最小周期
        min_period = template["min_period"]
        date_range = self.date_utils.convert_period_to_date_range(min_period)
        
        # 再次检查兼容性
        retry_result = self.compatibility_checker.check_compatibility(
            data_source, symbol,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            interval=interval
        )
        
        if retry_result.get("compatible", False):
            estimated_points = self._estimate_data_points(date_range, interval)
            return TimeRangeSuggestion(
                period=min_period,
                start_date=date_range.start_date.strftime("%Y-%m-%d"),
                end_date=date_range.end_date.strftime("%Y-%m-%d"),
                interval=interval,
                reason=f"由于兼容性问题调整到最小周期: {'; '.join(errors[:2])}",
                confidence=0.6,
                data_source=data_source,
                estimated_data_points=estimated_points,
                performance_impact=self._assess_performance_impact(estimated_points),
                download_time_estimate=self._estimate_download_time(estimated_points, data_source)
            )
        else:
            return self._create_fallback_suggestion(f"兼容性检查失败: {'; '.join(errors[:2])}")
    
    def _generate_alternatives(
        self,
        symbol: str,
        primary_source: DataSource,
        template: Dict,
        interval: str,
        date_range: DateRange
    ) -> List[Dict]:
        """生成替代建议"""
        alternatives = []
        
        # 替代数据源
        optimal_sources = self.compatibility_checker.get_optimal_sources(
            symbol, 
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            interval=interval
        )
        
        for source, score in optimal_sources[:3]:  # 前3个替代源
            if source != primary_source:
                alternatives.append({
                    "type": "data_source",
                    "data_source": source.value,
                    "reason": f"替代数据源，评分: {score:.1f}",
                    "score": score
                })
        
        # 替代时间间隔
        source_info = self.compatibility_checker.get_source_info(primary_source)
        if source_info:
            for alt_interval in source_info.supported_intervals[:3]:
                if alt_interval != interval:
                    estimated_points = self._estimate_data_points(date_range, alt_interval)
                    alternatives.append({
                        "type": "interval",
                        "interval": alt_interval,
                        "estimated_data_points": estimated_points,
                        "reason": f"替代时间间隔，预估{estimated_points:,}数据点"
                    })
        
        # 替代周期
        for alt_period in [template["min_period"], template["max_period"]]:
            if alt_period and alt_period != self._infer_period_from_range(date_range):
                alt_range = self.date_utils.convert_period_to_date_range(alt_period)
                estimated_points = self._estimate_data_points(alt_range, interval)
                alternatives.append({
                    "type": "period",
                    "period": alt_period.value,
                    "start_date": alt_range.start_date.strftime("%Y-%m-%d"),
                    "end_date": alt_range.end_date.strftime("%Y-%m-%d"),
                    "estimated_data_points": estimated_points,
                    "reason": f"替代周期，预估{estimated_points:,}数据点"
                })
        
        return alternatives[:5]  # 最多5个替代建议
    
    def _calculate_confidence(
        self,
        compatibility_result: Dict,
        source_info,
        estimated_points: int,
        max_data_points: Optional[int]
    ) -> float:
        """计算建议置信度"""
        confidence = 1.0
        
        # 兼容性影响
        if not compatibility_result.get("compatible", False):
            confidence *= 0.3
        elif compatibility_result.get("warnings"):
            confidence *= 0.8
        
        # 数据源质量影响
        if source_info:
            quality_scores = {"excellent": 1.0, "good": 0.9, "fair": 0.7, "poor": 0.5}
            confidence *= quality_scores.get(source_info.data_quality, 0.8)
            
            if source_info.requires_api_key:
                confidence *= 0.9
            if not source_info.is_free:
                confidence *= 0.8
        
        # 性能影响
        if max_data_points and estimated_points > max_data_points:
            confidence *= 0.6
        elif estimated_points > 100000:
            confidence *= 0.8
        
        return max(0.1, min(1.0, confidence))
    
    def _generate_reason(
        self,
        use_case: str,
        data_source: DataSource,
        date_range: DateRange,
        estimated_points: int
    ) -> str:
        """生成建议理由"""
        source_info = self.compatibility_checker.get_source_info(data_source)
        
        reasons = [
            f"针对{use_case}场景的最优配置",
            f"使用{source_info.name if source_info else data_source.value}数据源",
            f"时间范围: {date_range.duration_days}天",
            f"预估数据点: {estimated_points:,}个"
        ]
        
        if source_info:
            if source_info.is_free:
                reasons.append("免费数据源")
            if source_info.data_quality == "excellent":
                reasons.append("高质量数据")
        
        return "; ".join(reasons)
    
    # 辅助方法
    def _estimate_data_points(self, date_range: DateRange, interval: str) -> int:
        """估算数据点数量"""
        days = date_range.duration_days
        points_per_day = {
            "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "4h": 6, "1d": 1, "1wk": 1/7, "1mo": 1/30
        }
        multiplier = points_per_day.get(interval, 1)
        return int(days * multiplier)
    
    def _estimate_data_points_for_period(self, period: DataPeriod, interval: str) -> int:
        """估算指定周期的数据点数量"""
        date_range = self.date_utils.convert_period_to_date_range(period)
        return self._estimate_data_points(date_range, interval)
    
    def _assess_performance_impact(self, estimated_points: int) -> str:
        """评估性能影响"""
        if estimated_points > 100000:
            return "high"
        elif estimated_points > 10000:
            return "medium"
        else:
            return "low"
    
    def _estimate_download_time(self, estimated_points: int, data_source: DataSource) -> str:
        """估算下载时间"""
        source_info = self.compatibility_checker.get_source_info(data_source)
        base_time = estimated_points / 1000  # 基础秒数
        
        if source_info and source_info.rate_limit_per_minute:
            # 考虑频率限制
            rate_factor = 60 / source_info.rate_limit_per_minute
            base_time *= rate_factor
        
        if base_time < 10:
            return "< 10秒"
        elif base_time < 60:
            return f"约{int(base_time)}秒"
        elif base_time < 3600:
            return f"约{int(base_time/60)}分钟"
        else:
            return f"约{int(base_time/3600)}小时"
    
    def _create_fallback_suggestion(self, reason: str) -> TimeRangeSuggestion:
        """创建后备建议"""
        # 使用Yahoo Finance和1年数据作为后备
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        return TimeRangeSuggestion(
            period=DataPeriod.YEAR_1,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            reason=f"后备建议: {reason}",
            confidence=0.3,
            data_source=DataSource.YFINANCE,
            estimated_data_points=365,
            performance_impact="low",
            download_time_estimate="< 30秒"
        )
    
    def _period_to_years(self, period: DataPeriod) -> float:
        """将周期转换为年数"""
        period_days = {
            DataPeriod.DAYS_1: 1, DataPeriod.DAYS_7: 7,
            DataPeriod.MONTH_1: 30, DataPeriod.MONTH_3: 90,
            DataPeriod.MONTH_6: 180, DataPeriod.YEAR_1: 365,
            DataPeriod.YEAR_2: 730, DataPeriod.YEAR_5: 1825,
            DataPeriod.YEAR_10: 3650, DataPeriod.MAX: 7300
        }
        return period_days.get(period, 365) / 365
    
    def _years_to_period(self, years: float) -> DataPeriod:
        """将年数转换为周期"""
        if years >= 10:
            return DataPeriod.MAX
        elif years >= 5:
            return DataPeriod.YEAR_5
        elif years >= 2:
            return DataPeriod.YEAR_2
        elif years >= 1:
            return DataPeriod.YEAR_1
        else:
            return DataPeriod.MONTH_6
    
    def _select_period_in_range(self, min_period: DataPeriod, max_period: DataPeriod) -> DataPeriod:
        """在范围内选择最优周期"""
        # 简单实现：选择中间值偏大的周期
        min_days = self._period_to_years(min_period) * 365
        max_days = self._period_to_years(max_period) * 365
        target_days = (min_days + max_days * 2) / 3  # 偏向较长周期
        
        return self._years_to_period(target_days / 365)
    
    def _optimize_for_performance(self, max_data_points: int, prefer_longer_history: bool) -> Dict:
        """为性能约束优化配置"""
        if prefer_longer_history:
            # 优先历史长度，使用较大间隔
            if max_data_points >= 2000:
                return {"period": DataPeriod.YEAR_5, "interval": "1d", "days": 1825}
            elif max_data_points >= 1000:
                return {"period": DataPeriod.YEAR_2, "interval": "1d", "days": 730}
            else:
                return {"period": DataPeriod.YEAR_1, "interval": "1d", "days": 365}
        else:
            # 优先数据精度，使用较小间隔
            if max_data_points >= 10000:
                return {"period": None, "interval": "1h", "days": 417}  # ~10000小时数据点
            elif max_data_points >= 1000:
                return {"period": None, "interval": "1h", "days": 42}   # ~1000小时数据点
            else:
                return {"period": None, "interval": "1d", "days": max_data_points}
    
    def _adjust_range_for_source_limits(
        self, 
        data_source: DataSource, 
        date_range: DateRange, 
        min_period: DataPeriod, 
        max_period: DataPeriod
    ) -> Optional[DateRange]:
        """根据数据源限制调整范围"""
        source_info = self.compatibility_checker.get_source_info(data_source)
        if not source_info:
            return None
        
        start_date = date_range.start_date
        end_date = date_range.end_date
        
        # 调整到数据源支持范围
        if source_info.min_date:
            min_available = datetime.strptime(source_info.min_date, "%Y-%m-%d")
            if start_date < min_available:
                start_date = min_available
        
        if source_info.max_date:
            max_available = datetime.strptime(source_info.max_date, "%Y-%m-%d")
            if end_date > max_available:
                end_date = max_available
        
        # 确保调整后的范围仍在约束内
        adjusted_range = DateRange(start_date=start_date, end_date=end_date)
        adjusted_days = adjusted_range.duration_days
        
        min_days = self._period_to_years(min_period) * 365
        max_days = self._period_to_years(max_period) * 365
        
        if min_days <= adjusted_days <= max_days:
            return adjusted_range
        else:
            return None
    
    def _infer_period_from_range(self, date_range: DateRange) -> Optional[DataPeriod]:
        """从日期范围推断周期"""
        days = date_range.duration_days
        
        if days <= 7:
            return DataPeriod.DAYS_7
        elif days <= 30:
            return DataPeriod.MONTH_1
        elif days <= 90:
            return DataPeriod.MONTH_3
        elif days <= 180:
            
            return DataPeriod.MONTH_6
        elif days <= 365:
            return DataPeriod.YEAR_1
        elif days <= 730:
            return DataPeriod.YEAR_2
        elif days <= 1825:
            return DataPeriod.YEAR_5
        else:
            return DataPeriod.MAX


# 全局实例
_smart_time_advisor = None

def get_smart_time_advisor() -> SmartTimeAdvisor:
    """获取智能时间建议器实例（单例模式）"""
    global _smart_time_advisor
    if _smart_time_advisor is None:
        _smart_time_advisor = SmartTimeAdvisor()
    return _smart_time_advisor