"""
时间范围兼容性检查器
检查不同数据源对时间范围的支持情况
"""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..sources.base import DataSource, DataPeriod
from ...utils.date_range_utils import DateRangeUtils, DateRange


@dataclass
class DataSourceCapability:
    """数据源能力信息"""
    # 基础信息
    source: DataSource
    name: str
    description: str
    
    # 时间范围支持
    max_historical_years: int  # 最大历史数据年限
    min_date: Optional[str] = None  # 最早可用日期 (YYYY-MM-DD)
    max_date: Optional[str] = None  # 最晚可用日期 (YYYY-MM-DD，None表示实时)
    
    # 时间间隔支持
    supported_intervals: List[str] = None  # 支持的时间间隔
    default_interval: str = "1d"  # 默认间隔
    
    # 数据类型支持
    supported_asset_types: List[str] = None  # 支持的资产类型
    
    # 限制信息
    rate_limit_per_minute: Optional[int] = None  # 每分钟请求限制
    requires_api_key: bool = False  # 是否需要API密钥
    is_free: bool = True  # 是否免费
    
    # 数据质量
    data_quality: str = "good"  # excellent, good, fair, poor
    update_frequency: str = "daily"  # real-time, hourly, daily, weekly
    
    def __post_init__(self):
        if self.supported_intervals is None:
            self.supported_intervals = ["1d"]  # 默认只支持日数据
        
        if self.supported_asset_types is None:
            self.supported_asset_types = ["stock"]  # 默认只支持股票


class TimeCompatibilityChecker:
    """时间范围兼容性检查器"""
    
    def __init__(self):
        """初始化兼容性检查器"""
        self.data_source_capabilities = self._initialize_capabilities()
        self.date_utils = DateRangeUtils()
    
    def _initialize_capabilities(self) -> Dict[DataSource, DataSourceCapability]:
        """初始化数据源能力信息"""
        capabilities = {}
        
        # Yahoo Finance
        capabilities[DataSource.YFINANCE] = DataSourceCapability(
            source=DataSource.YFINANCE,
            name="Yahoo Finance",
            description="免费金融数据源，支持全球股票、ETF、指数等",
            max_historical_years=20,
            min_date="2000-01-01",
            max_date=None,  # 实时数据
            supported_intervals=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
            default_interval="1d",
            supported_asset_types=["stock", "etf", "index", "currency", "crypto"],
            rate_limit_per_minute=60,
            requires_api_key=False,
            is_free=True,
            data_quality="good",
            update_frequency="real-time"
        )
        
        # TrueFX
        capabilities[DataSource.TRUEFX] = DataSourceCapability(
            source=DataSource.TRUEFX,
            name="TrueFX",
            description="专业外汇数据源，提供实时汇率数据",
            max_historical_years=2,
            min_date="2022-01-01",
            max_date=None,
            supported_intervals=["1m", "5m", "15m", "1h", "1d"],
            default_interval="1h",
            supported_asset_types=["currency"],
            rate_limit_per_minute=120,
            requires_api_key=True,
            is_free=False,
            data_quality="excellent",
            update_frequency="real-time"
        )
        
        # OANDA
        capabilities[DataSource.OANDA] = DataSourceCapability(
            source=DataSource.OANDA,
            name="OANDA",
            description="专业外汇和差价合约数据源",
            max_historical_years=5,
            min_date="2019-01-01",
            max_date=None,
            supported_intervals=["S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5", "M10", "M15", "M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D", "W", "M"],
            default_interval="H1",
            supported_asset_types=["currency", "cfd", "metal", "commodity"],
            rate_limit_per_minute=100,
            requires_api_key=True,
            is_free=False,
            data_quality="excellent",
            update_frequency="real-time"
        )
        
        # FXMinute
        capabilities[DataSource.FXMINUTE] = DataSourceCapability(
            source=DataSource.FXMINUTE,
            name="FXMinute",
            description="外汇分钟级数据源",
            max_historical_years=10,
            min_date="2014-01-01",
            max_date=None,
            supported_intervals=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            default_interval="1m",
            supported_asset_types=["currency"],
            rate_limit_per_minute=200,
            requires_api_key=True,
            is_free=False,
            data_quality="good",
            update_frequency="real-time"
        )
        
        # HistData (示例，实际需要根据真实API调整)
        capabilities[DataSource.HISTDATA] = DataSourceCapability(
            source=DataSource.HISTDATA,
            name="HistData",
            description="历史金融数据源，提供高质量历史数据",
            max_historical_years=15,
            min_date="2009-01-01",
            max_date=None,
            supported_intervals=["1m", "1h", "1d"],
            default_interval="1d",
            supported_asset_types=["currency", "stock", "commodity"],
            rate_limit_per_minute=30,
            requires_api_key=True,
            is_free=False,
            data_quality="excellent",
            update_frequency="daily"
        )
        
        return capabilities
    
    def check_compatibility(
        self, 
        data_source: Union[DataSource, str],
        symbol: str,
        period: Optional[Union[str, DataPeriod]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> Dict[str, any]:
        """
        检查数据源对指定时间范围的兼容性
        
        Args:
            data_source: 数据源
            symbol: 交易标的
            period: 数据周期
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            
        Returns:
            Dict: 兼容性检查结果
        """
        # 转换数据源
        if isinstance(data_source, str):
            try:
                data_source = DataSource.from_string(data_source)
            except ValueError:
                return self._create_error_result(f"无效的数据源: {data_source}")
        
        # 检查数据源是否支持
        if data_source not in self.data_source_capabilities:
            return self._create_error_result(f"不支持的数据源: {data_source}")
        
        capability = self.data_source_capabilities[data_source]
        
        # 创建日期范围
        try:
            if period:
                date_range = self.date_utils.create_date_range(period=period)
            else:
                date_range = self.date_utils.create_date_range(
                    start_date=start_date, 
                    end_date=end_date
                )
        except Exception as e:
            return self._create_error_result(f"日期范围创建失败: {str(e)}")
        
        # 执行各项兼容性检查
        checks = []
        
        # 1. 检查时间间隔支持
        interval_check = self._check_interval_support(capability, interval)
        checks.append(interval_check)
        
        # 2. 检查历史数据范围
        historical_check = self._check_historical_range(capability, date_range)
        checks.append(historical_check)
        
        # 3. 检查资产类型支持
        asset_check = self._check_asset_type_support(capability, symbol)
        checks.append(asset_check)
        
        # 4. 检查数据量和性能
        performance_check = self._check_performance_impact(capability, date_range, interval)
        checks.append(performance_check)
        
        # 5. 检查API限制
        api_check = self._check_api_limitations(capability, date_range)
        checks.append(api_check)
        
        # 汇总结果
        all_passed = all(check["passed"] for check in checks)
        warnings = [check for check in checks if check.get("warning")]
        errors = [check for check in checks if not check["passed"]]
        
        return {
            "compatible": all_passed,
            "data_source": data_source.value,
            "data_source_name": capability.name,
            "date_range": {
                "start": date_range.start_date.strftime("%Y-%m-%d"),
                "end": date_range.end_date.strftime("%Y-%m-%d"),
                "days": date_range.duration_days
            },
            "interval": interval,
            "checks": checks,
            "warnings": [w["message"] for w in warnings],
            "errors": [e["message"] for e in errors],
            "recommendations": self._generate_recommendations(capability, date_range, interval, checks)
        }
    
    def _check_interval_support(self, capability: DataSourceCapability, interval: str) -> Dict:
        """检查时间间隔支持"""
        supported = interval in capability.supported_intervals
        
        return {
            "type": "interval_support",
            "passed": supported,
            "message": f"时间间隔 '{interval}' {'支持' if supported else '不支持'}",
            "details": {
                "requested_interval": interval,
                "supported_intervals": capability.supported_intervals,
                "default_interval": capability.default_interval
            }
        }
    
    def _check_historical_range(self, capability: DataSourceCapability, date_range: DateRange) -> Dict:
        """检查历史数据范围"""
        now = datetime.now()
        
        # 检查最大历史年限
        max_historical_date = now - timedelta(days=capability.max_historical_years * 365)
        too_old = date_range.start_date < max_historical_date
        
        # 检查最早可用日期
        if capability.min_date:
            min_date = datetime.strptime(capability.min_date, "%Y-%m-%d")
            before_min = date_range.start_date < min_date
        else:
            before_min = False
        
        # 检查最晚可用日期
        if capability.max_date:
            max_date = datetime.strptime(capability.max_date, "%Y-%m-%d")
            after_max = date_range.end_date > max_date
        else:
            after_max = False
        
        passed = not (too_old or before_min or after_max)
        
        issues = []
        if too_old:
            issues.append(f"开始日期超出最大历史范围({capability.max_historical_years}年)")
        if before_min:
            issues.append(f"开始日期早于最早可用日期({capability.min_date})")
        if after_max:
            issues.append(f"结束日期晚于最晚可用日期({capability.max_date})")
        
        return {
            "type": "historical_range",
            "passed": passed,
            "message": f"历史数据范围检查{'通过' if passed else '失败'}: {'; '.join(issues) if issues else '在支持范围内'}",
            "details": {
                "max_historical_years": capability.max_historical_years,
                "min_date": capability.min_date,
                "max_date": capability.max_date,
                "requested_start": date_range.start_date.strftime("%Y-%m-%d"),
                "requested_end": date_range.end_date.strftime("%Y-%m-%d"),
                "issues": issues
            }
        }
    
    def _check_asset_type_support(self, capability: DataSourceCapability, symbol: str) -> Dict:
        """检查资产类型支持"""
        # 简单的资产类型检测逻辑（实际应用中可能需要更复杂的逻辑）
        asset_type = self._detect_asset_type(symbol)
        supported = asset_type in capability.supported_asset_types
        
        return {
            "type": "asset_type_support",
            "passed": supported,
            "message": f"资产类型 '{asset_type}' {'支持' if supported else '不支持'}",
            "details": {
                "symbol": symbol,
                "detected_asset_type": asset_type,
                "supported_asset_types": capability.supported_asset_types
            },
            "warning": not supported
        }
    
    def _check_performance_impact(self, capability: DataSourceCapability, date_range: DateRange, interval: str) -> Dict:
        """检查性能影响"""
        # 估算数据点数量
        estimated_points = self._estimate_data_points(date_range, interval)
        
        # 性能评估
        if estimated_points > 100000:
            performance_level = "high_impact"
            message = f"数据量较大({estimated_points:,}个数据点)，可能影响性能"
            warning = True
        elif estimated_points > 10000:
            performance_level = "medium_impact"
            message = f"数据量中等({estimated_points:,}个数据点)，建议分批下载"
            warning = True
        else:
            performance_level = "low_impact"
            message = f"数据量较小({estimated_points:,}个数据点)，性能影响较小"
            warning = False
        
        return {
            "type": "performance_impact",
            "passed": True,
            "message": message,
            "details": {
                "estimated_data_points": estimated_points,
                "performance_level": performance_level,
                "date_range_days": date_range.duration_days,
                "interval": interval
            },
            "warning": warning
        }
    
    def _check_api_limitations(self, capability: DataSourceCapability, date_range: DateRange) -> Dict:
        """检查API限制"""
        issues = []
        
        if capability.requires_api_key:
            issues.append("需要API密钥")
        
        if not capability.is_free:
            issues.append("需要付费订阅")
        
        if capability.rate_limit_per_minute:
            # 估算请求数量
            estimated_requests = max(1, date_range.duration_days // 30)  # 假设每30天需要一个请求
            if estimated_requests > capability.rate_limit_per_minute:
                issues.append(f"可能超过频率限制(预估{estimated_requests}请求/分钟 > {capability.rate_limit_per_minute})")
        
        passed = len(issues) == 0
        
        return {
            "type": "api_limitations",
            "passed": passed,
            "message": f"API限制检查{'通过' if passed else '发现问题'}: {'; '.join(issues) if issues else '无限制'}",
            "details": {
                "requires_api_key": capability.requires_api_key,
                "is_free": capability.is_free,
                "rate_limit_per_minute": capability.rate_limit_per_minute,
                "issues": issues
            },
            "warning": not passed
        }
    
    def _detect_asset_type(self, symbol: str) -> str:
        """检测资产类型"""
        symbol = symbol.upper()
        
        # 外汇对检测
        if len(symbol) == 6 and symbol.isalpha():
            return "currency"
        
        # 加密货币检测
        crypto_suffixes = ["USD", "USDT", "BTC", "ETH"]
        if any(symbol.endswith(suffix) for suffix in crypto_suffixes):
            return "crypto"
        
        # 默认为股票
        return "stock"
    
    def _estimate_data_points(self, date_range: DateRange, interval: str) -> int:
        """估算数据点数量"""
        days = date_range.duration_days
        
        # 根据间隔估算每天的数据点数量
        points_per_day = {
            "1m": 1440,  # 分钟级
            "5m": 288,
            "15m": 96,
            "30m": 48,
            "1h": 24,    # 小时级
            "4h": 6,
            "1d": 1,     # 日级
            "1wk": 1/7,  # 周级
            "1mo": 1/30  # 月级
        }
        
        multiplier = points_per_day.get(interval, 1)  # 默认日级
        return int(days * multiplier)
    
    def _generate_recommendations(
        self, 
        capability: DataSourceCapability, 
        date_range: DateRange, 
        interval: str, 
        checks: List[Dict]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 时间间隔建议
        interval_check = next((c for c in checks if c["type"] == "interval_support"), None)
        if interval_check and not interval_check["passed"]:
            recommendations.append(f"建议使用支持的时间间隔: {', '.join(capability.supported_intervals[:3])}")
        
        # 历史数据建议
        historical_check = next((c for c in checks if c["type"] == "historical_range"), None)
        if historical_check and not historical_check["passed"]:
            if capability.min_date:
                recommendations.append(f"建议将开始日期设为 {capability.min_date} 或之后")
        
        # 性能建议
        performance_check = next((c for c in checks if c["type"] == "performance_impact"), None)
        if performance_check and performance_check.get("warning"):
            recommendations.append("建议使用分批下载功能以提高性能")
        
        # 数据源建议
        if not any(c["passed"] for c in checks if c["type"] in ["interval_support", "historical_range"]):
            recommendations.append("考虑使用其他数据源，如 Yahoo Finance (免费且支持范围广)")
        
        return recommendations
    
    def _create_error_result(self, error_message: str) -> Dict:
        """创建错误结果"""
        return {
            "compatible": False,
            "error": error_message,
            "checks": [],
            "warnings": [],
            "errors": [error_message],
            "recommendations": ["请检查输入参数并重试"]
        }
    
    def get_optimal_sources(
        self, 
        symbol: str,
        period: Optional[Union[str, DataPeriod]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> List[Tuple[DataSource, float]]:
        """
        获取最优数据源建议
        
        Args:
            symbol: 交易标的
            period: 数据周期
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            
        Returns:
            List[Tuple[DataSource, float]]: 数据源和评分列表（按评分降序）
        """
        scores = []
        
        for data_source in DataSource:
            if data_source == DataSource.AUTO:
                continue
                
            result = self.check_compatibility(
                data_source, symbol, period, start_date, end_date, interval
            )
            
            if result.get("error"):
                continue
            
            # 计算评分
            score = self._calculate_source_score(data_source, result)
            scores.append((data_source, score))
        
        # 按评分降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _calculate_source_score(self, data_source: DataSource, compatibility_result: Dict) -> float:
        """计算数据源评分"""
        if not compatibility_result.get("compatible", False):
            return 0.0
        
        capability = self.data_source_capabilities[data_source]
        score = 50.0  # 基础分
        
        # 兼容性加分
        if compatibility_result["compatible"]:
            score += 30.0
        
        # 免费加分
        if capability.is_free:
            score += 15.0
        
        # 数据质量加分
        quality_scores = {"excellent": 15, "good": 10, "fair": 5, "poor": 0}
        score += quality_scores.get(capability.data_quality, 0)
        
        # 更新频率加分
        frequency_scores = {"real-time": 10, "hourly": 8, "daily": 5, "weekly": 2}
        score += frequency_scores.get(capability.update_frequency, 0)
        
        # API密钥需求扣分
        if capability.requires_api_key:
            score -= 10.0
        
        # 警告扣分
        warnings_count = len(compatibility_result.get("warnings", []))
        score -= warnings_count * 5.0
        
        return max(0.0, min(100.0, score))
    
    def get_source_info(self, data_source: Union[DataSource, str]) -> Optional[DataSourceCapability]:
        """获取数据源详细信息"""
        if isinstance(data_source, str):
            try:
                data_source = DataSource.from_string(data_source)
            except ValueError:
                return None
        
        return self.data_source_capabilities.get(data_source)
    
    def list_supported_sources(self) -> List[DataSourceCapability]:
        """列出所有支持的数据源"""
        return list(self.data_source_capabilities.values())


# 全局实例
_time_compatibility_checker = None

def get_time_compatibility_checker() -> TimeCompatibilityChecker:
    """获取时间兼容性检查器实例（单例模式）"""
    global _time_compatibility_checker
    if _time_compatibility_checker is None:
        _time_compatibility_checker = TimeCompatibilityChecker()
    return _time_compatibility_checker