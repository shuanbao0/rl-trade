"""
数据验证服务
负责数据质量验证和数据清洗
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from ...utils.logger import get_logger


@dataclass
class DataValidationResult:
    """数据验证结果"""
    is_valid: bool
    issues: List[str]
    records_count: int
    missing_values: int
    date_range: Tuple[str, str]


class DataValidator:
    """数据验证服务"""
    
    def __init__(self, validation_rules: Optional[Dict] = None, logger=None):
        """
        初始化数据验证服务
        
        Args:
            validation_rules: 验证规则字典
            logger: 日志器
        """
        self.logger = logger or get_logger(__name__)
        
        # 默认验证规则
        self.validation_rules = validation_rules or {
            'min_records': 10,  # 最少记录数
            'max_missing_ratio': 0.05,  # 最大缺失值比例 5%
            'price_min': 0.01,  # 最小价格
            'price_max': 100000,  # 最大价格
            'volume_min': 0,  # 最小成交量
        }
    
    def validate_data(self, data: pd.DataFrame) -> DataValidationResult:
        """
        验证数据质量
        
        Args:
            data: 要验证的数据
            
        Returns:
            数据验证结果
        """
        issues = []
        
        # 基本检查
        if data is None or data.empty:
            return DataValidationResult(
                is_valid=False,
                issues=["Data is None or empty"],
                records_count=0,
                missing_values=0,
                date_range=("", "")
            )
        
        records_count = len(data)
        missing_values = data.isnull().sum().sum()
        
        # 检查记录数量
        if records_count < self.validation_rules['min_records']:
            issues.append(f"Insufficient records: {records_count} < {self.validation_rules['min_records']}")
        
        # 检查缺失值比例
        missing_ratio = missing_values / (records_count * len(data.columns))
        if missing_ratio > self.validation_rules['max_missing_ratio']:
            issues.append(f"Too many missing values: {missing_ratio:.2%} > {self.validation_rules['max_missing_ratio']:.2%}")
        
        # 检查时间连续性
        if isinstance(data.index, pd.DatetimeIndex):
            date_range = (
                data.index.min().strftime('%Y-%m-%d'),
                data.index.max().strftime('%Y-%m-%d')
            )
            
            # 检查重复日期
            duplicate_dates = data.index.duplicated()
            if duplicate_dates.any():
                issues.append(f"Duplicate dates found: {duplicate_dates.sum()} records")
        else:
            date_range = ("", "")
            issues.append("Index is not DatetimeIndex")
        
        # 检查价格数据合理性
        self._validate_price_data(data, issues)
        
        # 检查成交量数据
        self._validate_volume_data(data, issues)
        
        is_valid = len(issues) == 0
        
        return DataValidationResult(
            is_valid=is_valid,
            issues=issues,
            records_count=records_count,
            missing_values=missing_values,
            date_range=date_range
        )
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        cleaned_data = data.copy()
        
        # 删除重复行
        original_count = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        dropped_duplicates = original_count - len(cleaned_data)
        
        if dropped_duplicates > 0:
            self.logger.info(f"Removed {dropped_duplicates} duplicate records")
        
        # 处理缺失值 - 使用前向填充
        missing_before = cleaned_data.isnull().sum().sum()
        cleaned_data = cleaned_data.ffill()
        
        # 如果还有缺失值，使用后向填充
        cleaned_data = cleaned_data.bfill()
        
        missing_after = cleaned_data.isnull().sum().sum()
        
        if missing_before > missing_after:
            self.logger.info(f"Filled {missing_before - missing_after} missing values")
        
        # 确保数据类型正确
        cleaned_data = self._fix_data_types(cleaned_data)
        
        # 排序确保时间顺序
        if isinstance(cleaned_data.index, pd.DatetimeIndex):
            cleaned_data = cleaned_data.sort_index()
        
        # 数据范围清理
        cleaned_data = self._clean_data_ranges(cleaned_data)
        
        return cleaned_data
    
    def _validate_price_data(self, data: pd.DataFrame, issues: List[str]) -> None:
        """验证价格数据"""
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col in data.columns:
                col_data = pd.to_numeric(data[col], errors='coerce')
                
                # 检查负价格
                negative_prices = (col_data < 0).sum()
                if negative_prices > 0:
                    issues.append(f"Negative prices found in {col}: {negative_prices} records")
                
                # 检查极端价格
                too_low = (col_data < self.validation_rules['price_min']).sum()
                if too_low > 0:
                    issues.append(f"Prices too low in {col}: {too_low} records < {self.validation_rules['price_min']}")
                
                too_high = (col_data > self.validation_rules['price_max']).sum()
                if too_high > 0:
                    issues.append(f"Prices too high in {col}: {too_high} records > {self.validation_rules['price_max']}")
        
        # 检查OHLC逻辑关系
        self._validate_ohlc_logic(data, issues)
    
    def _validate_volume_data(self, data: pd.DataFrame, issues: List[str]) -> None:
        """验证成交量数据"""
        if 'Volume' in data.columns:
            volume_data = pd.to_numeric(data['Volume'], errors='coerce')
            
            # 检查负成交量
            negative_volume = (volume_data < self.validation_rules['volume_min']).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume found: {negative_volume} records")
            
            # 检查异常高的成交量（可能的数据错误）
            if len(volume_data) > 10:  # 至少要有10个数据点
                volume_mean = volume_data.mean()
                volume_std = volume_data.std()
                if volume_std > 0:
                    outlier_threshold = volume_mean + 5 * volume_std
                    outliers = (volume_data > outlier_threshold).sum()
                    if outliers > 0:
                        issues.append(f"Potential volume outliers: {outliers} records > {outlier_threshold:.0f}")
    
    def _validate_ohlc_logic(self, data: pd.DataFrame, issues: List[str]) -> None:
        """验证OHLC数据的逻辑关系"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        
        if all(col in data.columns for col in required_columns):
            # 转换为数值类型
            ohlc_data = data[required_columns].apply(pd.to_numeric, errors='coerce')
            
            # High应该是最高值
            invalid_high = (
                (ohlc_data['High'] < ohlc_data['Open']) |
                (ohlc_data['High'] < ohlc_data['Low']) |
                (ohlc_data['High'] < ohlc_data['Close'])
            ).sum()
            
            if invalid_high > 0:
                issues.append(f"Invalid High values: {invalid_high} records where High is not the maximum")
            
            # Low应该是最低值
            invalid_low = (
                (ohlc_data['Low'] > ohlc_data['Open']) |
                (ohlc_data['Low'] > ohlc_data['High']) |
                (ohlc_data['Low'] > ohlc_data['Close'])
            ).sum()
            
            if invalid_low > 0:
                issues.append(f"Invalid Low values: {invalid_low} records where Low is not the minimum")
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """修复数据类型"""
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _clean_data_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理数据范围异常值"""
        # 清理价格异常值
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col in data.columns:
                # 移除极端异常值（可能是数据错误）
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                
                # 如果99分位数远大于1分位数，说明可能存在异常值
                if q99 > 0 and q01 > 0 and (q99 / q01) > 100:
                    # 使用更保守的范围
                    q95 = data[col].quantile(0.95)
                    q05 = data[col].quantile(0.05)
                    
                    outlier_mask = (data[col] < q05 * 0.1) | (data[col] > q95 * 10)
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        self.logger.warning(f"Removing {outlier_count} extreme outliers from {col}")
                        # 用中位数替换极端异常值
                        median_val = data[col].median()
                        data.loc[outlier_mask, col] = median_val
        
        # 清理成交量异常值
        if 'Volume' in data.columns:
            # 负成交量设为0
            negative_mask = data['Volume'] < 0
            if negative_mask.any():
                self.logger.warning(f"Setting {negative_mask.sum()} negative volume values to 0")
                data.loc[negative_mask, 'Volume'] = 0
        
        return data
    
    def get_data_quality_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        计算数据质量评分
        
        Args:
            data: 数据DataFrame
            
        Returns:
            质量评分字典
        """
        if data is None or data.empty:
            return {
                'overall_score': 0.0,
                'completeness_score': 0.0,
                'consistency_score': 0.0,
                'accuracy_score': 0.0
            }
        
        # 完整性评分 (缺失值比例)
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        completeness_score = max(0, 1 - (missing_cells / total_cells))
        
        # 一致性评分 (重复值、时间序列连续性等)
        consistency_score = self._calculate_consistency_score(data)
        
        # 准确性评分 (数据范围合理性、OHLC逻辑等)
        accuracy_score = self._calculate_accuracy_score(data)
        
        # 总体评分 (加权平均)
        overall_score = (
            completeness_score * 0.3 +
            consistency_score * 0.3 +
            accuracy_score * 0.4
        )
        
        return {
            'overall_score': round(overall_score, 3),
            'completeness_score': round(completeness_score, 3),
            'consistency_score': round(consistency_score, 3),
            'accuracy_score': round(accuracy_score, 3)
        }
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算一致性评分"""
        score = 1.0
        
        # 重复记录扣分
        if isinstance(data.index, pd.DatetimeIndex):
            duplicate_ratio = data.index.duplicated().sum() / len(data)
            score -= duplicate_ratio * 0.5
        
        # 时间序列连续性检查（如果是时间序列数据）
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            # 检查时间序列是否有大的跳跃
            time_diffs = data.index.to_series().diff().dt.days
            if len(time_diffs) > 1:
                expected_diff = time_diffs.median()
                if expected_diff > 0:
                    large_gaps = (time_diffs > expected_diff * 3).sum()
                    gap_ratio = large_gaps / len(time_diffs)
                    score -= gap_ratio * 0.3
        
        return max(0, score)
    
    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """计算准确性评分"""
        score = 1.0
        
        # OHLC逻辑检查
        required_columns = ['Open', 'High', 'Low', 'Close']
        if all(col in data.columns for col in required_columns):
            try:
                ohlc_data = data[required_columns].apply(pd.to_numeric, errors='coerce')
                
                # 检查High和Low的逻辑关系
                invalid_high = (
                    (ohlc_data['High'] < ohlc_data['Low']) |
                    (ohlc_data['High'] < ohlc_data['Open']) |
                    (ohlc_data['High'] < ohlc_data['Close'])
                ).sum()
                
                invalid_low = (
                    (ohlc_data['Low'] > ohlc_data['High']) |
                    (ohlc_data['Low'] > ohlc_data['Open']) |
                    (ohlc_data['Low'] > ohlc_data['Close'])
                ).sum()
                
                invalid_ratio = (invalid_high + invalid_low) / (2 * len(data))
                score -= invalid_ratio
                
            except Exception:
                score -= 0.1  # 数据类型转换问题
        
        return max(0, score)