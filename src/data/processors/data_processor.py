#!/usr/bin/env python
"""
数据处理器

提供数据质量检查、特征分析、数据集划分等功能
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from ...utils.logger import setup_logger, get_default_log_file


@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_cells: int
    missing_values: int
    infinite_values: int
    missing_percentage: float
    constant_features: int
    outlier_features: int
    quality_score: float
    warnings: List[str]
    recommendations: List[str]


@dataclass
class FeatureAnalysisReport:
    """特征分析报告"""
    total_features: int
    feature_categories: Dict[str, int]
    feature_types: Dict[str, str]
    feature_ranges: Dict[str, Dict[str, float]]
    correlation_matrix: Optional[pd.DataFrame]
    top_correlations: List[Tuple[str, str, float]]


@dataclass
class DatasetSplitResult:
    """数据集划分结果"""
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    split_info: Dict[str, Any]


class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.logger = setup_logger(
            name="DataProcessor",
            level="INFO",
            log_file=get_default_log_file("data_processor")
        )
    
    def check_data_quality(self, data: pd.DataFrame, symbol: str = "") -> DataQualityReport:
        """
        检查数据质量
        
        Args:
            data: 要检查的数据
            symbol: 数据符号（用于日志）
            
        Returns:
            DataQualityReport: 数据质量报告
        """
        self.logger.info(f"开始数据质量检查: {symbol}")
        
        total_cells = data.shape[0] * data.shape[1]
        nan_count = data.isnull().sum().sum()
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        
        # 检查常量特征
        constant_features = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_features.append(col)
        
        # 检查异常值 (超过3个标准差)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            if data[col].std() > 0:  # 避免除零
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_count = (z_scores > 3).sum()
                if outlier_count > 0:
                    outliers[col] = outlier_count
        
        # 计算质量评分
        missing_percentage = round(nan_count / total_cells * 100, 2) if total_cells > 0 else 0
        quality_score = round((1 - (nan_count + inf_count) / total_cells) * 100, 2) if total_cells > 0 else 0
        
        # 生成警告和建议
        warnings = []
        recommendations = []
        
        if missing_percentage > 5:
            warnings.append(f"缺失值比例较高: {missing_percentage}%")
            recommendations.append("考虑使用插值或删除缺失值过多的行/列")
        
        if inf_count > 0:
            warnings.append(f"发现 {inf_count} 个无穷值")
            recommendations.append("处理无穷值，替换为NaN或合理的数值")
        
        if len(constant_features) > 0:
            warnings.append(f"发现 {len(constant_features)} 个常量特征")
            recommendations.append("考虑删除常量特征，它们对模型训练无用")
        
        if len(outliers) > len(numeric_cols) * 0.3:
            warnings.append("异常值较多，可能影响模型性能")
            recommendations.append("考虑使用异常值检测和处理方法")
        
        report = DataQualityReport(
            total_cells=total_cells,
            missing_values=int(nan_count),
            infinite_values=int(inf_count),
            missing_percentage=missing_percentage,
            constant_features=len(constant_features),
            outlier_features=len(outliers),
            quality_score=quality_score,
            warnings=warnings,
            recommendations=recommendations
        )
        
        self.logger.info(f"数据质量检查完成: {symbol}, 质量评分: {quality_score}")
        return report
    
    def analyze_features(self, data: pd.DataFrame, symbol: str = "", 
                        include_correlation: bool = True) -> FeatureAnalysisReport:
        """
        分析特征
        
        Args:
            data: 要分析的数据
            symbol: 数据符号（用于日志）
            include_correlation: 是否包含相关性分析
            
        Returns:
            FeatureAnalysisReport: 特征分析报告
        """
        self.logger.info(f"开始特征分析: {symbol}")
        
        # 特征类别分析
        feature_categories = self._categorize_features(data.columns)
        
        # 特征类型分析
        feature_types = {}
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                feature_types[col] = 'numeric'
            elif data[col].dtype == 'object':
                feature_types[col] = 'categorical'
            elif data[col].dtype == 'datetime64[ns]':
                feature_types[col] = 'datetime'
            else:
                feature_types[col] = str(data[col].dtype)
        
        # 数值特征范围分析
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_ranges = {}
        for col in numeric_cols:
            feature_ranges[col] = {
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'mean': float(data[col].mean()),
                'std': float(data[col].std()) if data[col].std() > 0 else 0.0,
                'median': float(data[col].median())
            }
        
        # 相关性分析
        correlation_matrix = None
        top_correlations = []
        
        if include_correlation and len(numeric_cols) > 1:
            try:
                correlation_matrix = data[numeric_cols].corr()
                
                # 找出最高的相关性（排除自相关）
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        corr_value = correlation_matrix.iloc[i, j]
                        if not np.isnan(corr_value):
                            corr_pairs.append((col1, col2, abs(corr_value)))
                
                # 按相关性排序，取前10个
                top_correlations = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:10]
                
            except Exception as e:
                self.logger.warning(f"相关性分析失败: {e}")
        
        report = FeatureAnalysisReport(
            total_features=len(data.columns),
            feature_categories=feature_categories,
            feature_types=feature_types,
            feature_ranges=feature_ranges,
            correlation_matrix=correlation_matrix,
            top_correlations=top_correlations
        )
        
        self.logger.info(f"特征分析完成: {symbol}, 总特征数: {len(data.columns)}")
        return report
    
    def split_dataset(self, data: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.1,
                     method: str = "temporal",
                     symbol: str = "") -> DatasetSplitResult:
        """
        数据集划分
        
        Args:
            data: 要划分的数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            method: 划分方法 ("temporal", "random", "stratified")
            symbol: 数据符号（用于日志）
            
        Returns:
            DatasetSplitResult: 数据集划分结果
        """
        self.logger.info(f"开始数据集划分: {symbol}, 方法: {method}")
        
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练、验证、测试集比例之和必须等于1.0")
        
        total_len = len(data)
        
        if method == "temporal":
            # 按时间顺序划分（适用于时间序列数据）
            train_end = int(total_len * train_ratio)
            val_end = int(total_len * (train_ratio + val_ratio))
            
            train_data = data.iloc[:train_end].copy()
            val_data = data.iloc[train_end:val_end].copy()
            test_data = data.iloc[val_end:].copy()
            
        elif method == "random":
            # 随机划分
            shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
            
            train_end = int(total_len * train_ratio)
            val_end = int(total_len * (train_ratio + val_ratio))
            
            train_data = shuffled_data.iloc[:train_end].copy()
            val_data = shuffled_data.iloc[train_end:val_end].copy()
            test_data = shuffled_data.iloc[val_end:].copy()
            
        else:
            raise ValueError(f"不支持的划分方法: {method}")
        
        # 创建划分信息
        split_info = {
            'method': method,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'total_samples': total_len,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'actual_ratios': {
                'train': len(train_data) / total_len,
                'val': len(val_data) / total_len,
                'test': len(test_data) / total_len
            },
            'split_time': datetime.now().isoformat()
        }
        
        result = DatasetSplitResult(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            split_info=split_info
        )
        
        self.logger.info(f"数据集划分完成: {symbol}")
        self.logger.info(f"  训练集: {len(train_data)} 条记录 ({len(train_data)/total_len:.1%})")
        self.logger.info(f"  验证集: {len(val_data)} 条记录 ({len(val_data)/total_len:.1%})")
        self.logger.info(f"  测试集: {len(test_data)} 条记录 ({len(test_data)/total_len:.1%})")
        
        return result
    
    def clean_data(self, data: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   fill_missing: bool = True,
                   fill_method: str = "forward",
                   remove_outliers: bool = False,
                   outlier_std_threshold: float = 3.0,
                   symbol: str = "") -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            data: 要清洗的数据
            remove_duplicates: 是否删除重复行
            fill_missing: 是否填充缺失值
            fill_method: 填充方法 ("forward", "backward", "mean", "median", "zero")
            remove_outliers: 是否删除异常值
            outlier_std_threshold: 异常值标准差阈值
            symbol: 数据符号（用于日志）
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        self.logger.info(f"开始数据清洗: {symbol}")
        
        cleaned_data = data.copy()
        original_shape = cleaned_data.shape
        
        # 删除重复行
        if remove_duplicates:
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            after_count = len(cleaned_data)
            if before_count != after_count:
                self.logger.info(f"删除重复行: {before_count - after_count} 行")
        
        # 填充缺失值
        if fill_missing:
            missing_before = cleaned_data.isnull().sum().sum()
            
            if fill_method == "forward":
                cleaned_data = cleaned_data.fillna(method='ffill')
            elif fill_method == "backward":
                cleaned_data = cleaned_data.fillna(method='bfill')
            elif fill_method == "mean":
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                    cleaned_data[numeric_cols].mean()
                )
            elif fill_method == "median":
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(
                    cleaned_data[numeric_cols].median()
                )
            elif fill_method == "zero":
                cleaned_data = cleaned_data.fillna(0)
            
            missing_after = cleaned_data.isnull().sum().sum()
            if missing_before != missing_after:
                self.logger.info(f"填充缺失值: {missing_before - missing_after} 个")
        
        # 删除异常值
        if remove_outliers:
            before_count = len(cleaned_data)
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if cleaned_data[col].std() > 0:
                    z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                    cleaned_data = cleaned_data[z_scores <= outlier_std_threshold]
            
            after_count = len(cleaned_data)
            if before_count != after_count:
                self.logger.info(f"删除异常值: {before_count - after_count} 行")
        
        final_shape = cleaned_data.shape
        self.logger.info(f"数据清洗完成: {symbol}, 形状变化: {original_shape} -> {final_shape}")
        
        return cleaned_data
    
    def _categorize_features(self, feature_columns: List[str]) -> Dict[str, int]:
        """
        特征分类
        
        Args:
            feature_columns: 特征列名列表
            
        Returns:
            Dict[str, int]: 各类别特征数量
        """
        categories = {
            'Price': 0,      # 价格相关
            'Volume': 0,     # 成交量相关
            'MA': 0,         # 移动平均
            'Momentum': 0,   # 动量指标
            'Volatility': 0, # 波动性指标
            'Trend': 0,      # 趋势指标
            'Statistical': 0,# 统计特征
            'Others': 0      # 其他
        }
        
        for col in feature_columns:
            col_upper = col.upper()
            if any(x in col_upper for x in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'PRICE']):
                categories['Price'] += 1
            elif 'VOLUME' in col_upper or 'OBV' in col_upper:
                categories['Volume'] += 1
            elif any(x in col_upper for x in ['SMA', 'EMA', 'WMA']):
                categories['MA'] += 1
            elif any(x in col_upper for x in ['RSI', 'MACD', 'ROC', 'MOM', 'STOCH', 'WILLIAMS', 'CCI']):
                categories['Momentum'] += 1
            elif any(x in col_upper for x in ['ATR', 'BBANDS', 'VOLATILITY', 'STD']):
                categories['Volatility'] += 1
            elif any(x in col_upper for x in ['ADX', 'DI', 'SAR', 'ICHIMOKU', 'TREND']):
                categories['Trend'] += 1
            elif any(x in col_upper for x in ['MEAN', 'MEDIAN', 'SKEW', 'KURT', 'CORR', 'Z_SCORE']):
                categories['Statistical'] += 1
            else:
                categories['Others'] += 1
        
        # 只返回非零类别
        return {k: v for k, v in categories.items() if v > 0}


# 全局数据处理器实例
_processor_instance = None

def get_data_processor() -> DataProcessor:
    """获取数据处理器单例"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DataProcessor()
    return _processor_instance