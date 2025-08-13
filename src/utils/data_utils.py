#!/usr/bin/env python
"""
数据处理工具模块

统一的数据加载、预处理和检查点管理功能
从train_model.py和TradingAgent中提取的通用数据处理函数
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from .logger import setup_logger, get_default_log_file


class DataProcessor:
    """
    数据处理器类
    
    提供统一的数据加载、预处理和检查点管理功能
    """
    
    def __init__(self, logger_name: str = "DataProcessor"):
        """
        初始化数据处理器
        
        Args:
            logger_name: 日志记录器名称
        """
        self.logger = setup_logger(
            name=logger_name,
            level="INFO",
            log_file=get_default_log_file("data_processor")
        )
        
    def load_training_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        加载训练数据 - 与train_model.py中的逻辑完全相同
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            Dict[str, pd.DataFrame]: 加载的数据集
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 查找数据文件
        datasets = {}
        
        for split in ['train', 'val', 'test']:
            # 优先尝试 CSV 文件，避免 pickle 兼容性问题
            csv_files = list(data_path.glob(f"*_{split}.csv"))
            pkl_files = list(data_path.glob(f"*_{split}.pkl"))
            
            if csv_files:
                csv_file = csv_files[0]  # 取第一个匹配的文件
                datasets[split] = pd.read_csv(csv_file, index_col=0)
                self.logger.info(f"加载 {split} CSV 数据: {len(datasets[split])} 条记录")
            elif pkl_files:
                pkl_file = pkl_files[0]  # 取第一个匹配的文件
                try:
                    with open(pkl_file, 'rb') as f:
                        datasets[split] = pickle.load(f)
                    self.logger.info(f"加载 {split} PKL 数据: {len(datasets[split])} 条记录")
                except Exception as e:
                    self.logger.error(f"加载 {split} PKL 文件失败: {e}")
                    # 尝试使用对应的 CSV 文件作为备选
                    csv_fallback = data_path / f"{pkl_file.stem.replace('pkl', 'csv')}.csv"
                    if csv_fallback.exists():
                        datasets[split] = pd.read_csv(csv_fallback, index_col=0)
                        self.logger.info(f"使用 CSV 备选文件加载 {split} 数据: {len(datasets[split])} 条记录")
                    else:
                        self.logger.warning(f"无法加载 {split} 数据文件")
            else:
                self.logger.warning(f"未找到 {split} 数据文件")
        
        if not datasets:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何数据文件")
        
        # 加载元数据
        metadata_file = data_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.logger.info(f"加载元数据: {metadata.get('symbol', 'Unknown')} - {metadata.get('features_data_shape', 'Unknown shape')}")
        
        return datasets
    
    def deep_preprocess_data(self, features_data: pd.DataFrame) -> pd.DataFrame:
        """
        深度预处理数据，确保完全兼容Ray RLlib
        
        从TradingAgent._deep_preprocess_data提取的逻辑，增强了对numpy.object_的处理
        
        Args:
            features_data: 原始特征数据
            
        Returns:
            pd.DataFrame: 完全清理的数据
        """
        try:
            self.logger.info("开始深度数据预处理...")
            self.logger.info(f"原始数据形状: {features_data.shape}")
            
            # 检查原始数据类型
            original_dtypes = features_data.dtypes.value_counts()
            self.logger.info(f"原始数据类型分布: {original_dtypes.to_dict()}")
            
            # 1. 创建完全新的DataFrame，避免任何引用问题
            processed = pd.DataFrame()
            
            # 2. 逐列处理，使用最严格的转换
            for col in features_data.columns:
                column_data = features_data[col]
                
                self.logger.debug(f"处理列 {col}, 原始类型: {column_data.dtype}")
                
                # 检查并处理object类型
                if column_data.dtype == 'object' or str(column_data.dtype).startswith('object'):
                    self.logger.warning(f"发现object类型列: {col}")
                    # 显示示例值用于调试
                    sample_values = column_data.head(3).tolist()
                    self.logger.warning(f"列 {col} 示例值: {sample_values}")
                    # 强制转换为数值类型
                    column_data = pd.to_numeric(column_data, errors='coerce')
                
                # 处理任何numpy.object_类型的残留
                if hasattr(column_data.dtype, 'type') and column_data.dtype.type == np.object_:
                    self.logger.warning(f"发现numpy.object_类型列: {col}")
                    column_data = pd.to_numeric(column_data, errors='coerce')
                
                # 处理NaN值
                if column_data.isna().any():
                    nan_count = column_data.isna().sum()
                    self.logger.info(f"列 {col} 有 {nan_count} 个NaN值，将填充为0")
                    column_data = column_data.fillna(0.0)
                
                # 处理无限值
                if np.isinf(column_data).any():
                    inf_count = np.isinf(column_data).sum()
                    self.logger.info(f"列 {col} 有 {inf_count} 个无限值，将替换为0")
                    column_data = column_data.replace([np.inf, -np.inf], 0.0)
                
                # 强制转换为float64并创建新数组
                try:
                    # 创建新的numpy数组，完全断开与原始数据的联系
                    new_values = np.array(column_data.values, dtype=np.float64, copy=True)
                    column_data = pd.Series(new_values, index=column_data.index, name=col)
                except (ValueError, TypeError) as e:
                    self.logger.error(f"列 {col} 转换为float64失败: {e}")
                    # 最后手段：创建零数组
                    new_values = np.zeros(len(column_data), dtype=np.float64)
                    column_data = pd.Series(new_values, index=column_data.index, name=col)
                
                # 验证转换结果
                if not np.issubdtype(column_data.dtype, np.number):
                    self.logger.error(f"列 {col} 转换失败，数据类型: {column_data.dtype}")
                    # 强制创建数值数组
                    new_values = np.zeros(len(column_data), dtype=np.float64)
                    column_data = pd.Series(new_values, index=column_data.index, name=col)
                
                # 将处理后的列添加到新DataFrame
                processed[col] = column_data
                self.logger.debug(f"列 {col} 处理完成，新类型: {column_data.dtype}")
            
            # 3. 最终验证 - 检查所有可能的object类型
            object_cols = []
            for col in processed.columns:
                dtype = processed[col].dtype
                if (dtype == 'object' or 
                    str(dtype).startswith('object') or 
                    (hasattr(dtype, 'type') and dtype.type == np.object_)):
                    object_cols.append(col)
            
            if object_cols:
                self.logger.error(f"仍有object类型列: {object_cols}")
                # 强制处理剩余的object列
                for col in object_cols:
                    self.logger.warning(f"强制重置列 {col} 为零数组")
                    processed[col] = pd.Series(
                        np.zeros(len(processed), dtype=np.float64), 
                        index=processed.index, 
                        name=col
                    )
            
            # 4. 数据完整性检查
            total_nan = processed.isna().sum().sum()
            if total_nan > 0:
                self.logger.error(f"数据中仍有 {total_nan} 个NaN值")
                processed = processed.fillna(0.0)
            
            # 检查无限值
            numeric_cols = processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                total_inf = np.isinf(processed[numeric_cols]).sum().sum()
                if total_inf > 0:
                    self.logger.error(f"数据中仍有 {total_inf} 个无限值")
                    processed[numeric_cols] = processed[numeric_cols].replace([np.inf, -np.inf], 0.0)
            
            # 5. 最终类型检查和强制转换
            final_dtypes = processed.dtypes.unique()
            self.logger.info(f"预处理后数据类型: {[str(dtype) for dtype in final_dtypes]}")
            
            # 确保所有列都是float64
            for col in processed.columns:
                if processed[col].dtype != np.float64:
                    self.logger.warning(f"强制转换列 {col} 从 {processed[col].dtype} 到 float64")
                    try:
                        processed[col] = processed[col].astype(np.float64)
                    except:
                        processed[col] = pd.Series(
                            np.zeros(len(processed), dtype=np.float64), 
                            index=processed.index
                        )
            
            # 6. 创建完全新的DataFrame确保没有任何object引用
            final_processed = pd.DataFrame(
                processed.values.astype(np.float64),
                index=processed.index,
                columns=processed.columns
            )
            
            # 7. 最终验证
            for dtype in final_processed.dtypes:
                if not np.issubdtype(dtype, np.number) or dtype == np.object_:
                    raise ValueError(f"数据预处理失败，发现非数值类型或object类型: {dtype}")
            
            self.logger.info(f"数据预处理完成: 形状 {final_processed.shape}, 所有列均为 {final_processed.dtypes.iloc[0]}")
            
            return final_processed
            
        except Exception as e:
            self.logger.error(f"深度数据预处理失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def find_best_checkpoint(self, model_path: str) -> Optional[str]:
        """
        查找最佳检查点文件
        
        Args:
            model_path: 模型基础路径
            
        Returns:
            str: 检查点文件路径，如果未找到则返回None
        """
        model_dir = Path(model_path)
        
        # 查找 training_results.json 文件
        results_file = model_dir / "training_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # 查找最佳检查点路径
                training_result = results.get('training_result', {})
                if 'best_checkpoint' in training_result:
                    best_checkpoint = training_result['best_checkpoint']
                    if os.path.exists(best_checkpoint):
                        self.logger.info(f"找到最佳检查点: {best_checkpoint}")
                        return best_checkpoint
                        
            except Exception as e:
                self.logger.error(f"读取训练结果文件失败: {e}")
        
        # 如果没有找到，尝试查找checkpoint目录
        checkpoint_dirs = list(model_dir.glob("checkpoint_*"))
        if checkpoint_dirs:
            # 选择最新的检查点
            latest_checkpoint = max(checkpoint_dirs)
            self.logger.info(f"找到最新检查点目录: {latest_checkpoint}")
            return str(latest_checkpoint)
            
        self.logger.error(f"未找到任何检查点: {model_path}")
        return None
    
    def auto_find_data_directory(self, symbol: str, base_dir: str = "datasets") -> Optional[str]:
        """
        自动查找指定股票的最新数据目录
        
        Args:
            symbol: 股票代码
            base_dir: 基础数据目录
            
        Returns:
            str: 数据目录路径，如果未找到则返回None
        """
        datasets_dir = Path(base_dir)
        if not datasets_dir.exists():
            self.logger.error(f"{base_dir} 目录不存在")
            return None
        
        # 查找指定股票的数据目录
        symbol_dirs = list(datasets_dir.glob(f"{symbol}_*"))
        if symbol_dirs:
            data_dir = str(max(symbol_dirs))  # 选择最新的
            self.logger.info(f"自动找到数据目录: {data_dir}")
            return data_dir
        else:
            self.logger.error(f"未找到 {symbol} 的数据目录")
            return None
    
    def auto_find_model_directory(self, symbol: str, base_dir: str = "models") -> Optional[str]:
        """
        自动查找指定股票的最新模型目录
        
        Args:
            symbol: 股票代码
            base_dir: 基础模型目录
            
        Returns:
            str: 模型目录路径，如果未找到则返回None
        """
        models_dir = Path(base_dir)
        if not models_dir.exists():
            self.logger.error(f"{base_dir} 目录不存在")
            return None
        
        # 查找指定股票的模型目录
        symbol_models = list(models_dir.glob(f"{symbol}_*"))
        if symbol_models:
            model_dir = str(max(symbol_models))  # 选择最新的
            self.logger.info(f"自动找到模型目录: {model_dir}")
            return model_dir
        else:
            self.logger.error(f"未找到 {symbol} 的模型目录")
            return None
    
    def validate_data_types(self, data: pd.DataFrame) -> bool:
        """
        验证数据类型是否符合Ray RLlib要求
        
        Args:
            data: 要验证的数据
            
        Returns:
            bool: 是否通过验证
        """
        try:
            # 检查object类型列
            object_cols = data.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                self.logger.error(f"发现object类型列: {object_cols}")
                return False
            
            # 检查NaN值
            total_nan = data.isna().sum().sum()
            if total_nan > 0:
                self.logger.error(f"数据中有 {total_nan} 个NaN值")
                return False
            
            # 检查无限值
            numeric_data = data.select_dtypes(include=[np.number])
            total_inf = np.isinf(numeric_data).sum().sum()
            if total_inf > 0:
                self.logger.error(f"数据中有 {total_inf} 个无限值")
                return False
            
            # 检查数据类型
            for col in data.columns:
                if not np.issubdtype(data[col].dtype, np.number):
                    self.logger.error(f"列 {col} 不是数值类型: {data[col].dtype}")
                    return False
            
            self.logger.info("数据类型验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"数据类型验证失败: {e}")
            return False


def get_data_processor(logger_name: str = "DataProcessor") -> DataProcessor:
    """
    获取数据处理器实例
    
    Args:
        logger_name: 日志记录器名称
        
    Returns:
        DataProcessor: 数据处理器实例
    """
    return DataProcessor(logger_name)


# 便捷函数 - 保持与原有接口的兼容性
def load_training_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    便捷函数：加载训练数据
    """
    processor = get_data_processor("DataLoader")
    return processor.load_training_data(data_dir)


def deep_preprocess_data(features_data: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：深度预处理数据
    """
    processor = get_data_processor("DataPreprocessor")
    return processor.deep_preprocess_data(features_data)


def find_best_checkpoint(model_path: str) -> Optional[str]:
    """
    便捷函数：查找最佳检查点
    """
    processor = get_data_processor("CheckpointFinder")
    return processor.find_best_checkpoint(model_path)


def auto_find_data_directory(symbol: str, base_dir: str = "datasets") -> Optional[str]:
    """
    便捷函数：自动查找数据目录
    """
    processor = get_data_processor("DataFinder")
    return processor.auto_find_data_directory(symbol, base_dir)


def auto_find_model_directory(symbol: str, base_dir: str = "models") -> Optional[str]:
    """
    便捷函数：自动查找模型目录
    """
    processor = get_data_processor("ModelFinder")
    return processor.auto_find_model_directory(symbol, base_dir)


def validate_data_types(data: pd.DataFrame) -> bool:
    """
    便捷函数：验证数据类型
    """
    processor = get_data_processor("DataValidator")
    return processor.validate_data_types(data)


# 额外的数据处理函数
def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    验证数据框是否包含所需列
    
    Args:
        df: 要验证的数据框
        required_columns: 必需的列名列表
        
    Returns:
        bool: 是否包含所有必需列
    """
    if df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    return len(missing_columns) == 0


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗市场数据
    
    Args:
        df: 原始市场数据
        
    Returns:
        pd.DataFrame: 清洗后的数据
    """
    cleaned_df = df.copy()
    
    # 处理无穷值
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
    # 处理负的成交量
    if 'Volume' in cleaned_df.columns:
        cleaned_df.loc[cleaned_df['Volume'] < 0, 'Volume'] = 0
    
    # 前向填充NaN值
    cleaned_df = cleaned_df.ffill().bfill()
    
    return cleaned_df


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    计算收益率
    
    Args:
        prices: 价格序列
        method: 计算方法 ('simple' 或 'log')
        
    Returns:
        pd.Series: 收益率序列
    """
    if method == 'simple':
        returns = prices.pct_change().fillna(0)
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).fillna(0)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return returns


def normalize_prices(prices: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    归一化价格
    
    Args:
        prices: 价格序列
        method: 归一化方法 ('minmax' 或 'zscore')
        
    Returns:
        pd.Series: 归一化后的价格序列
    """
    if method == 'minmax':
        min_val = prices.min()
        max_val = prices.max()
        return (prices - min_val) / (max_val - min_val)
    elif method == 'zscore':
        return (prices - prices.mean()) / prices.std()
    else:
        raise ValueError(f"Unsupported method: {method}")


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """
    检测异常值
    
    Args:
        data: 数据序列
        method: 检测方法 ('iqr' 或 'zscore')
        threshold: Z-score方法的阈值
        
    Returns:
        pd.Series: 布尔序列，True表示异常值
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    else:
        raise ValueError(f"Unsupported method: {method}")


def fill_missing_data(data: pd.Series, method: str = 'forward') -> pd.Series:
    """
    填充缺失数据
    
    Args:
        data: 数据序列
        method: 填充方法 ('forward', 'backward', 'interpolate')
        
    Returns:
        pd.Series: 填充后的数据序列
    """
    if method == 'forward':
        return data.ffill()
    elif method == 'backward':
        return data.bfill()
    elif method == 'interpolate':
        return data.interpolate()
    else:
        raise ValueError(f"Unsupported method: {method}")


def resample_data(data: pd.DataFrame, freq: str, agg_rules: dict = None) -> pd.DataFrame:
    """
    重采样数据
    
    Args:
        data: 输入数据框（需要时间索引）
        freq: 目标频率
        agg_rules: 聚合规则字典
        
    Returns:
        pd.DataFrame: 重采样后的数据
    """
    if agg_rules is None:
        agg_rules = {col: 'last' for col in data.columns}
    
    return data.resample(freq).agg(agg_rules)