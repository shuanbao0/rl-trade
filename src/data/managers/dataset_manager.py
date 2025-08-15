#!/usr/bin/env python
"""
数据集管理器

提供数据保存、加载、元数据管理等功能
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, asdict

from ...utils.logger import setup_logger, get_default_log_file


@dataclass
class DatasetMetadata:
    """数据集元数据"""
    symbol: str
    data_source: str
    period: str
    interval: str
    start_date: Optional[str]
    end_date: Optional[str]
    
    # 数据信息
    total_records: int
    total_features: int
    feature_categories: Dict[str, int]
    
    # 数据集划分
    train_records: int
    val_records: int
    test_records: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    
    # 质量信息
    quality_score: float
    missing_percentage: float
    
    # 文件信息
    saved_files: Dict[str, str]
    output_directory: str
    
    # 时间信息
    created_time: str
    processing_time: float
    
    # 版本信息
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """从字典创建"""
        return cls(**data)


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, base_dir: str = "datasets"):
        """
        初始化数据集管理器
        
        Args:
            base_dir: 基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            name="DatasetManager",
            level="INFO",
            log_file=get_default_log_file("dataset_manager")
        )
        
        self.logger.info(f"数据集管理器初始化完成，基础目录: {self.base_dir}")
    
    def save_dataset(self, 
                    symbol: str,
                    raw_data: Optional[pd.DataFrame] = None,
                    processed_data: Optional[Dict[str, pd.DataFrame]] = None,
                    features_data: Optional[pd.DataFrame] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    file_formats: List[str] = None,
                    custom_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        保存数据集
        
        Args:
            symbol: 数据符号
            raw_data: 原始数据
            processed_data: 处理后的数据集 (train/val/test)
            features_data: 特征数据
            metadata: 元数据
            file_formats: 文件格式列表 ['csv', 'pkl', 'parquet']
            custom_dir: 自定义目录名
            
        Returns:
            Dict[str, Any]: 保存结果
        """
        if file_formats is None:
            file_formats = ['csv', 'pkl']
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = custom_dir or f"{symbol}_{timestamp}"
        output_dir = self.base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        self.logger.info(f"开始保存数据集: {symbol} -> {output_dir}")
        
        try:
            # 保存原始数据
            if raw_data is not None:
                raw_files = self._save_dataframe(
                    raw_data, output_dir, f"{symbol}_raw", file_formats
                )
                saved_files.update({f"raw_{fmt}": path for fmt, path in raw_files.items()})
                self.logger.info(f"原始数据已保存: {len(raw_data)} 条记录")
            
            # 保存特征数据
            if features_data is not None:
                feature_files = self._save_dataframe(
                    features_data, output_dir, f"{symbol}_features", file_formats
                )
                saved_files.update({f"features_{fmt}": path for fmt, path in feature_files.items()})
                self.logger.info(f"特征数据已保存: {len(features_data)} 条记录, {len(features_data.columns)} 个特征")
            
            # 保存数据集划分
            if processed_data is not None:
                for split_name, data in processed_data.items():
                    if data is not None and len(data) > 0:
                        split_files = self._save_dataframe(
                            data, output_dir, f"{symbol}_{split_name}", file_formats
                        )
                        saved_files.update({f"{split_name}_{fmt}": path for fmt, path in split_files.items()})
                        self.logger.info(f"{split_name} 数据已保存: {len(data)} 条记录")
            
            # 创建和保存元数据
            dataset_metadata = self._create_dataset_metadata(
                symbol, raw_data, processed_data, features_data, 
                metadata, saved_files, str(output_dir)
            )
            
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_metadata.to_dict(), f, indent=2, ensure_ascii=False)
            
            saved_files['metadata'] = str(metadata_file)
            
            # 创建数据集信息文件（便于快速查看）
            info_file = output_dir / "dataset_info.txt"
            self._create_info_file(dataset_metadata, info_file)
            saved_files['info'] = str(info_file)
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'output_dir': str(output_dir),
                'saved_files': saved_files,
                'metadata': dataset_metadata.to_dict()
            }
            
            self.logger.info(f"数据集保存成功: {symbol}")
            return result
            
        except Exception as e:
            error_msg = f"数据集保存失败: {symbol}, 错误: {e}"
            self.logger.error(error_msg)
            
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e),
                'output_dir': str(output_dir) if 'output_dir' in locals() else None
            }
    
    def load_dataset(self, 
                    symbol: str,
                    data_dir: Optional[str] = None,
                    split: str = "all",
                    file_format: str = "pkl") -> Dict[str, Any]:
        """
        加载数据集
        
        Args:
            symbol: 数据符号
            data_dir: 数据目录（如果为None，会查找最新的）
            split: 数据集分割 ("all", "train", "val", "test", "raw", "features")
            file_format: 文件格式 ("pkl", "csv", "parquet")
            
        Returns:
            Dict[str, Any]: 加载结果
        """
        try:
            # 找到数据目录
            if data_dir is None:
                data_path = self._find_latest_dataset(symbol)
                if data_path is None:
                    return {
                        'status': 'error',
                        'error': f'未找到 {symbol} 的数据集'
                    }
            else:
                data_path = Path(data_dir)
                if not data_path.exists():
                    return {
                        'status': 'error',
                        'error': f'数据目录不存在: {data_dir}'
                    }
            
            self.logger.info(f"加载数据集: {symbol} from {data_path}")
            
            # 加载元数据
            metadata_file = data_path / "metadata.json"
            metadata = None
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                    metadata = DatasetMetadata.from_dict(metadata_dict)
            
            # 加载数据
            loaded_data = {}
            
            if split == "all":
                # 加载所有数据
                for data_type in ['raw', 'features', 'train', 'val', 'test']:
                    data = self._load_single_dataframe(data_path, symbol, data_type, file_format)
                    if data is not None:
                        loaded_data[data_type] = data
            else:
                # 加载指定数据
                data = self._load_single_dataframe(data_path, symbol, split, file_format)
                if data is not None:
                    loaded_data[split] = data
                else:
                    return {
                        'status': 'error',
                        'error': f'未找到 {split} 数据文件'
                    }
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'data_dir': str(data_path),
                'data': loaded_data,
                'metadata': metadata.to_dict() if metadata else None
            }
            
            total_records = sum(len(df) for df in loaded_data.values() if isinstance(df, pd.DataFrame))
            self.logger.info(f"数据集加载成功: {symbol}, 总记录数: {total_records}")
            
            return result
            
        except Exception as e:
            error_msg = f"数据集加载失败: {symbol}, 错误: {e}"
            self.logger.error(error_msg)
            
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    def list_datasets(self, symbol_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出数据集
        
        Args:
            symbol_pattern: 符号模式（支持通配符）
            
        Returns:
            List[Dict[str, Any]]: 数据集列表
        """
        datasets = []
        
        try:
            if symbol_pattern:
                # 使用模式匹配
                pattern_dirs = list(self.base_dir.glob(f"*{symbol_pattern}*"))
            else:
                # 列出所有目录
                pattern_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            
            for data_dir in pattern_dirs:
                metadata_file = data_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        dataset_info = {
                            'symbol': metadata.get('symbol', ''),
                            'directory': str(data_dir),
                            'created_time': metadata.get('created_time', ''),
                            'total_records': metadata.get('total_records', 0),
                            'total_features': metadata.get('total_features', 0),
                            'quality_score': metadata.get('quality_score', 0),
                            'data_source': metadata.get('data_source', ''),
                            'period': metadata.get('period', ''),
                            'interval': metadata.get('interval', ''),
                        }
                        datasets.append(dataset_info)
                        
                    except Exception as e:
                        self.logger.warning(f"读取元数据失败: {metadata_file}, 错误: {e}")
            
            # 按创建时间排序
            datasets.sort(key=lambda x: x['created_time'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"列出数据集失败: {e}")
        
        return datasets
    
    def delete_dataset(self, symbol: str, data_dir: Optional[str] = None, confirm: bool = False) -> Dict[str, Any]:
        """
        删除数据集
        
        Args:
            symbol: 数据符号
            data_dir: 数据目录（如果为None，会删除最新的）
            confirm: 确认删除
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        if not confirm:
            return {
                'status': 'error',
                'error': '需要确认删除操作 (confirm=True)'
            }
        
        try:
            # 找到数据目录
            if data_dir is None:
                data_path = self._find_latest_dataset(symbol)
                if data_path is None:
                    return {
                        'status': 'error',
                        'error': f'未找到 {symbol} 的数据集'
                    }
            else:
                data_path = Path(data_dir)
                if not data_path.exists():
                    return {
                        'status': 'error',
                        'error': f'数据目录不存在: {data_dir}'
                    }
            
            # 删除目录
            import shutil
            shutil.rmtree(data_path)
            
            self.logger.info(f"数据集已删除: {symbol} at {data_path}")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'deleted_dir': str(data_path)
            }
            
        except Exception as e:
            error_msg = f"数据集删除失败: {symbol}, 错误: {e}"
            self.logger.error(error_msg)
            
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    def get_dataset_info(self, symbol: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            symbol: 数据符号
            data_dir: 数据目录
            
        Returns:
            Dict[str, Any]: 数据集信息
        """
        try:
            # 找到数据目录
            if data_dir is None:
                data_path = self._find_latest_dataset(symbol)
                if data_path is None:
                    return {
                        'status': 'error',
                        'error': f'未找到 {symbol} 的数据集'
                    }
            else:
                data_path = Path(data_dir)
            
            # 读取元数据
            metadata_file = data_path / "metadata.json"
            if not metadata_file.exists():
                return {
                    'status': 'error',
                    'error': '元数据文件不存在'
                }
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'data_dir': str(data_path),
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    # 私有辅助方法
    
    def _save_dataframe(self, df: pd.DataFrame, output_dir: Path, 
                       filename: str, file_formats: List[str]) -> Dict[str, str]:
        """保存DataFrame到不同格式"""
        saved_files = {}
        
        for fmt in file_formats:
            if fmt == 'csv':
                file_path = output_dir / f"{filename}.csv"
                df.to_csv(file_path, index=True)
                saved_files['csv'] = str(file_path)
                
            elif fmt == 'pkl':
                file_path = output_dir / f"{filename}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(df, f)
                saved_files['pkl'] = str(file_path)
                
            elif fmt == 'parquet':
                file_path = output_dir / f"{filename}.parquet"
                df.to_parquet(file_path, index=True)
                saved_files['parquet'] = str(file_path)
        
        return saved_files
    
    def _load_single_dataframe(self, data_path: Path, symbol: str, 
                              data_type: str, file_format: str) -> Optional[pd.DataFrame]:
        """加载单个DataFrame"""
        filename = f"{symbol}_{data_type}.{file_format}"
        file_path = data_path / filename
        
        if not file_path.exists():
            return None
        
        try:
            if file_format == 'csv':
                return pd.read_csv(file_path, index_col=0)
            elif file_format == 'pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif file_format == 'parquet':
                return pd.read_parquet(file_path)
            else:
                self.logger.warning(f"不支持的文件格式: {file_format}")
                return None
                
        except Exception as e:
            self.logger.error(f"加载文件失败: {file_path}, 错误: {e}")
            return None
    
    def _find_latest_dataset(self, symbol: str) -> Optional[Path]:
        """查找最新的数据集目录"""
        symbol_dirs = list(self.base_dir.glob(f"{symbol}_*"))
        
        if not symbol_dirs:
            return None
        
        # 按修改时间排序，返回最新的
        symbol_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return symbol_dirs[0]
    
    def _create_dataset_metadata(self, symbol: str,
                                raw_data: Optional[pd.DataFrame],
                                processed_data: Optional[Dict[str, pd.DataFrame]],
                                features_data: Optional[pd.DataFrame],
                                metadata: Optional[Dict[str, Any]],
                                saved_files: Dict[str, str],
                                output_dir: str) -> DatasetMetadata:
        """创建数据集元数据"""
        
        # 从现有元数据获取信息，或使用默认值
        meta = metadata or {}
        
        # 计算记录数
        total_records = len(raw_data) if raw_data is not None else 0
        total_features = len(features_data.columns) if features_data is not None else 0
        
        train_records = len(processed_data.get('train', [])) if processed_data else 0
        val_records = len(processed_data.get('val', [])) if processed_data else 0
        test_records = len(processed_data.get('test', [])) if processed_data else 0
        
        return DatasetMetadata(
            symbol=symbol,
            data_source=meta.get('data_source', 'unknown'),
            period=str(meta.get('period', 'unknown')),
            interval=str(meta.get('interval', 'unknown')),
            start_date=meta.get('start_date'),
            end_date=meta.get('end_date'),
            
            total_records=total_records,
            total_features=total_features,
            feature_categories=meta.get('feature_categories', {}),
            
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            train_ratio=meta.get('train_ratio', 0.7),
            val_ratio=meta.get('val_ratio', 0.2),
            test_ratio=meta.get('test_ratio', 0.1),
            
            quality_score=meta.get('quality_score', 0),
            missing_percentage=meta.get('missing_percentage', 0),
            
            saved_files=saved_files,
            output_directory=output_dir,
            
            created_time=datetime.now().isoformat(),
            processing_time=meta.get('processing_time', 0),
        )
    
    def _create_info_file(self, metadata: DatasetMetadata, info_file: Path):
        """创建数据集信息文件"""
        info_content = f"""数据集信息
===========

基本信息:
- 符号: {metadata.symbol}
- 数据源: {metadata.data_source}
- 时间周期: {metadata.period}
- 数据间隔: {metadata.interval}
- 创建时间: {metadata.created_time}

数据统计:
- 总记录数: {metadata.total_records:,}
- 总特征数: {metadata.total_features}
- 训练集: {metadata.train_records:,} 条记录 ({metadata.train_ratio:.1%})
- 验证集: {metadata.val_records:,} 条记录 ({metadata.val_ratio:.1%})
- 测试集: {metadata.test_records:,} 条记录 ({metadata.test_ratio:.1%})

数据质量:
- 质量评分: {metadata.quality_score:.1f}%
- 缺失值比例: {metadata.missing_percentage:.2f}%

特征分布:
"""
        
        for category, count in metadata.feature_categories.items():
            info_content += f"- {category}: {count} 个特征\n"
        
        info_content += f"""
保存的文件:
"""
        for file_type, file_path in metadata.saved_files.items():
            info_content += f"- {file_type}: {Path(file_path).name}\n"
        
        info_content += f"""
处理时间: {metadata.processing_time:.2f} 秒
版本: {metadata.version}
"""
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(info_content)


# 全局数据集管理器实例
_manager_instance = None

def get_dataset_manager(base_dir: str = "datasets") -> DatasetManager:
    """获取数据集管理器单例"""
    global _manager_instance
    if _manager_instance is None or _manager_instance.base_dir != Path(base_dir):
        _manager_instance = DatasetManager(base_dir)
    return _manager_instance