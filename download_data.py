#!/usr/bin/env python
"""
市场数据下载和预处理脚本

功能:
1. 支持多数据源 (YFinance, TrueFX, Oanda, HistData, FXMinute)
2. 批量下载市场历史数据 (股票/外汇/期货等)
3. 执行特征工程处理
4. 数据集划分(训练/验证/测试)
5. 保存处理后的数据到本地

使用示例:
  # 直接运行 (默认下载EURUSD外汇数据，FX-1-Minute-Data数据源)
  python download_data.py

  # 下载指定外汇数据，使用1小时间隔
  python download_data.py --symbol GBPUSD --period 2y --interval 1m

  # 批量下载多个外汇对，使用1分钟间隔
  python download_data.py --symbols EURUSD,GBPUSD,USDJPY --period 60d --interval 1m

  # 使用YFinance数据源下载股票数据
  python download_data.py --data-source yfinance --symbol AAPL --period 1y

  # 使用TrueFX数据源下载外汇数据
  python download_data.py --data-source truefx --symbol EURUSD --period 1mo

  # 使用Oanda数据源（需要配置文件）
  python download_data.py --data-source oanda --data-source-config oanda_config.json --symbol GBPUSD

  # 使用HistData数据源下载历史数据
  python download_data.py --data-source histdata --symbol GBPUSD --period 1y

  # 使用FX-1-Minute-Data数据源下载历史外汇数据
  python download_data.py --data-source fxminute --symbol EURUSD --period 1mo

  # 指定数据集划分比例
  python download_data.py --symbol AAPL --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

  # 使用自定义代理 (适用于yfinance等需要代理的数据源)
  python download_data.py --proxy-host 127.0.0.1 --proxy-port 7891

  # 禁用代理
  python download_data.py --no-proxy

数据源说明:
  - fxminute: FX-1-Minute-Data，本地缓存外汇数据 (默认，2000-2024年高质量数据)
  - yfinance: Yahoo Finance，免费股票数据 (支持代理)
  - truefx: TrueFX，免费外汇数据 (需要注册账户)
  - oanda: Oanda，专业外汇/CFD数据 (需要API访问)
  - histdata: HistData，历史外汇文件 (免费历史数据)

数据间隔说明:
  - 各数据源支持的间隔不同
  - fxminute: 仅支持1分钟OHLC数据，66+个交易对
  - yfinance: 1m(7天), 5m(60天), 1h(2年), 1d(历史)
  - truefx: tick级实时数据和日内数据
  - oanda: 秒级到月级多种间隔
  - histdata: 1分钟历史数据文件

代理配置说明:
  - 适用于需要代理访问的数据源 (如yfinance)
  - 默认使用 socks5://127.0.0.1:7891 代理
  - 支持通过环境变量配置: USE_PROXY, PROXY_HOST, PROXY_PORT
"""

import os
import sys
import argparse
import json
import warnings
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from src.data.data_manager import DataManager
    from src.features.feature_engineer import FeatureEngineer
    from src.utils.config import Config
    from src.utils.logger import setup_logger, get_default_log_file
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)


class DataDownloader:
    """
    数据下载和预处理类
    
    负责批量下载股票数据、特征工程和数据集划分
    """
    
    def __init__(self, config_path: Optional[str] = None, output_dir: str = "datasets"):
        """
        初始化数据下载器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        # 设置代理（如果需要）
        self._setup_proxy()
        
        # 加载配置
        self.config = Config(config_file=config_path) if config_path else Config()
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化日志系统
        self.logger = setup_logger(
            name="DataDownloader",
            level="INFO",
            log_file=get_default_log_file("data_downloader")
        )
        
        # 初始化核心组件，传递代理配置和数据源类型
        data_source_type = os.getenv('DATA_SOURCE_TYPE', 'fxminute')
        data_source_config = self._load_data_source_config()
        
        # 为FXMinute数据源提供默认配置
        if data_source_type == 'fxminute' and not data_source_config:
            data_source_config = {
                'data_directory': str(PROJECT_ROOT / 'data_cache' / 'FX-1-Minute-Data'),
                'auto_extract': True,
                'cache_extracted': True,
                'extracted_cache_dir': str(PROJECT_ROOT / 'fx_minute_cache')
            }
        
        self.data_manager = DataManager(
            config=self.config,
            data_source_type=data_source_type,
            data_source_config=data_source_config
        )
        self.feature_engineer = FeatureEngineer(self.config)
        
        # 设置代理配置到 DataManager
        self._configure_data_manager_proxy()
        
        self.logger.info("数据下载器初始化完成")
    
    def _setup_proxy(self):
        """
        设置代理配置
        """
        use_proxy = os.getenv('USE_PROXY', 'true').lower() == 'true'
        
        if use_proxy:
            proxy_host = os.getenv('PROXY_HOST', '127.0.0.1')
            proxy_port = os.getenv('PROXY_PORT', '7891')
            proxy_url = f"socks5://{proxy_host}:{proxy_port}"
            
            # 设置全局环境变量代理
            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url
            print(f"SUCCESS: 全局代理已设置: {proxy_url}")
        else:
            print("SUCCESS: 不使用代理")
    
    def _configure_data_manager_proxy(self):
        """
        为 DataManager 配置代理设置
        """
        use_proxy = os.getenv('USE_PROXY', 'true').lower() == 'true'
        
        if use_proxy:
            proxy_host = os.getenv('PROXY_HOST', '127.0.0.1')
            proxy_port = os.getenv('PROXY_PORT', '7891')
            proxy_url = f"socks5://{proxy_host}:{proxy_port}"
            
            # 配置 DataManager 代理
            if hasattr(self.data_manager, 'set_proxy'):
                self.data_manager.set_proxy(proxy_url)
                self.logger.info(f"DataManager 代理配置: {proxy_url}")
            else:
                self.logger.warning("DataManager 不支持代理配置")
        else:
            self.logger.info("DataManager 不使用代理")
    
    def _load_data_source_config(self) -> Dict:
        """
        加载数据源配置
        
        Returns:
            数据源配置字典
        """
        config_file = os.getenv('DATA_SOURCE_CONFIG')
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    import json
                    config = json.load(f)
                    self.logger.info(f"Loaded data source config from: {config_file}")
                    return config
            except Exception as e:
                self.logger.warning(f"Failed to load data source config: {e}")
        
        return {}
    
    def download_single_stock(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        下载单个股票数据并进行预处理
        
        Args:
            symbol: 股票代码
            period: 数据周期
            interval: 数据间隔
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        data_source = os.getenv('DATA_SOURCE_TYPE', 'yfinance')
        start_time = time.time()
        
        # 打印开始信息
        print(f"\n{'='*60}")
        print(f"📊 开始处理: {symbol}")
        print(f"数据源: {data_source} | 周期: {period} | 间隔: {interval}")
        print(f"数据集划分: 训练({train_ratio:.0%}) 验证({val_ratio:.0%}) 测试({test_ratio:.0%})")
        print(f"{'='*60}")
        
        self.logger.info(f"开始下载数据: {symbol} (数据源: {data_source}), 周期: {period}, 间隔: {interval}")
        
        try:
            # 验证比例
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("训练、验证、测试集比例之和必须等于1.0")
            
            # 创建进度条
            with tqdm(total=5, desc="数据处理进度", ncols=80, colour='green') as pbar:
                
                # 1. 获取原始数据
                pbar.set_description("📥 获取原始数据")
                self.logger.info(f"从 {data_source} 获取 {symbol} 数据...")
                step_start = time.time()
                raw_data = self.data_manager.get_stock_data(symbol, period=period, interval=interval)
                step_time = time.time() - step_start
                
                print(f"✅ 数据获取完成: {len(raw_data):,} 条记录 ({step_time:.2f}秒)")
                self.logger.info(f"数据获取完成: {len(raw_data)} 条记录 (耗时: {step_time:.2f}秒)")
                pbar.update(1)
                
                # 2. 特征工程
                pbar.set_description("🔧 执行特征工程")
                self.logger.info("执行特征工程...")
                step_start = time.time()
                features_data = self.feature_engineer.prepare_features(raw_data)
                step_time = time.time() - step_start
                
                # 分析特征类别
                feature_stats = self._analyze_feature_categories(features_data.columns)
                print(f"✅ 特征工程完成: {len(features_data):,} 条记录, {len(features_data.columns)} 个特征 ({step_time:.2f}秒)")
                print(f"   特征分布: {feature_stats}")
                self.logger.info(f"特征工程完成: {len(features_data)} 条记录, {len(features_data.columns)} 个特征 (耗时: {step_time:.2f}秒)")
                pbar.update(1)
                
                # 3. 数据质量检查
                pbar.set_description("🔍 数据质量检查")
                step_start = time.time()
                quality_report = self._check_data_quality(features_data)
                step_time = time.time() - step_start
                
                print(f"✅ 数据质量检查: {quality_report} ({step_time:.2f}秒)")
                self.logger.info(f"数据质量检查完成 (耗时: {step_time:.2f}秒)")
                pbar.update(1)
                
                # 4. 数据集划分
                pbar.set_description("✂️ 划分数据集")
                self.logger.info("划分数据集...")
                step_start = time.time()
                datasets = self._split_dataset(features_data, train_ratio, val_ratio, test_ratio)
                step_time = time.time() - step_start
                
                print(f"✅ 数据集划分完成 ({step_time:.2f}秒)")
                for split_name, data in datasets.items():
                    print(f"   {split_name}: {len(data):,} 条记录")
                pbar.update(1)
                
                # 5. 保存数据
                pbar.set_description("💾 保存数据文件")
                step_start = time.time()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                symbol_dir = self.output_dir / f"{symbol}_{timestamp}"
                symbol_dir.mkdir(exist_ok=True)
                
                saved_files = self._save_datasets(symbol, datasets, symbol_dir)
                
                # 保存元数据
                metadata = {
                    'symbol': symbol,
                    'period': period,
                    'interval': interval,
                    'download_time': timestamp,
                    'raw_data_shape': raw_data.shape,
                    'features_data_shape': features_data.shape,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'dataset_shapes': {k: v.shape for k, v in datasets.items()},
                    'feature_columns': list(features_data.columns),
                    'feature_categories': feature_stats,
                    'quality_report': quality_report,
                    'processing_time': time.time() - start_time,
                    'saved_files': saved_files
                }
                
                metadata_file = symbol_dir / "metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                
                step_time = time.time() - step_start
                print(f"✅ 数据保存完成: {symbol_dir.name} ({step_time:.2f}秒)")
                self.logger.info(f"数据保存完成: {symbol_dir} (耗时: {step_time:.2f}秒)")
                pbar.update(1)
            
            # 总结信息
            total_time = time.time() - start_time
            print(f"\n🎉 {symbol} 处理成功完成!")
            print(f"📁 输出目录: {symbol_dir}")
            print(f"⏱️  总耗时: {total_time:.2f}秒")
            print(f"📊 数据统计: {len(raw_data):,} → {len(features_data):,} 条记录, {len(features_data.columns)} 个特征")
            print(f"{'='*60}")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'output_dir': str(symbol_dir),
                'metadata': metadata
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"\n❌ {symbol} 处理失败!")
            print(f"🔥 错误: {e}")
            print(f"⏱️  耗时: {total_time:.2f}秒")
            print(f"{'='*60}")
            
            self.logger.error(f"下载 {symbol} 数据失败: {e} (耗时: {total_time:.2f}秒)")
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    def download_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        批量下载多个股票数据
        
        Args:
            symbols: 股票代码列表
            period: 数据周期
            interval: 数据间隔
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            Dict[str, Any]: 批量处理结果
        """
        data_source = os.getenv('DATA_SOURCE_TYPE', 'yfinance')
        batch_start_time = time.time()
        
        # 打印批量处理开始信息
        print(f"\n{'='*80}")
        print(f"🚀 批量数据处理开始")
        print(f"符号列表: {', '.join(symbols)}")
        print(f"数据源: {data_source} | 周期: {period} | 间隔: {interval}")
        print(f"总数: {len(symbols)} 个符号")
        print(f"{'='*80}")
        
        self.logger.info(f"开始批量下载 {len(symbols)} 个符号数据 (数据源: {data_source})")
        
        results = {}
        successful_downloads = []
        failed_downloads = []
        
        # 使用tqdm显示批量处理进度
        with tqdm(symbols, desc="批量处理", ncols=100, colour='blue') as pbar:
            for i, symbol in enumerate(pbar, 1):
                pbar.set_description(f"处理 {symbol} ({i}/{len(symbols)})")
                self.logger.info(f"处理符号: {symbol} ({i}/{len(symbols)})")
                
                symbol_start_time = time.time()
                result = self.download_single_stock(symbol, period, interval, train_ratio, val_ratio, test_ratio)
                symbol_time = time.time() - symbol_start_time
                
                results[symbol] = result
                result['processing_time'] = symbol_time
                
                if result['status'] == 'success':
                    successful_downloads.append(symbol)
                    self.logger.info(f"SUCCESS: {symbol} 下载成功 (耗时: {symbol_time:.2f}秒)")
                    pbar.set_postfix(success=len(successful_downloads), failed=len(failed_downloads))
                else:
                    failed_downloads.append(symbol)
                    self.logger.error(f"FAILED: {symbol} 下载失败: {result.get('error', '未知错误')} (耗时: {symbol_time:.2f}秒)")
                    pbar.set_postfix(success=len(successful_downloads), failed=len(failed_downloads))
        
        # 批量处理总结
        total_batch_time = time.time() - batch_start_time
        
        # 保存批量处理摘要
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'batch_download_time': timestamp,
            'total_symbols': len(symbols),
            'successful_count': len(successful_downloads),
            'failed_count': len(failed_downloads),
            'successful_symbols': successful_downloads,
            'failed_symbols': failed_downloads,
            'period': period,
            'interval': interval,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'total_processing_time': total_batch_time,
            'average_time_per_symbol': total_batch_time / len(symbols) if symbols else 0,
            'detailed_results': results
        }
        
        summary_file = self.output_dir / f"batch_download_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 打印批量处理完成信息
        print(f"\n{'='*80}")
        print(f"🎯 批量处理完成!")
        print(f"✅ 成功: {len(successful_downloads)}/{len(symbols)} 个符号")
        print(f"❌ 失败: {len(failed_downloads)}/{len(symbols)} 个符号")
        print(f"⏱️  总耗时: {total_batch_time:.2f}秒")
        print(f"📊 平均耗时: {total_batch_time/len(symbols):.2f}秒/符号")
        
        if successful_downloads:
            print(f"✅ 成功符号: {', '.join(successful_downloads)}")
        if failed_downloads:
            print(f"❌ 失败符号: {', '.join(failed_downloads)}")
            
        print(f"📁 摘要文件: {summary_file}")
        print(f"{'='*80}")
        
        self.logger.info(f"批量下载完成: 成功 {len(successful_downloads)}, 失败 {len(failed_downloads)}, 总耗时: {total_batch_time:.2f}秒")
        
        return summary
    
    def _split_dataset(
        self,
        data: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Dict[str, pd.DataFrame]:
        """
        按时间顺序划分数据集
        
        Args:
            data: 特征数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            Dict[str, pd.DataFrame]: 划分后的数据集
        """
        total_len = len(data)
        
        # 计算分割点
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        
        # 按时间顺序划分
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        self.logger.info(f"数据集划分完成:")
        self.logger.info(f"  训练集: {len(train_data)} 条记录 ({len(train_data)/total_len:.1%})")
        self.logger.info(f"  验证集: {len(val_data)} 条记录 ({len(val_data)/total_len:.1%})")
        self.logger.info(f"  测试集: {len(test_data)} 条记录 ({len(test_data)/total_len:.1%})")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _save_datasets(
        self,
        symbol: str,
        datasets: Dict[str, pd.DataFrame],
        output_dir: Path
    ) -> Dict[str, str]:
        """
        保存数据集到文件
        
        Args:
            symbol: 股票代码
            datasets: 数据集字典
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 保存的文件路径
        """
        saved_files = {}
        
        for split_name, data in datasets.items():
            # 同时保存CSV和Pickle格式
            csv_file = output_dir / f"{symbol}_{split_name}.csv"
            pkl_file = output_dir / f"{symbol}_{split_name}.pkl"
            
            # 保存CSV (便于查看)
            data.to_csv(csv_file, index=True)
            
            # 保存Pickle (便于快速加载)
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
            
            saved_files[f'{split_name}_csv'] = str(csv_file)
            saved_files[f'{split_name}_pkl'] = str(pkl_file)
            
            self.logger.info(f"保存 {split_name} 数据: {len(data)} 条记录 -> {csv_file.name}")
        
        return saved_files
    
    def load_dataset(
        self,
        symbol: str,
        split: str = "train",
        data_dir: Optional[str] = None,
        format_type: str = "pkl"
    ) -> pd.DataFrame:
        """
        加载数据集
        
        Args:
            symbol: 股票代码
            split: 数据集分割 (train/val/test)
            data_dir: 数据目录
            format_type: 文件格式 (pkl/csv)
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        if data_dir:
            base_dir = Path(data_dir)
        else:
            # 查找最新的数据目录
            symbol_dirs = list(self.output_dir.glob(f"{symbol}_*"))
            if not symbol_dirs:
                raise FileNotFoundError(f"未找到 {symbol} 的数据目录")
            base_dir = max(symbol_dirs)  # 选择最新的
        
        if format_type == "pkl":
            file_path = base_dir / f"{symbol}_{split}.pkl"
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            file_path = base_dir / f"{symbol}_{split}.csv"
            data = pd.read_csv(file_path, index_col=0)
        
        self.logger.info(f"加载数据集: {file_path} ({len(data)} 条记录)")
        return data
    
    def _analyze_feature_categories(self, feature_columns: List[str]) -> Dict[str, int]:
        """
        分析特征类别分布
        
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
    
    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            data: 特征数据
            
        Returns:
            Dict[str, Any]: 数据质量报告
        """
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
        
        return {
            'total_cells': total_cells,
            'missing_values': int(nan_count),
            'infinite_values': int(inf_count),
            'missing_percentage': round(nan_count / total_cells * 100, 2),
            'constant_features': len(constant_features),
            'outlier_features': len(outliers),
            'quality_score': round((1 - (nan_count + inf_count) / total_cells) * 100, 2)
        }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="数据下载和预处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 股票参数
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--symbol', '-s',
                      help='单个交易代码 (默认: EURUSD)')
    
    group.add_argument('--symbols',
                      help='多个交易代码，逗号分隔 (例如: EURUSD,GBPUSD,USDJPY)')
    
    parser.add_argument('--period', '-p',
                       help='数据周期 (例如: 1y, 2y, 6m, 3m)')
    
    parser.add_argument('--interval', '-i',
                       help='数据间隔 (fxminute仅支持1m, yfinance: 1m仅7天/1h支持2年/1d支持历史)')
    
    # 数据集划分参数
    parser.add_argument('--train-ratio',
                       type=float,
                       help='训练集比例 (默认: 0.7)')
    
    parser.add_argument('--val-ratio',
                       type=float,
                       help='验证集比例 (默认: 0.2)')
    
    parser.add_argument('--test-ratio',
                       type=float,
                       help='测试集比例 (默认: 0.1)')
    
    # 输出参数
    parser.add_argument('--output-dir',
                       help='输出目录 (默认: datasets)')
    
    parser.add_argument('--config', '-c',
                       help='配置文件路径')
    
    # 代理参数
    parser.add_argument('--use-proxy',
                       action='store_true',
                       help='启用代理')
    
    parser.add_argument('--no-proxy',
                       action='store_true',
                       help='禁用代理')
    
    parser.add_argument('--proxy-host',
                       default='127.0.0.1',
                       help='代理主机地址 (默认: 127.0.0.1)')
    
    parser.add_argument('--proxy-port',
                       default='7891',
                       help='代理端口 (默认: 7891)')
    
    # 数据源参数
    parser.add_argument('--data-source',
                       choices=['fxminute', 'yfinance', 'truefx', 'oanda', 'histdata'],
                       default='fxminute',
                       help='数据源类型 (默认: fxminute)')
    
    parser.add_argument('--data-source-config',
                       help='数据源配置文件路径 (JSON格式)')
    
    # 其他参数
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='详细输出')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("=" * 60)
    print("TensorTrade 市场数据下载和预处理脚本")
    print("支持多数据源的市场数据获取和特征工程")
    print("=" * 60)
    
    # =============================================
    # 默认参数配置 - 在这里修改默认值
    # =============================================
    DEFAULT_SYMBOL = "EURUSD"         # 默认外汇代码 (欧元/美元，FX-1-Minute-Data支持)
    DEFAULT_SYMBOLS = None            # 默认多外汇（None表示使用单外汇）
    DEFAULT_PERIOD = "max"            # 默认数据周期 (所有可用数据，2000-2024年)
    DEFAULT_INTERVAL = "1m"           # 默认数据间隔 (1分钟粒度，FXMinute仅支持1分钟)
    DEFAULT_TRAIN_RATIO = 0.7         # 默认训练集比例
    DEFAULT_VAL_RATIO = 0.2           # 默认验证集比例  
    DEFAULT_TEST_RATIO = 0.1          # 默认测试集比例
    DEFAULT_OUTPUT_DIR = "datasets"   # 默认输出目录
    DEFAULT_CONFIG = None             # 默认配置文件
    DEFAULT_VERBOSE = False           # 默认详细输出
    
    # 代理配置 - 修改这里来配置代理设置（FXMinute本地数据不需要代理）
    DEFAULT_USE_PROXY = False         # 默认不启用代理（FXMinute使用本地数据）
    DEFAULT_PROXY_HOST = "127.0.0.1"  # 默认代理主机
    DEFAULT_PROXY_PORT = "7891"       # 默认代理端口（本地socket代理）
    
    # 数据源配置 - 修改这里来配置默认数据源
    DEFAULT_DATA_SOURCE = "fxminute"   # 默认数据源类型 (FX-1-Minute-Data本地缓存数据)
    DEFAULT_DATA_SOURCE_CONFIG = None  # 默认数据源配置文件
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 使用默认值覆盖未指定的参数
    symbol = args.symbol or DEFAULT_SYMBOL
    symbols = args.symbols or DEFAULT_SYMBOLS
    period = args.period or DEFAULT_PERIOD
    interval = args.interval or DEFAULT_INTERVAL
    train_ratio = args.train_ratio or DEFAULT_TRAIN_RATIO
    val_ratio = args.val_ratio or DEFAULT_VAL_RATIO
    test_ratio = args.test_ratio or DEFAULT_TEST_RATIO
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    config_path = args.config or DEFAULT_CONFIG
    verbose = args.verbose or DEFAULT_VERBOSE
    data_source = args.data_source or DEFAULT_DATA_SOURCE
    data_source_config_file = args.data_source_config or DEFAULT_DATA_SOURCE_CONFIG
    
    # 处理代理设置
    if args.no_proxy:
        use_proxy = False
    elif args.use_proxy:
        use_proxy = True
    else:
        use_proxy = DEFAULT_USE_PROXY
    
    # 设置环境变量（在创建下载器之前）
    os.environ['USE_PROXY'] = str(use_proxy).lower()
    os.environ['PROXY_HOST'] = args.proxy_host or DEFAULT_PROXY_HOST
    os.environ['PROXY_PORT'] = args.proxy_port or DEFAULT_PROXY_PORT
    os.environ['DATA_SOURCE_TYPE'] = data_source
    if data_source_config_file:
        os.environ['DATA_SOURCE_CONFIG'] = data_source_config_file
    
    # 显示配置状态
    if use_proxy:
        proxy_info = f"{os.environ['PROXY_HOST']}:{os.environ['PROXY_PORT']}"
        print(f"代理配置: http://{proxy_info}")
    else:
        print("代理配置: 已禁用")
    
    print(f"数据源: {data_source}")
    if data_source_config_file:
        print(f"配置文件: {data_source_config_file}")
    
    try:
        # 创建数据下载器
        downloader = DataDownloader(
            config_path=config_path,
            output_dir=output_dir
        )
        
        # 执行下载
        if symbols:
            # 多个股票
            symbols_list = [s.strip().upper() for s in symbols.split(',')]
            result = downloader.download_multiple_stocks(
                symbols=symbols_list,
                period=period,
                interval=interval,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
            
            print(f"\n批量下载完成:")
            print(f"总计: {result['total_symbols']} 个股票")
            print(f"成功: {result['successful_count']} 个")
            print(f"失败: {result['failed_count']} 个")
            
            if result['successful_symbols']:
                print(f"成功的股票: {', '.join(result['successful_symbols'])}")
            
            if result['failed_symbols']:
                print(f"失败的股票: {', '.join(result['failed_symbols'])}")
            
            if verbose:
                print("\n详细结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        
        else:
            # 单个股票 (默认或指定)
            result = downloader.download_single_stock(
                symbol=symbol,
                period=period,
                interval=interval,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
            
            if result['status'] == 'success':
                print(f"\nSUCCESS: {symbol} 数据下载成功!")
                print(f"输出目录: {result['output_dir']}")
                if verbose:
                    print("\n详细信息:")
                    print(json.dumps(result['metadata'], indent=2, ensure_ascii=False, default=str))
            else:
                print(f"\nFAILED: {symbol} 数据下载失败:")
                print(f"错误: {result.get('error', '未知错误')}")
                return 1
    
    except Exception as e:
        print(f"\n执行失败: {e}")
        return 1
    
    print(f"\n数据下载完成，保存至: {output_dir}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)