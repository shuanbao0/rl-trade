#!/usr/bin/env python
"""
EURUSD 外汇数据下载脚本

专门用于下载 EURUSD 的全部历史数据
包含完整的时间序列数据和特征工程
"""

from src.data.services import download_single
from src.data.sources.base import DataSource, DataPeriod, DataInterval

def main():
    """主函数 - 下载 EURUSD 全部数据"""
    
    print("=" * 60)
    print("EURUSD 外汇数据下载脚本")
    print("=" * 60)
    print("交易对: EURUSD")
    print("数据源: FX-1-Minute-Data (本地高质量外汇数据)")
    print("数据质量: 1分钟级OHLC数据 (2000-2024年)")
    print("=" * 60)
    
    # 配置下载参数 - 下载全部可用数据
    data_source = DataSource.FXMINUTE
    period = DataPeriod.MAX           # 下载最大可用数据
    interval = DataInterval.MINUTE_1  # 1分钟数据
    
    print("下载配置:")
    print(f"  交易对: EURUSD")
    print(f"  数据源: {data_source.display_name}")
    print(f"  数据周期: 全部可用数据 (2000-2024年)")
    print(f"  数据间隔: {interval.value}")
    print(f"  特征工程: 启用")
    print(f"  数据集划分: 训练(70%) 验证(20%) 测试(10%)")
    print("=" * 60)
    
    print("开始下载 EURUSD 数据...")
    print("=" * 60)
    
    try:
        # 执行下载
        result = download_single(
            symbol="EURUSD",
            data_source=data_source,
            period=period,
            interval=interval,
            include_features=True,      # 包含特征工程
            split_datasets=True,        # 划分数据集
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            save_data=True
        )
        
        # 显示下载结果
        print("\n" + "=" * 60)
        print("EURUSD 数据下载完成!")
        print("=" * 60)
        
        if result.is_successful():
            print(f"下载状态: 成功")
            print(f"数据点数: {result.data_points:,} 条")
            print(f"时间范围: {result.time_range.start_date} 到 {result.time_range.end_date}")
            print(f"处理耗时: {result.processing_time:.2f} 秒")
            print(f"数据质量: {result.quality_report.quality_score:.2f}/5.0")
            
            if result.feature_stats:
                print(f"特征数量: {result.feature_stats.total_features} 个")
                print(f"原始特征: {result.feature_stats.original_features} 个")
                print(f"技术指标: {result.feature_stats.technical_features} 个")
            
            if result.dataset_splits:
                print("\n数据集划分:")
                for split_name, split_info in result.dataset_splits.items():
                    print(f"  {split_name}: {split_info['size']:,} 条记录")
            
            print(f"\n数据保存位置: {result.output_dir}")
            print("包含文件:")
            print("  - EURUSD_raw.csv (原始OHLC数据)")
            print("  - EURUSD_features.csv (特征工程后数据)")
            print("  - EURUSD_train.csv (训练集)")
            print("  - EURUSD_val.csv (验证集)")
            print("  - EURUSD_test.csv (测试集)")
            print("  - EURUSD_metadata.json (数据集元信息)")
            
            print("=" * 60)
            print("EURUSD 数据准备完成，可用于模型训练!")
            return 0
            
        else:
            print(f"下载状态: 失败")
            print(f"错误信息: {result.error_message}")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\n下载过程中发生错误: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    print("EURUSD 数据下载工具")
    print("即将下载 EURUSD 的全部历史数据...")
    
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n用户取消下载")
        exit_code = 0
    
    print(f"\n程序结束，退出码: {exit_code}")
    exit(exit_code)