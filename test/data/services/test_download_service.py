#!/usr/bin/env python
"""
测试新的下载接口
"""

from src.data.services import (
    download_single, 
    download_multiple,
    DownloadService,
    DownloadRequest,
    get_realtime_service,
    RealtimeRequest
)
from src.data.sources.base import DataSource, DataPeriod, DataInterval


def test_single_download():
    """测试单个下载"""
    print("=== 测试单个下载 ===")
    
    result = download_single(
        symbol="TSLA",
        data_source=DataSource.YFINANCE,
        period=DataPeriod.MONTH_1,  # 使用1个月数据确保有足够记录
        interval=DataInterval.DAY_1,
        include_features=False,  # 先跳过特征工程避免列名问题
        split_datasets=True,
        save_data=True
    )
    
    if result.is_successful():
        print(f"[SUCCESS] 单个下载成功!")
        print(f"  数据点: {result.data_points}")
        print(f"  处理时间: {result.processing_time:.2f}秒")
        print(f"  输出目录: {result.output_dir}")
        print(f"  原始数据形状: {result.raw_data.shape if result.raw_data is not None else 'N/A'}")
        
        if result.processed_data:
            for split, data in result.processed_data.items():
                print(f"  {split}: {data.shape if data is not None else 'N/A'}")
    else:
        print(f"[ERROR] 单个下载失败: {result.error_message}")
    
    return result.is_successful()


def test_multiple_download():
    """测试多个下载"""
    print("\n=== 测试多个下载 ===")
    
    result = download_multiple(
        symbols=["NVDA", "AMD"],
        data_source=DataSource.YFINANCE,
        period=DataPeriod.MONTH_1,  # 使用1个月数据确保有足够记录
        interval=DataInterval.DAY_1,
        include_features=False,
        split_datasets=False,
        save_data=True,
        concurrent=True,
        max_workers=2
    )
    
    print(f"[INFO] 多个下载结果:")
    print(f"  总数: {result.total_symbols}")
    print(f"  成功: {result.successful_count}")
    print(f"  失败: {result.failed_count}")
    print(f"  成功率: {result.success_rate():.1%}")
    print(f"  总耗时: {result.total_processing_time:.2f}秒")
    
    if result.successful_symbols:
        print(f"  成功股票: {', '.join(result.successful_symbols)}")
    if result.failed_symbols:
        print(f"  失败股票: {', '.join(result.failed_symbols)}")
    
    return result.successful_count > 0


def test_advanced_download():
    """测试高级下载"""
    print("\n=== 测试高级下载 ===")
    
    service = DownloadService()
    request = DownloadRequest(
        symbol="META",
        data_source=DataSource.YFINANCE,
        period=DataPeriod.WEEK_2,
        interval=DataInterval.DAY_1,
        include_features=False,  # 先跳过特征工程
        split_datasets=True,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        save_data=True,
        file_formats=['csv', 'pkl']
    )
    
    result = service.download_single(request)
    
    if result.is_successful():
        print(f"[SUCCESS] 高级下载成功!")
        print(f"  原始数据: {result.raw_data.shape if result.raw_data is not None else 'N/A'}")
        print(f"  保存文件数: {len(result.saved_files) if result.saved_files else 0}")
        print(f"  输出目录: {result.output_dir}")
        
        if result.processed_data:
            for split, data in result.processed_data.items():
                print(f"  {split}: {data.shape if data is not None else 'N/A'}")
    else:
        print(f"[ERROR] 高级下载失败: {result.error_message}")
    
    return result.is_successful()


def test_realtime_service():
    """测试实时数据服务"""
    print("\n=== 测试实时数据服务 ===")
    
    try:
        service = get_realtime_service()
        
        # 创建实时数据请求
        request = RealtimeRequest(
            symbol="AAPL",
            data_source=DataSource.YFINANCE,
            interval=DataInterval.MINUTE_1,
            buffer_size=50,
            update_frequency=30,  # 30秒更新一次
            auto_save=False
        )
        
        # 创建数据流
        stream = service.create_stream(request)
        print(f"[SUCCESS] 创建数据流: {stream.stream_id}")
        
        # 列出所有流
        streams = service.list_streams()
        print(f"[INFO] 当前流数量: {len(streams)}")
        
        for stream_info in streams:
            print(f"  流ID: {stream_info['stream_id']}")
            print(f"  符号: {stream_info['symbol']}")
            print(f"  状态: {stream_info['status']}")
        
        # 清理
        service.stop_stream(stream.stream_id)
        print(f"[SUCCESS] 实时服务测试完成")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 实时服务测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始测试新的下载接口...")
    
    results = []
    
    # 测试各个功能
    results.append(("单个下载", test_single_download()))
    results.append(("多个下载", test_multiple_download()))
    results.append(("高级下载", test_advanced_download()))
    results.append(("实时服务", test_realtime_service()))
    
    # 输出测试结果
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    
    success_count = 0
    for test_name, success in results:
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"{status} {test_name}")
        if success:
            success_count += 1
    
    print(f"\n总体结果: {success_count}/{len(results)} 项测试通过")
    
    if success_count == len(results):
        print("[SUCCESS] 所有测试通过！新的下载接口工作正常")
        return 0
    else:
        print("[WARNING] 部分测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)