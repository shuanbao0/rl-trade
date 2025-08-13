"""
日志配置工具
提供统一的日志格式和配置
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则不输出到文件
        console_output: 是否输出到控制台
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 防止日志消息传播到根日志记录器，避免重复输出
    logger.propagate = False
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        # 设置为无缓冲模式，立即写入
        file_handler.stream.reconfigure(line_buffering=True)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加刷新方法
    def flush_all():
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
    logger.flush_all = flush_all
    
    return logger


def get_default_log_file(module_name: str) -> str:
    """
    获取默认日志文件路径
    
    Args:
        module_name: 模块名称
        
    Returns:
        日志文件路径
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    return os.path.join(log_dir, f"{module_name}_{timestamp}.log") 