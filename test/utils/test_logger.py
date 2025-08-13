"""
测试日志模块
"""

import pytest
import logging
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.utils.logger import setup_logger, get_default_log_file


class TestLogger:
    def test_setup_logger_basic(self):
        """测试基本日志设置"""
        logger = setup_logger("test_logger", level="INFO")
        
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logger_with_file(self):
        """测试带文件的日志设置"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            logger = setup_logger("test_file_logger", level="DEBUG", log_file=log_file)
            
            # 测试日志记录
            logger.info("Test message")
            
            # 验证文件存在且包含日志
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
                
        finally:
            if os.path.exists(log_file):
                try:
                    os.unlink(log_file)
                except PermissionError:
                    # Windows权限问题，跳过文件删除
                    pass
    
    def test_get_default_log_file(self):
        """测试获取默认日志文件路径"""
        log_file = get_default_log_file("test_module")
        
        assert "test_module" in log_file
        assert log_file.endswith(".log")
        assert "logs" in log_file
    
    def test_logger_levels(self):
        """测试不同日志级别"""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in levels:
            logger = setup_logger(f"test_{level.lower()}", level=level)
            expected_level = getattr(logging, level)
            assert logger.level == expected_level
    
    def test_logger_formatting(self):
        """测试日志格式"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            logger = setup_logger("test_format", level="INFO", log_file=log_file)
            logger.info("Formatting test message")
            
            # 读取日志文件内容
            with open(log_file, 'r') as f:
                content = f.read()
                
            # 验证日志格式包含时间戳、级别和消息
            assert "INFO" in content
            assert "Formatting test message" in content
            assert "-" in content  # 时间戳分隔符
                
        finally:
            if os.path.exists(log_file):
                try:
                    os.unlink(log_file)
                except PermissionError:
                    # Windows权限问题，跳过文件删除
                    pass
    
    def test_console_and_file_handlers(self):
        """测试同时使用控制台和文件处理器"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            logger = setup_logger("test_handlers", level="INFO", log_file=log_file)
            
            # 验证有多个处理器
            assert len(logger.handlers) >= 1
            
            # 查找文件处理器
            has_file_handler = any(
                isinstance(handler, logging.FileHandler) 
                for handler in logger.handlers
            )
            assert has_file_handler
                
        finally:
            if os.path.exists(log_file):
                try:
                    os.unlink(log_file)
                except PermissionError:
                    # Windows权限问题，跳过文件删除
                    pass
    
    def test_logger_singleton_behavior(self):
        """测试日志器的单例行为"""
        logger1 = setup_logger("singleton_test", level="INFO")
        logger2 = setup_logger("singleton_test", level="DEBUG")
        
        # 同名日志器应该是同一个实例
        assert logger1 is logger2
    
    def test_log_directory_creation(self):
        """测试日志目录自动创建"""
        log_file = get_default_log_file("directory_test")
        log_dir = os.path.dirname(log_file)
        
        # 如果目录不存在，设置日志应该创建它
        if os.path.exists(log_dir):
            import shutil
            shutil.rmtree(log_dir)
        
        logger = setup_logger("dir_test", level="INFO", log_file=log_file)
        logger.info("Directory creation test")
        
        # 验证目录被创建
        assert os.path.exists(log_dir)
        assert os.path.exists(log_file)