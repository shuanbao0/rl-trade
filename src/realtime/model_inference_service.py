"""
模型推理服务
提供实时模型推理和预测服务

主要功能:
1. Ray Serve模型部署
2. 模型加载和预热
3. 实时预测接口
4. 批量推理优化
5. 模型版本管理
6. 性能监控
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import threading
from queue import Queue, Empty
import json

# Ray相关导入
try:
    import ray
    from ray import serve
    from ray.serve import deployment
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config
from ..features.feature_engineer import FeatureEngineer


@dataclass
class PredictionRequest:
    """预测请求"""
    request_id: str
    symbol: str
    features: np.ndarray
    timestamp: datetime
    callback: Optional[callable] = None


@dataclass
class PredictionResult:
    """预测结果"""
    request_id: str
    symbol: str
    action: float  # -1到1之间的连续动作
    confidence: float  # 置信度
    timestamp: datetime
    inference_time_ms: float
    model_version: str


@dataclass
class ModelMetrics:
    """模型性能指标"""
    total_requests: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    average_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    requests_per_second: float = 0.0
    last_reset_time: datetime = None
    
    def reset(self):
        """重置指标"""
        self.total_requests = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.average_inference_time_ms = 0.0
        self.max_inference_time_ms = 0.0
        self.requests_per_second = 0.0
        self.last_reset_time = datetime.now()


class ModelWrapper:
    """
    模型包装器
    
    封装训练好的强化学习模型，提供统一的推理接口
    """
    
    def __init__(self, model_path: str, feature_engineer: FeatureEngineer):
        """
        初始化模型包装器
        
        Args:
            model_path: 模型文件路径
            feature_engineer: 特征工程器
        """
        self.model_path = model_path
        self.feature_engineer = feature_engineer
        self.model = None
        self.model_version = "unknown"
        self.is_loaded = False
        
        # 初始化日志
        self.logger = setup_logger(
            name="ModelWrapper",
            level="INFO",
            log_file=get_default_log_file("model_wrapper")
        )
        
        # 加载模型
        self.load_model()
    
    def load_model(self) -> None:
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 根据文件扩展名选择加载方式
            if self.model_path.endswith('.pkl'):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif self.model_path.endswith('.zip'):
                # Ray RLlib模型
                if RAY_AVAILABLE:
                    from ray.rllib.algorithms.ppo import PPO
                    self.model = PPO.from_checkpoint(self.model_path)
                else:
                    raise ImportError("Ray未安装，无法加载RLlib模型")
            else:
                raise ValueError(f"不支持的模型格式: {self.model_path}")
            
            # 获取模型版本
            self.model_version = self._extract_model_version()
            self.is_loaded = True
            
            self.logger.info(f"模型加载成功: {self.model_path}, 版本: {self.model_version}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.is_loaded = False
            raise
    
    def _extract_model_version(self) -> str:
        """提取模型版本信息"""
        try:
            # 从文件名或路径中提取版本信息
            filename = os.path.basename(self.model_path)
            if '_v' in filename:
                version = filename.split('_v')[1].split('.')[0]
                return f"v{version}"
            else:
                # 使用文件修改时间作为版本
                mtime = os.path.getmtime(self.model_path)
                return datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        except:
            return "unknown"
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        模型推理
        
        Args:
            features: 特征数组
            
        Returns:
            Tuple[float, float]: (动作, 置信度)
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        try:
            start_time = time.time()
            
            # 根据模型类型进行推理
            if hasattr(self.model, 'compute_single_action'):
                # Ray RLlib模型
                action = self.model.compute_single_action(features)
                if isinstance(action, (list, np.ndarray)):
                    action = float(action[0])
                else:
                    action = float(action)
                
                # 简单的置信度计算（可以根据需要改进）
                confidence = min(abs(action), 1.0)
                
            elif hasattr(self.model, 'predict'):
                # Sklearn类模型
                action = self.model.predict(features.reshape(1, -1))[0]
                confidence = 0.8  # 默认置信度
                
            else:
                # 自定义模型接口
                if hasattr(self.model, 'forward') or hasattr(self.model, '__call__'):
                    result = self.model(features)
                    if isinstance(result, tuple):
                        action, confidence = result
                    else:
                        action = result
                        confidence = 0.8
                else:
                    raise ValueError("模型没有可识别的推理方法")
            
            # 确保动作在有效范围内
            action = np.clip(action, -1.0, 1.0)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            return float(action), float(confidence)
            
        except Exception as e:
            self.logger.error(f"模型推理失败: {e}")
            raise
    
    def preprocess_data(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        数据预处理
        
        Args:
            market_data: 市场数据
            
        Returns:
            np.ndarray: 处理后的特征
        """
        try:
            # 使用特征工程器处理数据
            features = self.feature_engineer.prepare_features(market_data)
            
            # 获取最新的特征向量
            if len(features) > 0:
                latest_features = features.iloc[-1].values
                # 处理NaN值
                latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
                return latest_features
            else:
                raise ValueError("特征工程返回空数据")
                
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            raise
    
    def warmup(self, sample_features: Optional[np.ndarray] = None) -> None:
        """
        模型预热
        
        Args:
            sample_features: 样本特征（可选）
        """
        try:
            if sample_features is None:
                # 创建随机样本特征
                feature_size = getattr(self.feature_engineer, 'feature_size', 20)
                sample_features = np.random.randn(feature_size)
            
            # 执行几次预测来预热模型
            for _ in range(3):
                self.predict(sample_features)
            
            self.logger.info("模型预热完成")
            
        except Exception as e:
            self.logger.warning(f"模型预热失败: {e}")


if RAY_AVAILABLE:
    @serve.deployment(
        ray_actor_options={"num_cpus": 1, "num_gpus": 0},
        max_ongoing_requests=100,  # Ray 2.x使用max_ongoing_requests替代max_concurrent_queries
        autoscaling_config={"min_replicas": 1, "max_replicas": 3}
    )
    class ModelInferenceActor:
        """Ray Serve推理Actor"""
        
        def __init__(self, model_path: str, feature_engineer_config: Dict[str, Any]):
            self.model_wrapper = None
            self.metrics = ModelMetrics()
            self.metrics.last_reset_time = datetime.now()
            
            # 初始化特征工程器
            config = Config()
            config.update(feature_engineer_config)
            feature_engineer = FeatureEngineer(config)
            
            # 初始化模型
            self.model_wrapper = ModelWrapper(model_path, feature_engineer)
            
            # 预热模型
            self.model_wrapper.warmup()
        
        async def predict(self, features: List[float]) -> Dict[str, Any]:
            """异步预测接口"""
            start_time = time.time()
            request_id = f"req_{int(time.time() * 1000000)}"
            
            try:
                features_array = np.array(features, dtype=np.float32)
                action, confidence = self.model_wrapper.predict(features_array)
                
                inference_time_ms = (time.time() - start_time) * 1000
                
                # 更新指标
                self.metrics.total_requests += 1
                self.metrics.successful_predictions += 1
                self._update_timing_metrics(inference_time_ms)
                
                return {
                    "request_id": request_id,
                    "action": action,
                    "confidence": confidence,
                    "inference_time_ms": inference_time_ms,
                    "model_version": self.model_wrapper.model_version,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.metrics.total_requests += 1
                self.metrics.failed_predictions += 1
                
                return {
                    "request_id": request_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        def _update_timing_metrics(self, inference_time_ms: float):
            """更新时间指标"""
            if self.metrics.average_inference_time_ms == 0:
                self.metrics.average_inference_time_ms = inference_time_ms
            else:
                # 指数移动平均
                alpha = 0.1
                self.metrics.average_inference_time_ms = (
                    (1 - alpha) * self.metrics.average_inference_time_ms + 
                    alpha * inference_time_ms
                )
            
            if inference_time_ms > self.metrics.max_inference_time_ms:
                self.metrics.max_inference_time_ms = inference_time_ms
            
            # 计算RPS
            if self.metrics.last_reset_time:
                elapsed = (datetime.now() - self.metrics.last_reset_time).total_seconds()
                if elapsed > 0:
                    self.metrics.requests_per_second = self.metrics.total_requests / elapsed
        
        async def get_metrics(self) -> Dict[str, Any]:
            """获取性能指标"""
            return {
                "total_requests": self.metrics.total_requests,
                "successful_predictions": self.metrics.successful_predictions,
                "failed_predictions": self.metrics.failed_predictions,
                "success_rate": (
                    self.metrics.successful_predictions / self.metrics.total_requests 
                    if self.metrics.total_requests > 0 else 0
                ),
                "average_inference_time_ms": self.metrics.average_inference_time_ms,
                "max_inference_time_ms": self.metrics.max_inference_time_ms,
                "requests_per_second": self.metrics.requests_per_second,
                "model_version": self.model_wrapper.model_version if self.model_wrapper else "unknown"
            }


class ModelInferenceService:
    """
    模型推理服务
    
    提供统一的模型推理接口，支持本地和分布式推理
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Config] = None,
        use_ray_serve: bool = True,
        batch_size: int = 32,
        max_batch_wait_ms: int = 50
    ):
        """
        初始化推理服务
        
        Args:
            model_path: 模型文件路径
            config: 配置对象
            use_ray_serve: 是否使用Ray Serve
            batch_size: 批处理大小
            max_batch_wait_ms: 最大批处理等待时间
        """
        self.model_path = model_path
        self.config = config or Config()
        self.use_ray_serve = use_ray_serve and RAY_AVAILABLE
        self.batch_size = batch_size
        self.max_batch_wait_ms = max_batch_wait_ms
        
        # 初始化日志
        self.logger = setup_logger(
            name="ModelInferenceService",
            level="INFO",
            log_file=get_default_log_file("model_inference_service")
        )
        
        # 推理状态
        self.is_running = False
        self.model_handle = None
        self.local_model = None
        
        # 批处理队列
        self.request_queue = Queue()
        self.batch_thread = None
        
        # 性能指标
        self.metrics = ModelMetrics()
        self.metrics.last_reset_time = datetime.now()
        
        self.logger.info(f"ModelInferenceService初始化 - Ray Serve: {self.use_ray_serve}")
    
    def start(self) -> None:
        """启动推理服务"""
        if self.is_running:
            self.logger.warning("推理服务已在运行")
            return
        
        try:
            if self.use_ray_serve:
                self._start_ray_serve()
            else:
                self._start_local_service()
            
            # 启动批处理线程
            if self.batch_size > 1:
                self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
                self.batch_thread.start()
            
            self.is_running = True
            self.logger.info("模型推理服务已启动")
            
        except Exception as e:
            self.logger.error(f"启动推理服务失败: {e}")
            raise
    
    def _start_ray_serve(self) -> None:
        """启动Ray Serve服务"""
        if not RAY_AVAILABLE:
            raise ImportError("Ray未安装，无法使用Ray Serve")
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init()
        
        # 启动Serve
        serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
        
        # 部署模型
        feature_engineer_config = self.config.to_dict()
        
        ModelInferenceActor.deploy(
            self.model_path,
            feature_engineer_config,
            name="trading_model"
        )
        
        # 获取模型句柄
        self.model_handle = serve.get_deployment("trading_model").get_handle()
        
        self.logger.info("Ray Serve模型部署完成")
    
    def _start_local_service(self) -> None:
        """启动本地推理服务"""
        feature_engineer = FeatureEngineer(self.config)
        self.local_model = ModelWrapper(self.model_path, feature_engineer)
        self.logger.info("本地模型加载完成")
    
    async def predict_async(
        self,
        features: Union[np.ndarray, List[float]],
        symbol: str = "UNKNOWN"
    ) -> PredictionResult:
        """
        异步预测
        
        Args:
            features: 特征数组
            symbol: 交易对符号
            
        Returns:
            PredictionResult: 预测结果
        """
        if not self.is_running:
            raise RuntimeError("推理服务未启动")
        
        start_time = time.time()
        request_id = f"{symbol}_{int(time.time() * 1000000)}"
        
        try:
            if isinstance(features, list):
                features = np.array(features, dtype=np.float32)
            
            if self.use_ray_serve and self.model_handle:
                # Ray Serve推理
                result = await self.model_handle.predict.remote(features.tolist())
                
                if "error" in result:
                    raise RuntimeError(result["error"])
                
                action = result["action"]
                confidence = result["confidence"]
                model_version = result["model_version"]
                
            else:
                # 本地推理
                if self.local_model is None:
                    raise RuntimeError("本地模型未加载")
                
                action, confidence = self.local_model.predict(features)
                model_version = self.local_model.model_version
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # 更新指标
            self.metrics.total_requests += 1
            self.metrics.successful_predictions += 1
            self._update_metrics(inference_time_ms)
            
            return PredictionResult(
                request_id=request_id,
                symbol=symbol,
                action=action,
                confidence=confidence,
                timestamp=datetime.now(),
                inference_time_ms=inference_time_ms,
                model_version=model_version
            )
            
        except Exception as e:
            self.metrics.total_requests += 1
            self.metrics.failed_predictions += 1
            self.logger.error(f"预测失败: {e}")
            raise
    
    def predict_sync(
        self,
        features: Union[np.ndarray, List[float]],
        symbol: str = "UNKNOWN"
    ) -> PredictionResult:
        """
        同步预测（阻塞）
        
        Args:
            features: 特征数组
            symbol: 交易对符号
            
        Returns:
            PredictionResult: 预测结果
        """
        import asyncio
        
        if asyncio.iscoroutinefunction(self.predict_async):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.predict_async(features, symbol))
            finally:
                loop.close()
        else:
            return self.predict_async(features, symbol)
    
    def _batch_processor(self) -> None:
        """批处理处理器"""
        batch_requests = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # 收集批处理请求
                while len(batch_requests) < self.batch_size:
                    try:
                        request = self.request_queue.get(timeout=0.01)  # 10ms超时
                        batch_requests.append(request)
                    except Empty:
                        break
                
                # 检查是否需要处理批次
                current_time = time.time()
                should_process = (
                    len(batch_requests) >= self.batch_size or
                    (batch_requests and (current_time - last_batch_time) * 1000 >= self.max_batch_wait_ms)
                )
                
                if should_process and batch_requests:
                    self._process_batch(batch_requests)
                    batch_requests.clear()
                    last_batch_time = current_time
                
            except Exception as e:
                self.logger.error(f"批处理异常: {e}")
    
    def _process_batch(self, requests: List[PredictionRequest]) -> None:
        """处理批次请求"""
        try:
            # 提取特征
            features_batch = np.array([req.features for req in requests])
            
            # 批量推理（这里简化为逐个处理，实际可以优化为真正的批量推理）
            for i, request in enumerate(requests):
                try:
                    result = self.predict_sync(features_batch[i], request.symbol)
                    if request.callback:
                        request.callback(result)
                except Exception as e:
                    self.logger.error(f"批处理中的单个请求失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"批处理失败: {e}")
    
    def _update_metrics(self, inference_time_ms: float) -> None:
        """更新性能指标"""
        if self.metrics.average_inference_time_ms == 0:
            self.metrics.average_inference_time_ms = inference_time_ms
        else:
            alpha = 0.1
            self.metrics.average_inference_time_ms = (
                (1 - alpha) * self.metrics.average_inference_time_ms + 
                alpha * inference_time_ms
            )
        
        if inference_time_ms > self.metrics.max_inference_time_ms:
            self.metrics.max_inference_time_ms = inference_time_ms
        
        # 计算RPS
        if self.metrics.last_reset_time:
            elapsed = (datetime.now() - self.metrics.last_reset_time).total_seconds()
            if elapsed > 0:
                self.metrics.requests_per_second = self.metrics.total_requests / elapsed
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_predictions": self.metrics.successful_predictions,
            "failed_predictions": self.metrics.failed_predictions,
            "success_rate": (
                self.metrics.successful_predictions / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            ),
            "average_inference_time_ms": self.metrics.average_inference_time_ms,
            "max_inference_time_ms": self.metrics.max_inference_time_ms,
            "requests_per_second": self.metrics.requests_per_second,
            "queue_size": self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0,
            "is_running": self.is_running,
            "use_ray_serve": self.use_ray_serve
        }
    
    def stop(self) -> None:
        """停止推理服务"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止批处理线程
        if self.batch_thread and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=5)
        
        # 清理Ray Serve
        if self.use_ray_serve:
            try:
                serve.shutdown()
            except:
                pass
        
        self.logger.info("模型推理服务已停止")
    
    def cleanup(self) -> None:
        """清理资源"""
        self.stop()
        
        # 清空队列
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except Empty:
                break