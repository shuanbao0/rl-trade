#!/usr/bin/env python
"""
TensorTrade交易系统 FastAPI Web服务

提供RESTful API接口用于:
1. 模型训练
2. 回测验证
3. 实时交易
4. 系统监控
5. 结果查询
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from main import TensorTradeSystem
    from src.utils.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 创建FastAPI应用
app = FastAPI(
    title="TensorTrade强化学习交易系统",
    description="基于TensorTrade + Ray RLlib的智能交易平台API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局系统实例
trading_system: Optional[TensorTradeSystem] = None
background_tasks: Dict[str, Any] = {}

# 配置日志
logger = setup_logger("TensorTradeAPI", "INFO")

# Pydantic模型定义
class TrainingRequest(BaseModel):
    symbol: str
    period: str = "2y"
    iterations: int = 100
    save_path: str = "models"
    hyperparameter_search: bool = False

class ValidationRequest(BaseModel):
    symbol: str
    period: str = "2y"
    num_folds: int = 5
    save_results: bool = True

class BacktestRequest(BaseModel):
    symbol: str
    period: str = "1y"
    model_path: str

class EvaluationRequest(BaseModel):
    symbol: str
    period: str = "6m"
    model_path: str

class LiveTradingRequest(BaseModel):
    symbol: str
    model_path: str
    duration_hours: int = 8

class SystemStatus(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    metrics: Dict[str, Any]

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动初始化"""
    global trading_system
    
    logger.info("启动TensorTrade API服务...")
    
    try:
        # 初始化交易系统
        trading_system = TensorTradeSystem()
        trading_system.initialize_components()
        
        logger.info("TensorTrade系统初始化完成")
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭清理"""
    global trading_system
    
    logger.info("关闭TensorTrade API服务...")
    
    if trading_system:
        trading_system.cleanup()
    
    logger.info("系统清理完成")

# 依赖注入
def get_trading_system() -> TensorTradeSystem:
    """获取交易系统实例"""
    if trading_system is None:
        raise HTTPException(status_code=503, detail="交易系统未初始化")
    return trading_system

# API路由定义

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径 - API信息"""
    return {
        "name": "TensorTrade强化学习交易系统",
        "version": "1.0.0",
        "description": "基于TensorTrade + Ray RLlib的智能交易平台",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=SystemStatus)
async def health_check(system: TensorTradeSystem = Depends(get_trading_system)):
    """系统健康检查"""
    try:
        # 检查各组件状态
        components = {
            "data_manager": "ok" if system.data_manager else "error",
            "feature_engineer": "ok" if system.feature_engineer else "error",
            "trading_environment": "ok" if system.trading_environment else "error",
            "trading_agent": "ok" if system.trading_agent else "error",
            "validator": "ok" if system.validator else "error",
            "risk_manager": "ok" if system.risk_manager else "error"
        }
        
        # 系统指标
        metrics = {
            "uptime": "running",
            "background_tasks": len(background_tasks),
            "timestamp": datetime.now().isoformat()
        }
        
        overall_status = "healthy" if all(status == "ok" for status in components.values()) else "degraded"
        
        return SystemStatus(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            components=components,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/train")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    system: TensorTradeSystem = Depends(get_trading_system)
):
    """训练强化学习模型"""
    try:
        task_id = f"train_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 添加后台任务
        background_tasks.add_task(
            run_training_task,
            task_id,
            request,
            system
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"开始训练 {request.symbol} 模型",
            "estimated_time": f"{request.iterations * 2} 分钟"
        }
        
    except Exception as e:
        logger.error(f"训练请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/validate")
async def validate_model(
    request: ValidationRequest,
    background_tasks: BackgroundTasks,
    system: TensorTradeSystem = Depends(get_trading_system)
):
    """执行Walk Forward验证"""
    try:
        task_id = f"validate_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            run_validation_task,
            task_id,
            request,
            system
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"开始验证 {request.symbol} 模型",
            "estimated_time": f"{request.num_folds * 5} 分钟"
        }
        
    except Exception as e:
        logger.error(f"验证请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/backtest")
async def backtest_model(
    request: BacktestRequest,
    system: TensorTradeSystem = Depends(get_trading_system)
):
    """回测模型性能"""
    try:
        result = system.backtest_mode(
            symbol=request.symbol,
            period=request.period,
            model_path=request.model_path
        )
        
        return result
        
    except Exception as e:
        logger.error(f"回测请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evaluate")
async def evaluate_model(
    request: EvaluationRequest,
    system: TensorTradeSystem = Depends(get_trading_system)
):
    """评估模型性能"""
    try:
        result = system.evaluate_mode(
            symbol=request.symbol,
            period=request.period,
            model_path=request.model_path
        )
        
        return result
        
    except Exception as e:
        logger.error(f"评估请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/live")
async def start_live_trading(
    request: LiveTradingRequest,
    background_tasks: BackgroundTasks,
    system: TensorTradeSystem = Depends(get_trading_system)
):
    """启动实时交易"""
    try:
        task_id = f"live_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            run_live_trading_task,
            task_id,
            request,
            system
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"开始实时交易 {request.symbol}",
            "duration": f"{request.duration_hours} 小时"
        }
        
    except Exception as e:
        logger.error(f"实时交易请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return background_tasks[task_id]

@app.get("/api/v1/tasks")
async def list_tasks():
    """列出所有任务"""
    return {
        "tasks": list(background_tasks.keys()),
        "total": len(background_tasks)
    }

@app.get("/api/v1/models")
async def list_models():
    """列出可用模型"""
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            return {"models": [], "total": 0}
        
        models = []
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                models.append({
                    "name": model_path.name,
                    "path": str(model_path),
                    "created": datetime.fromtimestamp(model_path.stat().st_ctime).isoformat(),
                    "size": sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                })
        
        return {
            "models": sorted(models, key=lambda x: x["created"], reverse=True),
            "total": len(models)
        }
        
    except Exception as e:
        logger.error(f"列出模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/results/{result_type}")
async def get_results(result_type: str, limit: int = 10):
    """获取结果文件"""
    try:
        results_dir = Path("results")
        if not results_dir.exists():
            return {"results": [], "total": 0}
        
        pattern = f"{result_type}_*.json"
        result_files = list(results_dir.glob(pattern))
        
        results = []
        for file_path in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "filename": file_path.name,
                        "created": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "data": data
                    })
            except Exception as e:
                logger.warning(f"读取结果文件失败 {file_path}: {e}")
        
        return {
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"获取结果失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """下载文件"""
    try:
        if file_type not in ["models", "results", "reports", "logs"]:
            raise HTTPException(status_code=400, detail="无效的文件类型")
        
        file_path = Path(file_type) / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 后台任务函数
async def run_training_task(task_id: str, request: TrainingRequest, system: TensorTradeSystem):
    """执行训练任务"""
    try:
        background_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "正在训练模型...",
            "start_time": datetime.now().isoformat()
        }
        
        result = system.train_mode(
            symbol=request.symbol,
            period=request.period,
            iterations=request.iterations,
            save_path=request.save_path
        )
        
        background_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "训练完成",
            "result": result,
            "end_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        background_tasks[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"训练失败: {str(e)}",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        }

async def run_validation_task(task_id: str, request: ValidationRequest, system: TensorTradeSystem):
    """执行验证任务"""
    try:
        background_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "正在执行验证...",
            "start_time": datetime.now().isoformat()
        }
        
        result = system.validate_mode(
            symbol=request.symbol,
            period=request.period,
            num_folds=request.num_folds,
            save_results=request.save_results
        )
        
        background_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "验证完成",
            "result": result,
            "end_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        background_tasks[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"验证失败: {str(e)}",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        }

async def run_live_trading_task(task_id: str, request: LiveTradingRequest, system: TensorTradeSystem):
    """执行实时交易任务"""
    try:
        background_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "正在启动实时交易...",
            "start_time": datetime.now().isoformat()
        }
        
        result = system.live_mode(
            symbol=request.symbol,
            model_path=request.model_path,
            duration_hours=request.duration_hours
        )
        
        background_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "实时交易完成",
            "result": result,
            "end_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        background_tasks[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"实时交易失败: {str(e)}",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        }

# 静态文件服务（可选）
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 开发模式启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )