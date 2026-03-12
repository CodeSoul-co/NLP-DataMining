"""
THETA - ETM Topic Model Agent System
FastAPI Application Entry Point
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 动态添加 ETM 路径
from .core.config import settings
from .core.etm_paths import setup_etm_paths

# 设置 ETM 路径（会自动添加 ETM 目录和父目录）
setup_etm_paths()

from .api.routes import router
from .api.websocket import websocket_router
from .api.auth import router as auth_router
from .api.scripts import router as scripts_router
from .api.oss import router as oss_router
from .api.projects import router as projects_router
from .api.agent_compat import agent_router
from .api.data_api import router as data_api_router
from .core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # 生产环境必须显式配置 SECRET_KEY
    _default_key = "theta-secure-key-change-in-production-2025"
    if not settings.DEBUG and (not settings.SECRET_KEY or settings.SECRET_KEY == _default_key):
        raise ValueError(
            "SECRET_KEY must be explicitly set in production. "
            "Set SECRET_KEY in environment or .env file."
        )
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"Simulation Mode: {settings.SIMULATION_MODE}")
    logger.info(f"ETM Dir: {settings.ETM_DIR}")
    logger.info(f"Data Dir: {settings.DATA_DIR}")
    logger.info(f"Result Dir: {settings.RESULT_DIR}")
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.GPU_ID)
    
    # 初始化数据库（模拟模式用 SQLite，不堵）
    try:
        from .core.database import init_db
        await init_db()
        logger.info("Database initialized")
        # 从 PostgreSQL 恢复任务到内存缓存
        from .services.task_store import task_store
        await task_store.load_from_db()
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")

    # 模拟模式下跳过 Redis，避免连不上时卡住
    if not settings.SIMULATION_MODE:
        try:
            from .core.redis import get_redis
            redis = await get_redis()
            await redis.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    # 检查 OSS 配置
    if settings.OSS_ENABLED:
        logger.info(f"OSS enabled: bucket={settings.RESOLVED_OSS_BUCKET}, endpoint={settings.OSS_ENDPOINT}")
    else:
        logger.info("OSS not configured, using local storage")
    
    # 检查 PAI/DLC 配置
    if settings.DLC_ENABLED:
        logger.info(f"DLC enabled: workspace={settings.RESOLVED_WORKSPACE_ID}, dataset={settings.OSS_DATASET_ID}, region={settings.RESOLVED_REGION}")
    elif settings.PAI_ENABLED:
        logger.info(f"PAI-DLC enabled (no OSS_DATASET_ID): workspace={settings.RESOLVED_WORKSPACE_ID}, region={settings.RESOLVED_REGION}")
    else:
        logger.info("PAI-DLC not configured, using local training")
    
    if settings.EAS_ENABLED:
        logger.info("PAI-EAS enabled for inference")
    else:
        logger.info("PAI-EAS not configured, using local inference")
    
    yield
    
    # 关闭数据库连接
    try:
        from .core.database import close_db
        await close_db()
        logger.info("Database connection closed")
    except Exception:
        pass
    
    # 关闭 Redis 连接
    try:
        from .core.redis import close_redis
        await close_redis()
        logger.info("Redis connection closed")
    except Exception:
        pass
    
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    description="LangGraph-based Agent System for ETM Topic Modeling",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 不能用 allow_origins=["*"] + allow_credentials=True，浏览器会拦截；始终用具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(agent_router, prefix="/api/agent", tags=["agent"])
app.include_router(projects_router, prefix="/api")
app.include_router(data_api_router)  # theta_1 风格: /api/data/presigned-url 等
app.include_router(router, prefix="/api")
app.include_router(oss_router, prefix="/api")
app.include_router(websocket_router, prefix="/api")
app.include_router(scripts_router, prefix="/api")

# 挂载静态文件目录（如果存在）
if settings.RESULT_DIR.exists():
    app.mount(
        "/static/results",
        StaticFiles(directory=str(settings.RESULT_DIR)),
        name="results"
    )


from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

class LegacyChatRequest(BaseModel):
    message: str
    job_id: str = ""
    session_id: Optional[str] = None

@app.post("/chat")
async def legacy_chat(request: LegacyChatRequest):
    """theta_1-main Legacy: Simple Q&A chat (no /api prefix)"""
    from .services.chat_service import chat_service
    from .schemas.agent import ChatRequest, ChatResponse
    req = ChatRequest(message=request.message, context={"job_id": request.job_id})
    resp = chat_service.process_message(req)
    return {
        "job_id": request.job_id or "",
        "session_id": request.session_id or "",
        "message": resp.message,
        "status": "ok"
    }


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api": "/api"
    }


@app.get("/config")
async def get_config():
    """theta_1 风格配置（前端 getConfig 使用）"""
    return {
        "oss_bucket": settings.OSS_BUCKET_NAME or "theta-prod",
        "oss_endpoint": settings.OSS_ENDPOINT or "oss-cn-shanghai.aliyuncs.com",
        "default_num_topics": settings.DEFAULT_NUM_TOPICS,
        "default_epochs": settings.DEFAULT_EPOCHS,
        "supported_models": ["theta", "lda", "hdp", "btm", "etm", "ctm", "dtm", "nvdm", "gsm", "prodlda", "bertopic"],
        "supported_modes": ["zero_shot", "supervised", "unsupervised"],
        "supported_model_sizes": ["0.6B", "4B", "8B"],
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    from sqlalchemy import text
    health_status = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "simulation_mode": settings.SIMULATION_MODE
    }
    try:
        from .core.database import async_session_maker
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)[:50]}"
    if not settings.SIMULATION_MODE:
        try:
            from .core.redis import get_redis
            r = await get_redis()
            await r.ping()
            health_status["redis"] = "connected"
        except Exception as e:
            health_status["redis"] = f"error: {str(e)[:50]}"

    # OSS / DLC 状态
    health_status["oss_enabled"] = settings.OSS_ENABLED
    health_status["dlc_enabled"] = settings.DLC_ENABLED
    if settings.OSS_ENABLED:
        health_status["oss_bucket"] = settings.RESOLVED_OSS_BUCKET
        health_status["oss_endpoint"] = settings.OSS_ENDPOINT
    if settings.PAI_ENABLED:
        health_status["dlc_workspace_id"] = settings.RESOLVED_WORKSPACE_ID
        health_status["dlc_region"] = settings.RESOLVED_REGION
        health_status["dlc_image"] = settings.PAI_TRAINING_IMAGE

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
