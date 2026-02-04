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
from .core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
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
        logger.info(f"OSS enabled: {settings.OSS_BUCKET_NAME}")
    else:
        logger.info("OSS not configured, using local storage")
    
    # 检查 PAI 配置
    if settings.PAI_ENABLED:
        logger.info(f"PAI-DLC enabled: {settings.PAI_REGION}")
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
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
