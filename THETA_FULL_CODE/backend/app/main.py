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

# Add ETM to path
sys.path.insert(0, str(Path("/root/autodl-tmp/ETM")))

from .api.routes import router
from .api.websocket import websocket_router
from .api.dlc_routes import router as dlc_router
from .core.config import settings
from .core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"GPU ID: {settings.GPU_ID}, Device: {settings.DEVICE}")
    logger.info(f"ETM Dir: {settings.ETM_DIR}")
    logger.info(f"Data Dir: {settings.DATA_DIR}")
    logger.info(f"Result Dir: {settings.RESULT_DIR}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.GPU_ID)
    
    yield
    
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    description="LangGraph-based Agent System for ETM Topic Modeling",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(websocket_router, prefix="/api")
app.include_router(dlc_router, prefix="/api")

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
        "docs": "/docs",
        "api": "/api"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
