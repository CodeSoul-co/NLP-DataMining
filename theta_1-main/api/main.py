"""
THETA API 主入口
提供完整的前端对接 API 服务

功能：
1. 数据上传 API（前端直传 OSS）
2. DLC 训练任务自动提交
3. 任务状态查询
4. 结果下载
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.data_api import router as data_router
from api.auth_router import router as auth_router

# 创建 FastAPI 应用
app = FastAPI(
    title="THETA Topic Model API",
    description="主题模型训练 API - 支持前端直传 OSS + DLC 自动训练",
    version="2.12.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(data_router)


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "service": "THETA Topic Model API",
        "version": "2.12.0",
        "endpoints": {
            "login": "POST /api/auth/login",
            "logout": "POST /api/auth/logout",
            "me": "GET /api/auth/me",
            "presigned_url": "POST /api/data/presigned-url",
            "upload_complete": "POST /api/data/upload-complete",
            "job_status": "GET /api/data/jobs/{job_id}/status",
            "job_results": "GET /api/data/jobs/{job_id}/results",
            "list_jobs": "GET /api/data/jobs"
        },
        "test_account": {
            "username": "admin",
            "password": "admin123"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "theta-api"}


@app.get("/config")
async def get_config():
    """获取配置信息（不含敏感信息）"""
    return {
        "oss_bucket": os.environ.get("OSS_BUCKET", "theta-prod-20260123"),
        "oss_endpoint": os.environ.get("OSS_ENDPOINT", "oss-cn-shanghai.aliyuncs.com"),
        "dlc_workspace_id": os.environ.get("DLC_WORKSPACE_ID", "464377"),
        "dlc_region": "cn-shanghai",
        "default_num_topics": 20,
        "default_epochs": 100,
        "supported_models": ["theta", "lda", "hdp", "btm", "etm", "ctm", "dtm", "nvdm", "gsm", "prodlda", "bertopic"],
        "supported_modes": ["zero_shot", "supervised", "unsupervised"],
        "supported_model_sizes": ["0.6B", "4B", "8B"]
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    print(f"Starting THETA API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
