"""
Topic Agent API Interface
严格按照THETA-main的接口规范实现

提供RESTful API接口，支持：
- 主题分析任务管理
- 文件下载（report.docx, 词云图, 热力图等）
- 交互式问答
- 主题查询
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent_integration import AgentIntegration, get_agent_integration


# ============== Request/Response Models ==============

class ChatRequest(BaseModel):
    """Chat request model - 兼容THETA-main"""
    message: str
    job_id: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    job_id: str
    session_id: str
    message: str
    status: str


class AnalysisRequest(BaseModel):
    """Analysis request model"""
    job_id: str


class AnalysisResponse(BaseModel):
    """Analysis response model"""
    job_id: str
    status: str
    message: Optional[str] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Job status response model"""
    job_id: str
    status: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class TopicWordsResponse(BaseModel):
    """Topic words response model - 兼容THETA-main"""
    topic_id: int
    words: List[Dict[str, Any]]


class TopicsListResponse(BaseModel):
    """Topics list response model"""
    job_id: str
    topics: List[Dict[str, Any]]


# ============== FastAPI Application ==============

app = FastAPI(
    title="Topic Agent API",
    description="主题模型分析Agent API接口 - 兼容THETA-main规范",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global integration instance
_integration = None


def get_integration() -> AgentIntegration:
    """Get or create AgentIntegration instance"""
    global _integration
    if _integration is None:
        base_dir = os.environ.get("THETA_ROOT", str(Path(__file__).parent.parent))
        _integration = AgentIntegration(base_dir=base_dir)
    return _integration


# ============== Health Check ==============

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "version": "1.0.0", "service": "topic_agent"}


# ============== Analysis Endpoints ==============

@app.post("/analyze", response_model=AnalysisResponse)
def run_analysis(request: AnalysisRequest, integration: AgentIntegration = Depends(get_integration)):
    """
    运行完整的主题分析流程
    
    对应THETA-main的 /analyze 接口
    """
    try:
        result = integration.run_full_analysis(request.job_id)
        return AnalysisResponse(
            job_id=request.job_id,
            status=result.get("status", "unknown"),
            message="Analysis completed" if result.get("status") == "success" else None,
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str, integration: AgentIntegration = Depends(get_integration)):
    """
    获取任务状态
    """
    try:
        status = integration.get_job_status(job_id)
        return JobStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
def list_jobs(integration: AgentIntegration = Depends(get_integration)):
    """
    列出所有任务
    """
    try:
        return integration.list_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Chat/Query Endpoints ==============

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, integration: AgentIntegration = Depends(get_integration)):
    """
    交互式问答接口
    
    对应THETA-main的 /chat 接口
    """
    try:
        result = integration.handle_query(request.job_id, request.message)
        
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return ChatResponse(
            job_id=request.job_id,
            session_id=session_id,
            message=result.get("answer", result.get("error", "No response")),
            status=result.get("status", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(
    job_id: str = Body(...),
    question: str = Body(...),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    查询接口 - 简化版
    
    对应THETA-main的 /query 接口
    """
    try:
        result = integration.handle_query(job_id, question)
        return {
            "job_id": job_id,
            "question": question,
            "answer": result.get("answer"),
            "status": result.get("status")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Topics Endpoints ==============

@app.get("/jobs/{job_id}/topics", response_model=TopicsListResponse)
def get_topics(job_id: str, integration: AgentIntegration = Depends(get_integration)):
    """
    获取任务的所有主题
    """
    try:
        result = integration.get_analysis_result(job_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return TopicsListResponse(
            job_id=job_id,
            topics=result.get("topics", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/topics/{topic_id}/words", response_model=TopicWordsResponse)
def get_topic_words(
    job_id: str, 
    topic_id: int, 
    top_k: int = Query(10, ge=1, le=50),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    获取指定主题的关键词
    
    对应THETA-main的 /topics/{topic_id}/words 接口
    """
    try:
        result = integration.get_analysis_result(job_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        topics = result.get("topics", [])
        topic = next((t for t in topics if t.get("id") == topic_id), None)
        
        if not topic:
            raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")
        
        keywords = topic.get("keywords", [])[:top_k]
        
        return TopicWordsResponse(
            topic_id=topic_id,
            words=[{"word": w, "weight": 1.0 / (i + 1)} for i, w in enumerate(keywords)]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Download Endpoints ==============

@app.get("/api/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str, integration: AgentIntegration = Depends(get_integration)):
    """
    文件下载接口
    
    支持下载：
    - report.docx - Word报告
    - wordcloud_topic_{i}.png - 词云图
    - topic_distribution.png - 主题分布图
    - heatmap_doc_topic.png - 热力图
    - coherence_curve.png - 一致性曲线
    - topic_similarity.png - 主题相似度矩阵
    - theta.csv - 文档-主题分布
    - beta.csv - 主题-词分布
    - analysis_result.json - 分析结果
    """
    try:
        file_path = integration.get_download_path(job_id, filename)
        
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found for job {job_id}")
        
        # Determine media type
        ext = Path(filename).suffix.lower()
        media_types = {
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".csv": "text/csv",
            ".json": "application/json",
            ".pdf": "application/pdf"
        }
        media_type = media_types.get(ext, "application/octet-stream")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/report")
def download_report(job_id: str, integration: AgentIntegration = Depends(get_integration)):
    """
    下载Word报告的便捷接口
    """
    return download_file(job_id, "report.docx", integration)


@app.get("/jobs/{job_id}/charts")
def get_charts_list(job_id: str, integration: AgentIntegration = Depends(get_integration)):
    """
    获取所有可用图表列表
    """
    try:
        result = integration.get_analysis_result(job_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        charts = result.get("charts", {})
        topics = result.get("topics", [])
        
        # Add wordcloud URLs
        wordclouds = [t.get("wordcloud_url") for t in topics if t.get("wordcloud_url")]
        
        return {
            "job_id": job_id,
            "charts": charts,
            "wordclouds": wordclouds,
            "downloads": result.get("downloads", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Analysis Result Endpoint ==============

@app.get("/jobs/{job_id}/result")
def get_analysis_result(job_id: str, integration: AgentIntegration = Depends(get_integration)):
    """
    获取完整的分析结果
    """
    try:
        result = integration.get_analysis_result(job_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
