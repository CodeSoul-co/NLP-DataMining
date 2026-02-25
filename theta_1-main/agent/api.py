"""
Topic Agent API Interface

Implemented strictly in accordance with the THETA-main API specifications.

Provides RESTful API endpoints supporting:
    Topic analysis task management
    File downloads (e.g., report.docx, word clouds, heatmaps)
Interactive Q&A (question-answering)
Topic querying
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

from agent.agent_integration import AgentIntegration, get_agent_integration


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


# ============== Interpretation Endpoints (新增) ==============

@app.post("/api/interpret/metrics")
def interpret_metrics(
    job_id: str = Body(...),
    language: str = Body("zh"),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    解读评估指标
    
    将技术指标转化为业务可理解的解读
    
    Args:
        job_id: 任务ID
        language: 语言 (zh/en)
    """
    try:
        from .core.result_interpreter_agent import ResultInterpreterAgent
        interpreter = ResultInterpreterAgent(
            base_dir=integration.base_dir,
            llm_config=integration.llm_config
        )
        return interpreter.interpret_metrics(job_id, language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interpret/topics")
def interpret_topics(
    job_id: str = Body(...),
    language: str = Body("zh"),
    use_llm: bool = Body(True),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    解读主题内容
    
    使用LLM生成主题语义解读
    
    Args:
        job_id: 任务ID
        language: 语言 (zh/en)
        use_llm: 是否使用LLM生成深度解读
    """
    try:
        from .core.result_interpreter_agent import ResultInterpreterAgent
        interpreter = ResultInterpreterAgent(
            base_dir=integration.base_dir,
            llm_config=integration.llm_config
        )
        return interpreter.interpret_topics(job_id, language, use_llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interpret/summary")
def generate_summary(
    job_id: str = Body(...),
    language: str = Body("zh"),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    生成分析摘要
    
    使用LLM生成业务化的分析摘要
    
    Args:
        job_id: 任务ID
        language: 语言 (zh/en)
    """
    try:
        from .core.result_interpreter_agent import ResultInterpreterAgent
        interpreter = ResultInterpreterAgent(
            base_dir=integration.base_dir,
            llm_config=integration.llm_config
        )
        return interpreter.generate_summary(job_id, language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/v2")
def chat_v2(
    job_id: str = Body(...),
    message: str = Body(...),
    session_id: Optional[str] = Body(None),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    增强版对话接口 - 支持多轮对话
    
    Args:
        job_id: 任务ID
        message: 用户消息
        session_id: 会话ID（用于多轮对话，可选）
    """
    try:
        from .core.result_interpreter_agent import ResultInterpreterAgent
        interpreter = ResultInterpreterAgent(
            base_dir=integration.base_dir,
            llm_config=integration.llm_config
        )
        return interpreter.answer_question(job_id, message, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Vision Analysis Endpoints ==============

@app.post("/api/vision/analyze")
def analyze_image_endpoint(
    image_url: str = Body(..., description="URL of the image to analyze"),
    question: str = Body(..., description="Question about the image"),
    language: str = Body("zh", description="Response language (zh/en)")
):
    """
    Analyze an image using Qwen3 Vision API.
    
    Args:
        image_url: URL of the image to analyze
        question: Question about the image
        language: Response language
    """
    try:
        from .utils.vision_utils import analyze_image
        result = analyze_image(image_url, question)
        return {
            "success": True,
            "image_url": image_url,
            "question": question,
            "answer": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vision/analyze-chart")
def analyze_chart_endpoint(
    job_id: str = Body(..., description="Job ID"),
    chart_name: str = Body(..., description="Chart filename (e.g., wordcloud_0.png)"),
    analysis_type: str = Body("general", description="Analysis type: general, wordcloud, distribution, heatmap"),
    language: str = Body("zh", description="Response language (zh/en)"),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    Analyze a chart from job results using Qwen3 Vision API.
    
    Args:
        job_id: Job ID
        chart_name: Name of the chart file
        analysis_type: Type of analysis
        language: Response language
    """
    try:
        from .utils.vision_utils import analyze_chart
        
        # Build chart path
        chart_path = Path(integration.base_dir) / "result" / job_id / chart_name
        
        if not chart_path.exists():
            raise HTTPException(status_code=404, detail=f"Chart not found: {chart_name}")
        
        result = analyze_chart(str(chart_path), analysis_type, language)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vision/analyze-job-charts")
def analyze_job_charts_endpoint(
    job_id: str = Body(..., description="Job ID"),
    language: str = Body("zh", description="Response language (zh/en)"),
    integration: AgentIntegration = Depends(get_integration)
):
    """
    Analyze all charts from a job using Qwen3 Vision API.
    
    Args:
        job_id: Job ID
        language: Response language
    """
    try:
        from .utils.vision_utils import analyze_chart
        
        result_dir = Path(integration.base_dir) / "result" / job_id
        
        if not result_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        # Find all image files
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
        charts = []
        
        for ext in image_extensions:
            charts.extend(result_dir.glob(f"*{ext}"))
        
        results = []
        for chart_path in charts:
            # Determine analysis type based on filename
            name = chart_path.stem.lower()
            if "wordcloud" in name:
                analysis_type = "wordcloud"
            elif "distribution" in name or "dist" in name:
                analysis_type = "distribution"
            elif "heatmap" in name or "heat" in name:
                analysis_type = "heatmap"
            else:
                analysis_type = "general"
            
            result = analyze_chart(str(chart_path), analysis_type, language)
            result["filename"] = chart_path.name
            results.append(result)
        
        return {
            "job_id": job_id,
            "charts_analyzed": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== LangChain Agent Endpoints (v3) ==============

class AgentChatRequest(BaseModel):
    """LangChain agent chat request"""
    message: str
    session_id: Optional[str] = "default"


class AgentChatResponse(BaseModel):
    """LangChain agent chat response"""
    session_id: str
    message: str
    status: str


def _get_theta_agent():
    """Lazy-init the LangChain THETAAgent singleton."""
    try:
        from .langchain_agent import get_agent
        return get_agent()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Agent not available: {e}")


@app.post("/api/agent/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest):
    """
    LangChain Agent 对话接口 (v3)

    智能体可以自动调用工具完成数据清洗、预处理、训练、评估、可视化等操作，
    也可以回答关于主题模型的问题。

    Args:
        message: 用户消息（自然语言指令或问题）
        session_id: 会话 ID（用于多轮对话）
    """
    agent = _get_theta_agent()
    session_id = request.session_id or "default"
    response = agent.chat(request.message, session_id=session_id)
    return AgentChatResponse(
        session_id=session_id,
        message=response,
        status="success",
    )


@app.post("/api/agent/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    """
    LangChain Agent 流式对话接口

    返回 Server-Sent Events (SSE) 格式的流式响应。
    """
    from fastapi.responses import StreamingResponse
    import json as _json

    agent = _get_theta_agent()
    session_id = request.session_id or "default"

    def event_generator():
        for chunk in agent.chat_stream(request.message, session_id=session_id):
            yield f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/api/agent/sessions/{session_id}")
def clear_agent_session(session_id: str):
    """清除指定会话的对话历史"""
    agent = _get_theta_agent()
    agent.clear_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.get("/api/agent/sessions")
def list_agent_sessions():
    """列出所有活跃会话"""
    agent = _get_theta_agent()
    return {"sessions": agent.list_sessions()}


@app.get("/api/agent/tools")
def list_agent_tools():
    """列出 Agent 可用的所有工具及其描述"""
    from .langchain_tools import ALL_TOOLS
    tools_info = []
    for t in ALL_TOOLS:
        tools_info.append({
            "name": t.name,
            "description": t.description,
        })
    return {"tools": tools_info}


# ============== Environment Loading ==============

def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Load environment on module import
load_env()


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    debug = os.environ.get("API_DEBUG", "false").lower() == "true"
    
    print(f"Starting THETA Agent API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=debug)
