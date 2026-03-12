"""
theta_1-main README 兼容 API 路由

提供与 theta_1-main 文档一致的 Agent 接口，映射到现有 langgraph 服务。
- /api/agent/* : LangChain Agent v3
- /api/chat/v2, /api/interpret/*, /api/vision/* : Legacy
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.chat_service import chat_service
from ..schemas.agent import ChatRequest
from ..core.logging import get_logger

logger = get_logger(__name__)

# Agent v3 路由，挂载于 /api/agent
agent_router = APIRouter()


# ============== Request Models (theta_1-main 兼容) ==============

class AgentChatRequest(BaseModel):
    """theta_1-main 格式：message, job_id, session_id"""
    message: str
    job_id: str = ""
    session_id: Optional[str] = None


class AgentChatResponse(BaseModel):
    """theta_1-main 格式"""
    job_id: str = ""
    session_id: str = ""
    message: str
    status: str = "ok"


# ============== Agent v3 Endpoints ==============

@agent_router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """Agent chat (auto tool-calling) - 映射到现有 chat"""
    try:
        req = ChatRequest(
            message=request.message,
            context={"job_id": request.job_id, "session_id": request.session_id}
        )
        resp = chat_service.process_message(req)
        return AgentChatResponse(
            job_id=request.job_id or "",
            session_id=request.session_id or "",
            message=resp.message,
            status="ok"
        )
    except Exception as e:
        logger.error(f"Agent chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@agent_router.post("/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    """Agent chat with SSE streaming - 占位，返回非流式"""
    try:
        req = ChatRequest(
            message=request.message,
            context={"job_id": request.job_id, "session_id": request.session_id}
        )
        resp = chat_service.process_message(req)
        return {"message": resp.message, "status": "ok"}
    except Exception as e:
        logger.error(f"Agent chat stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@agent_router.get("/sessions")
async def agent_list_sessions():
    """List active sessions"""
    try:
        from .routes import _conversation_history
        return {"sessions": list(_conversation_history.keys())}
    except Exception:
        return {"sessions": []}


@agent_router.delete("/sessions/{session_id}")
async def agent_clear_session(session_id: str):
    """Clear session history"""
    try:
        from .routes import _conversation_history
        if session_id in _conversation_history:
            del _conversation_history[session_id]
    except Exception:
        pass
    return {"message": "Session cleared", "session_id": session_id}


@agent_router.get("/tools")
async def agent_list_tools():
    """List available tools"""
    return {
        "tools": [
            {"name": "list_datasets", "description": "列出可用数据集"},
            {"name": "list_experiments", "description": "列出实验"},
            {"name": "clean_data", "description": "数据清洗"},
            {"name": "prepare_data", "description": "数据准备"},
            {"name": "train_theta", "description": "训练 THETA 模型"},
            {"name": "train_baseline", "description": "训练基线模型"},
            {"name": "visualize", "description": "生成可视化"},
            {"name": "evaluate_model", "description": "评估模型"},
            {"name": "compare_models", "description": "对比模型"},
            {"name": "get_training_results", "description": "获取训练结果"},
            {"name": "list_visualizations", "description": "列出可视化"},
        ]
    }
