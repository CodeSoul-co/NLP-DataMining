"""
ETM Agent - LangGraph Implementation
Defines the complete ETM pipeline as a directed graph

支持异步任务执行和持久化存储
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, AsyncGenerator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 动态添加 ETM 路径
from ..core.config import settings
sys.path.insert(0, str(settings.ETM_DIR))

from ..schemas.agent import AgentState, TaskRequest, TaskStatus, StepStatus
from ..core.logging import get_logger
from ..services.task_store import task_store
from .nodes import (
    preprocess_node,
    embedding_node,
    training_node,
    evaluation_node,
    visualization_node,
    check_mode_requirements
)

logger = get_logger(__name__)


def create_initial_state(request: TaskRequest) -> AgentState:
    """Create initial state from task request"""
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    result_dir = settings.get_result_path(request.dataset, request.mode)
    
    return AgentState(
        task_id=task_id,
        dataset=request.dataset,
        mode=request.mode,
        data_path=str(settings.DATA_DIR / request.dataset),
        result_dir=str(result_dir),
        embeddings_dir=str(result_dir / "embeddings"),
        bow_dir=str(result_dir / "bow"),
        model_dir=str(result_dir / "model"),
        evaluation_dir=str(result_dir / "evaluation"),
        visualization_dir=str(result_dir / "visualization"),
        num_topics=request.num_topics,
        vocab_size=request.vocab_size,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        hidden_dim=request.hidden_dim,
        current_step="preprocess",
        status="running",
        error_message=None,
        preprocess_completed=False,
        embedding_completed=False,
        training_completed=False,
        evaluation_completed=False,
        visualization_completed=False,
        bow_shape=None,
        vocab_size_actual=None,
        doc_embeddings_shape=None,
        theta_shape=None,
        beta_shape=None,
        metrics=None,
        topic_words=None,
        training_history=None,
        visualization_paths=None,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        completed_at=None,
        logs=[]
    )


def create_etm_graph(callback: Optional[Callable] = None) -> StateGraph:
    """
    Create the ETM pipeline graph
    
    Graph structure:
    preprocess -> embedding -> training -> evaluation -> visualization -> END
    
    With conditional edges for error handling and mode-specific requirements
    """
    
    async def preprocess_with_callback(state: AgentState) -> Dict[str, Any]:
        return await preprocess_node(state, callback)
    
    async def embedding_with_callback(state: AgentState) -> Dict[str, Any]:
        return await embedding_node(state, callback)
    
    async def training_with_callback(state: AgentState) -> Dict[str, Any]:
        return await training_node(state, callback)
    
    async def evaluation_with_callback(state: AgentState) -> Dict[str, Any]:
        return await evaluation_node(state, callback)
    
    async def visualization_with_callback(state: AgentState) -> Dict[str, Any]:
        return await visualization_node(state, callback)
    
    async def error_node(state: AgentState) -> Dict[str, Any]:
        """Handle errors gracefully"""
        return {
            "status": "failed",
            "error_message": state.get("error_message", "Unknown error"),
            "updated_at": datetime.now().isoformat()
        }
    
    def should_continue(state: AgentState) -> str:
        """Check if pipeline should continue or stop on error"""
        if state.get("status") == "failed":
            return "error"
        return "continue"
    
    def route_after_embedding(state: AgentState) -> str:
        """Route after embedding based on mode requirements"""
        if state.get("status") == "failed":
            return "error"
        return check_mode_requirements(state)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("preprocess", preprocess_with_callback)
    workflow.add_node("embedding", embedding_with_callback)
    workflow.add_node("training", training_with_callback)
    workflow.add_node("evaluation", evaluation_with_callback)
    workflow.add_node("visualization", visualization_with_callback)
    workflow.add_node("error", error_node)
    
    workflow.set_entry_point("preprocess")
    
    workflow.add_conditional_edges(
        "preprocess",
        should_continue,
        {
            "continue": "embedding",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "embedding",
        route_after_embedding,
        {
            "training": "training",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "training",
        should_continue,
        {
            "continue": "evaluation",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "evaluation",
        should_continue,
        {
            "continue": "visualization",
            "error": "error"
        }
    )
    
    workflow.add_edge("visualization", END)
    workflow.add_edge("error", END)
    
    return workflow


class ETMAgent:
    """
    ETM Agent class for managing pipeline execution
    
    支持:
    - 异步任务执行
    - 持久化任务存储 (通过 TaskStore)
    - 实时状态更新回调
    - 任务恢复和取消
    """
    
    def __init__(self, use_checkpointer: bool = True):
        self.use_checkpointer = use_checkpointer
        self.checkpointer = MemorySaver() if use_checkpointer else None
        # 内存中的活动任务（运行时快速访问）
        self.active_tasks: Dict[str, AgentState] = {}
        self.callbacks: Dict[str, Callable] = {}
        
        # 从持久化存储加载任务
        self._load_persisted_tasks()
        
    def _load_persisted_tasks(self):
        """从 TaskStore 加载持久化的任务"""
        try:
            stored_tasks = task_store.get_all_tasks()
            for task_id, task_data in stored_tasks.items():
                self.active_tasks[task_id] = task_data
            logger.info(f"Loaded {len(stored_tasks)} tasks from persistent storage")
        except Exception as e:
            logger.error(f"Failed to load persisted tasks: {e}")
        
    def register_callback(self, task_id: str, callback: Callable):
        """Register a callback for task updates"""
        self.callbacks[task_id] = callback
    
    def unregister_callback(self, task_id: str):
        """Unregister a callback"""
        self.callbacks.pop(task_id, None)
    
    async def _step_callback(self, task_id: str, step: str, status: str, message: str, **kwargs):
        """Internal callback to notify registered listeners and update storage"""
        # 更新持久化存储
        task_store.add_log(task_id, step, status, message)
        
        if task_id in self.callbacks:
            await self.callbacks[task_id](step, status, message, **kwargs)
    
    def create_task(self, request: TaskRequest) -> Dict[str, Any]:
        """
        创建新任务（不立即执行）
        
        Returns:
            包含 task_id 和初始状态的字典
        """
        initial_state = create_initial_state(request)
        task_id = initial_state["task_id"]
        
        # 存储到内存和持久化存储
        self.active_tasks[task_id] = initial_state
        task_store.create_task(task_id, dict(initial_state))
        
        logger.info(f"Created task {task_id} for dataset {request.dataset}")
        
        return initial_state
    
    async def run_pipeline(
        self, 
        request: TaskRequest,
        callback: Optional[Callable] = None,
        task_id: Optional[str] = None
    ) -> AgentState:
        """
        Run the complete ETM pipeline
        
        Args:
            request: Task request with configuration
            callback: Optional async callback for status updates
            task_id: Optional existing task_id to use
        
        Returns:
            Final agent state with results
        """
        if task_id and task_id in self.active_tasks:
            initial_state = self.active_tasks[task_id]
        else:
            initial_state = create_initial_state(request)
            task_id = initial_state["task_id"]
            self.active_tasks[task_id] = initial_state
            task_store.create_task(task_id, dict(initial_state))
        
        if callback:
            self.register_callback(task_id, callback)
        
        async def wrapped_callback(step: str, status: str, message: str, **kwargs):
            await self._step_callback(task_id, step, status, message, **kwargs)
        
        try:
            logger.info(f"Starting pipeline for task {task_id}")
            logger.info(f"Dataset: {request.dataset}, Mode: {request.mode}, Topics: {request.num_topics}")
            
            # 更新状态为运行中
            self.active_tasks[task_id]["status"] = "running"
            task_store.update_task(task_id, {"status": "running"})
            
            workflow = create_etm_graph(wrapped_callback if callback else None)
            
            if self.use_checkpointer:
                app = workflow.compile(checkpointer=self.checkpointer)
                config = {"configurable": {"thread_id": task_id}}
                final_state = await app.ainvoke(initial_state, config)
            else:
                app = workflow.compile()
                final_state = await app.ainvoke(initial_state)
            
            self.active_tasks[task_id] = final_state
            
            # 更新持久化存储
            if final_state.get("status") == "completed":
                task_store.set_completed(task_id, {
                    "metrics": final_state.get("metrics"),
                    "topic_words": final_state.get("topic_words"),
                    "visualization_paths": final_state.get("visualization_paths"),
                })
            else:
                task_store.update_task(task_id, dict(final_state))
            
            logger.info(f"Pipeline completed for task {task_id}: {final_state.get('status')}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Pipeline failed for task {task_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_state = {
                **initial_state,
                "status": "failed",
                "error_message": str(e),
                "updated_at": datetime.now().isoformat()
            }
            self.active_tasks[task_id] = error_state
            task_store.set_failed(task_id, str(e))
            
            return error_state
            
        finally:
            self.unregister_callback(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[AgentState]:
        """Get current status of a task (memory first, then storage)"""
        # 优先从内存获取
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        # 否则从持久化存储获取
        return task_store.get_task(task_id)
    
    def get_all_tasks(self) -> Dict[str, AgentState]:
        """Get all tasks (merges memory and storage)"""
        # 合并内存和持久化存储中的任务
        all_tasks = task_store.get_all_tasks()
        all_tasks.update(self.active_tasks)
        return all_tasks
    
    def get_recent_tasks(self, limit: int = 20) -> list:
        """获取最近的任务列表"""
        return task_store.get_recent_tasks(limit)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.active_tasks:
            state = self.active_tasks[task_id]
            if state.get("status") in ["running", "pending"]:
                self.active_tasks[task_id] = {
                    **state,
                    "status": "cancelled",
                    "updated_at": datetime.now().isoformat()
                }
                task_store.cancel_task(task_id)
                return True
        return False
    
    async def resume_task(self, task_id: str) -> Optional[AgentState]:
        """Resume a task from checkpoint (if checkpointer is enabled)"""
        if not self.use_checkpointer:
            logger.warning("Checkpointer not enabled, cannot resume task")
            return None
        
        state = self.get_task_status(task_id)
        if not state:
            logger.warning(f"Task {task_id} not found")
            return None
        
        if state.get("status") not in ["failed", "cancelled"]:
            logger.warning(f"Task {task_id} is not in a resumable state")
            return None
        
        logger.info(f"Resuming task {task_id} from step {state.get('current_step')}")
        
        workflow = create_etm_graph()
        app = workflow.compile(checkpointer=self.checkpointer)
        config = {"configurable": {"thread_id": task_id}}
        
        resumed_state = {
            **state,
            "status": "running",
            "error_message": None,
            "updated_at": datetime.now().isoformat()
        }
        
        self.active_tasks[task_id] = resumed_state
        task_store.update_task(task_id, {"status": "running", "error_message": None})
        
        final_state = await app.ainvoke(resumed_state, config)
        self.active_tasks[task_id] = final_state
        
        if final_state.get("status") == "completed":
            task_store.complete_task(
                task_id,
                metrics=final_state.get("metrics"),
                topic_words=final_state.get("topic_words"),
                visualization_paths=final_state.get("visualization_paths")
            )
        else:
            task_store.update_task(task_id, dict(final_state))
        
        return final_state
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务（仅限已完成/失败/取消的任务）"""
        state = self.get_task_status(task_id)
        if not state:
            return False
        
        if state.get("status") in ["running", "pending"]:
            logger.warning(f"Cannot delete running/pending task: {task_id}")
            return False
        
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        task_store.delete_task(task_id)
        return True


etm_agent = ETMAgent(use_checkpointer=True)
