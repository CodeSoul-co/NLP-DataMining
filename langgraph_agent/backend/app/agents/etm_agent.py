"""
ETM Agent - LangGraph Implementation
Defines the complete ETM pipeline as a directed graph
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, AsyncGenerator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path("/root/autodl-tmp/ETM")))

from ..schemas.agent import AgentState, TaskRequest, TaskStatus, StepStatus
from ..core.config import settings
from ..core.logging import get_logger
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
    Supports async execution with real-time status updates
    """
    
    def __init__(self, use_checkpointer: bool = True):
        self.use_checkpointer = use_checkpointer
        self.checkpointer = MemorySaver() if use_checkpointer else None
        self.active_tasks: Dict[str, AgentState] = {}
        self.callbacks: Dict[str, Callable] = {}
        
    def register_callback(self, task_id: str, callback: Callable):
        """Register a callback for task updates"""
        self.callbacks[task_id] = callback
    
    def unregister_callback(self, task_id: str):
        """Unregister a callback"""
        self.callbacks.pop(task_id, None)
    
    async def _step_callback(self, task_id: str, step: str, status: str, message: str, **kwargs):
        """Internal callback to notify registered listeners"""
        if task_id in self.callbacks:
            await self.callbacks[task_id](step, status, message, **kwargs)
    
    async def run_pipeline(
        self, 
        request: TaskRequest,
        callback: Optional[Callable] = None
    ) -> AgentState:
        """
        Run the complete ETM pipeline
        
        Args:
            request: Task request with configuration
            callback: Optional async callback for status updates
        
        Returns:
            Final agent state with results
        """
        initial_state = create_initial_state(request)
        task_id = initial_state["task_id"]
        
        self.active_tasks[task_id] = initial_state
        
        if callback:
            self.register_callback(task_id, callback)
        
        async def wrapped_callback(step: str, status: str, message: str, **kwargs):
            await self._step_callback(task_id, step, status, message, **kwargs)
        
        try:
            logger.info(f"Starting pipeline for task {task_id}")
            logger.info(f"Dataset: {request.dataset}, Mode: {request.mode}, Topics: {request.num_topics}")
            
            workflow = create_etm_graph(wrapped_callback if callback else None)
            
            if self.use_checkpointer:
                app = workflow.compile(checkpointer=self.checkpointer)
                config = {"configurable": {"thread_id": task_id}}
                final_state = await app.ainvoke(initial_state, config)
            else:
                app = workflow.compile()
                final_state = await app.ainvoke(initial_state)
            
            self.active_tasks[task_id] = final_state
            
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
            return error_state
            
        finally:
            self.unregister_callback(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[AgentState]:
        """Get current status of a task"""
        return self.active_tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, AgentState]:
        """Get all active tasks"""
        return self.active_tasks.copy()
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.active_tasks:
            state = self.active_tasks[task_id]
            if state.get("status") == "running":
                self.active_tasks[task_id] = {
                    **state,
                    "status": "cancelled",
                    "updated_at": datetime.now().isoformat()
                }
                return True
        return False
    
    async def resume_task(self, task_id: str) -> Optional[AgentState]:
        """Resume a task from checkpoint (if checkpointer is enabled)"""
        if not self.use_checkpointer:
            logger.warning("Checkpointer not enabled, cannot resume task")
            return None
        
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found")
            return None
        
        state = self.active_tasks[task_id]
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
        
        final_state = await app.ainvoke(resumed_state, config)
        self.active_tasks[task_id] = final_state
        
        return final_state


etm_agent = ETMAgent(use_checkpointer=True)
