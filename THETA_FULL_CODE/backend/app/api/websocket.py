"""
WebSocket Routes
Real-time communication for task status updates
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..agents.etm_agent import etm_agent
from ..schemas.agent import TaskRequest, StepUpdate, StepStatus
from ..core.logging import get_logger

logger = get_logger(__name__)
websocket_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.broadcast_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, task_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        if task_id:
            if task_id not in self.active_connections:
                self.active_connections[task_id] = set()
            self.active_connections[task_id].add(websocket)
        else:
            self.broadcast_connections.add(websocket)
        
        logger.info(f"WebSocket connected: task_id={task_id}")
    
    def disconnect(self, websocket: WebSocket, task_id: str = None):
        """Remove a WebSocket connection"""
        if task_id and task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        else:
            self.broadcast_connections.discard(websocket)
        
        logger.info(f"WebSocket disconnected: task_id={task_id}")
    
    async def send_to_task(self, task_id: str, message: dict):
        """Send message to all connections watching a specific task"""
        if task_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)
            
            for conn in disconnected:
                self.active_connections[task_id].discard(conn)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        disconnected = set()
        for connection in self.broadcast_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        for conn in disconnected:
            self.broadcast_connections.discard(conn)
        
        for task_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@websocket_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """General WebSocket endpoint for broadcasts"""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            
            elif message.get("type") == "subscribe":
                task_id = message.get("task_id")
                if task_id:
                    manager.broadcast_connections.discard(websocket)
                    if task_id not in manager.active_connections:
                        manager.active_connections[task_id] = set()
                    manager.active_connections[task_id].add(websocket)
                    await websocket.send_json({
                        "type": "subscribed",
                        "task_id": task_id
                    })
            
            elif message.get("type") == "start_task":
                request_data = message.get("request", {})
                try:
                    request = TaskRequest(**request_data)
                    
                    async def status_callback(step: str, status: str, msg: str, **kwargs):
                        update = StepUpdate(
                            task_id=kwargs.get("task_id", ""),
                            step=step,
                            status=StepStatus(status),
                            message=msg,
                            progress=kwargs.get("progress"),
                            details=kwargs
                        )
                        await manager.send_to_task(update.task_id, update.model_dump(mode="json"))
                        await manager.broadcast({
                            "type": "task_update",
                            **update.model_dump(mode="json")
                        })
                    
                    asyncio.create_task(etm_agent.run_pipeline(request, status_callback))
                    
                    await websocket.send_json({
                        "type": "task_started",
                        "message": "Task started successfully"
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            elif message.get("type") == "get_status":
                task_id = message.get("task_id")
                if task_id:
                    state = etm_agent.get_task_status(task_id)
                    if state:
                        await websocket.send_json({
                            "type": "task_status",
                            "task_id": task_id,
                            "status": state.get("status"),
                            "current_step": state.get("current_step"),
                            "logs": state.get("logs", [])[-10:]
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Task {task_id} not found"
                        })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@websocket_router.websocket("/ws/task/{task_id}")
async def task_websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for a specific task"""
    await manager.connect(websocket, task_id)
    
    try:
        state = etm_agent.get_task_status(task_id)
        if state:
            await websocket.send_json({
                "type": "initial_state",
                "task_id": task_id,
                "status": state.get("status"),
                "current_step": state.get("current_step"),
                "logs": state.get("logs", [])
            })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif message.get("type") == "get_logs":
                state = etm_agent.get_task_status(task_id)
                if state:
                    await websocket.send_json({
                        "type": "logs",
                        "logs": state.get("logs", [])
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
        manager.disconnect(websocket, task_id)


async def notify_task_update(task_id: str, step: str, status: str, message: str, **kwargs):
    """Helper function to notify all subscribers of a task update"""
    update = {
        "type": "step_update",
        "task_id": task_id,
        "step": step,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    await manager.send_to_task(task_id, update)
    await manager.broadcast(update)
