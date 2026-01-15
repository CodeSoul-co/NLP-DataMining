"""
Chat Service
Handles conversational interactions and intent parsing
"""

import re
from typing import Optional, Dict, Any, Tuple
from ..schemas.agent import ChatRequest, ChatResponse, TaskRequest
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class ChatService:
    """
    Service for handling chat interactions
    Parses user intent and generates appropriate responses/actions
    """
    
    INTENT_PATTERNS = {
        "train": [
            r"train.*(?:on|with|using)?\s*(\w+)",
            r"run.*pipeline.*(\w+)",
            r"start.*training.*(\w+)",
            r"analyze.*(\w+)",
        ],
        "status": [
            r"status.*task",
            r"how.*going",
            r"progress",
            r"what.*running",
        ],
        "results": [
            r"show.*results?",
            r"get.*results?",
            r"view.*results?",
            r"results?\s+(?:for|of)?\s*(\w+)",
        ],
        "topics": [
            r"show.*topics?",
            r"what.*topics?",
            r"topic.*words?",
        ],
        "datasets": [
            r"list.*datasets?",
            r"available.*datasets?",
            r"what.*datasets?",
        ],
        "help": [
            r"help",
            r"what.*can.*do",
            r"how.*use",
        ],
    }
    
    def __init__(self):
        self.conversation_context: Dict[str, Any] = {}
    
    def parse_intent(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Parse user message to determine intent and extract parameters"""
        message_lower = message.lower().strip()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    params = {}
                    if match.groups():
                        params["dataset"] = match.group(1)
                    return intent, params
        
        return "unknown", {}
    
    def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message and generate response"""
        intent, params = self.parse_intent(request.message)
        
        if intent == "train":
            return self._handle_train_intent(params, request.context)
        elif intent == "status":
            return self._handle_status_intent()
        elif intent == "results":
            return self._handle_results_intent(params)
        elif intent == "topics":
            return self._handle_topics_intent(params)
        elif intent == "datasets":
            return self._handle_datasets_intent()
        elif intent == "help":
            return self._handle_help_intent()
        else:
            return self._handle_unknown_intent(request.message)
    
    def _handle_train_intent(self, params: Dict, context: Optional[Dict]) -> ChatResponse:
        """Handle training request"""
        dataset = params.get("dataset")
        
        if not dataset:
            available = settings.get_available_datasets()
            if available:
                return ChatResponse(
                    message=f"Which dataset would you like to train on? Available datasets: {', '.join(available)}",
                    action="request_dataset"
                )
            else:
                return ChatResponse(
                    message="No datasets found. Please upload a dataset first.",
                    action=None
                )
        
        dataset_dir = settings.DATA_DIR / dataset
        if not dataset_dir.exists():
            available = settings.get_available_datasets()
            return ChatResponse(
                message=f"Dataset '{dataset}' not found. Available datasets: {', '.join(available)}",
                action=None
            )
        
        return ChatResponse(
            message=f"Starting ETM training on dataset '{dataset}' with default settings (20 topics, zero_shot mode). You can monitor the progress in real-time.",
            action="start_task",
            data={
                "dataset": dataset,
                "mode": "zero_shot",
                "num_topics": 20
            }
        )
    
    def _handle_status_intent(self) -> ChatResponse:
        """Handle status query"""
        from ..agents.etm_agent import etm_agent
        
        tasks = etm_agent.get_all_tasks()
        running_tasks = [t for t in tasks.values() if t.get("status") == "running"]
        
        if not running_tasks:
            return ChatResponse(
                message="No tasks are currently running.",
                action="show_tasks"
            )
        
        status_lines = []
        for task in running_tasks:
            status_lines.append(
                f"- Task {task['task_id']}: {task.get('current_step', 'unknown')} "
                f"({task.get('dataset', 'unknown')}/{task.get('mode', 'unknown')})"
            )
        
        return ChatResponse(
            message=f"Currently running tasks:\n" + "\n".join(status_lines),
            action="show_tasks",
            data={"tasks": running_tasks}
        )
    
    def _handle_results_intent(self, params: Dict) -> ChatResponse:
        """Handle results query"""
        dataset = params.get("dataset")
        results = settings.get_available_results()
        
        if dataset:
            results = [r for r in results if r["dataset"] == dataset]
        
        if not results:
            return ChatResponse(
                message="No results found." + (f" for dataset '{dataset}'" if dataset else ""),
                action=None
            )
        
        result_lines = []
        for r in results[:5]:
            result_lines.append(f"- {r['dataset']}/{r['mode']}")
        
        return ChatResponse(
            message=f"Available results:\n" + "\n".join(result_lines),
            action="show_results",
            data={"results": results}
        )
    
    def _handle_topics_intent(self, params: Dict) -> ChatResponse:
        """Handle topic words query"""
        dataset = params.get("dataset")
        
        if not dataset:
            results = settings.get_available_results()
            if results:
                latest = results[0]
                dataset = latest["dataset"]
                mode = latest["mode"]
            else:
                return ChatResponse(
                    message="No results available. Please train a model first.",
                    action=None
                )
        else:
            mode = "zero_shot"
        
        return ChatResponse(
            message=f"Showing topic words for {dataset}/{mode}",
            action="show_topics",
            data={"dataset": dataset, "mode": mode}
        )
    
    def _handle_datasets_intent(self) -> ChatResponse:
        """Handle datasets listing"""
        datasets = settings.get_available_datasets()
        
        if not datasets:
            return ChatResponse(
                message="No datasets found in the data directory.",
                action=None
            )
        
        return ChatResponse(
            message=f"Available datasets: {', '.join(datasets)}",
            action="show_datasets",
            data={"datasets": datasets}
        )
    
    def _handle_help_intent(self) -> ChatResponse:
        """Handle help request"""
        help_text = """
I can help you with the following:

**Training**
- "Train on socialTwitter" - Start training on a dataset
- "Run pipeline on hatespeech with 30 topics" - Custom training

**Status**
- "What's the status?" - Check running tasks
- "Show progress" - View current task progress

**Results**
- "Show results" - List all training results
- "Show topics for socialTwitter" - View topic words

**Data**
- "List datasets" - Show available datasets

Just type your request and I'll help you!
        """
        return ChatResponse(
            message=help_text.strip(),
            action=None
        )
    
    def _handle_unknown_intent(self, message: str) -> ChatResponse:
        """Handle unknown intent"""
        return ChatResponse(
            message=f"I'm not sure what you mean by '{message}'. Type 'help' to see what I can do.",
            action=None
        )


chat_service = ChatService()
