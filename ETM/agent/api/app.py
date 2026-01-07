"""
Topic-Aware Agent API Interface

Provides RESTful API interface for interacting with the Topic-Aware Agent.
"""

import os
import sys
import json
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root directory to path
sys.path.append(str(Path(__file__).parents[2]))

# Import Agent components
from agent.core.topic_aware_agent import TopicAwareAgent
from agent.utils.config import AgentConfig


# Define request and response models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    use_tools: bool = True
    use_topic_enhancement: bool = True


class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str
    message: str
    dominant_topics: List[List[float]]
    topic_shifted: bool


class DocumentRequest(BaseModel):
    """Document addition request model"""
    text: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Document addition response model"""
    document_id: int
    topic_dist: List[float]


class TopicRequest(BaseModel):
    """Topic request model"""
    topic_id: int
    top_k: int = 10


class TopicWordsResponse(BaseModel):
    """Topic words response model"""
    topic_id: int
    words: List[Dict[str, Any]]


# Create FastAPI application
app = FastAPI(
    title="ETM Agent API",
    description="Topic-Aware Agent API Interface",
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

# Global Agent instance
agent = None


# Dependency: Get Agent instance
def get_agent():
    global agent
    if agent is None:
        # Load configuration from environment variables
        config = AgentConfig.from_env()
        
        # Use default values if not set in environment variables
        if not config.etm_model_path:
            config.etm_model_path = os.environ.get(
                "ETM_MODEL_PATH",
                "/root/autodl-tmp/ETM/outputs/models/etm_model.pt"
            )
        
        if not config.vocab_path:
            config.vocab_path = os.environ.get(
                "VOCAB_PATH",
                "/root/autodl-tmp/ETM/outputs/engine_a/vocab.json"
            )
        
        # Initialize Agent
        agent = TopicAwareAgent(config, dev_mode=True)
    
    return agent


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent=Depends(get_agent)):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process user input
        result = agent.process(
            user_input=request.message,
            session_id=session_id,
            use_tools=request.use_tools,
            use_topic_enhancement=request.use_topic_enhancement
        )
        
        # Build response
        response = {
            "session_id": session_id,
            "message": result["content"],
            "dominant_topics": [[topic_id, float(weight)] for topic_id, weight in result["dominant_topics"]],
            "topic_shifted": result["topic_shifted"]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add document endpoint
@app.post("/documents", response_model=DocumentResponse)
def add_document(request: DocumentRequest, agent=Depends(get_agent)):
    try:
        # Create document
        document = {
            "text": request.text,
            "metadata": request.metadata or {}
        }
        
        # Get topic distribution for the text
        topic_dist = agent.get_topic_distribution(request.text)
        
        # Add to knowledge base
        document_id = agent.add_document(document, topic_dist=topic_dist)
        
        # Build response
        response = {
            "document_id": document_id,
            "topic_dist": topic_dist.tolist()
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Query knowledge base endpoint
@app.post("/query")
def query_knowledge(query: str = Body(..., embed=True), top_k: int = 5, agent=Depends(get_agent)):
    try:
        # Query knowledge base
        results = agent.query_knowledge(query, top_k=top_k)
        
        # Build response
        response = [
            {
                "document_id": doc_id,
                "score": float(score),
                "content": document
            }
            for doc_id, score, document in results
        ]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get topic words endpoint
@app.get("/topics/{topic_id}/words", response_model=TopicWordsResponse)
def get_topic_words(topic_id: int, top_k: int = 10, agent=Depends(get_agent)):
    try:
        # Get topic words
        topic_words = agent.get_topic_words(topic_id, top_k=top_k)
        
        # Build response
        response = {
            "topic_id": topic_id,
            "words": [
                {
                    "word": word,
                    "weight": float(weight)
                }
                for word, weight in topic_words
            ]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get text topic distribution endpoint
@app.post("/analyze")
def analyze_text(text: str = Body(..., embed=True), agent=Depends(get_agent)):
    try:
        # Get topic distribution
        topic_dist = agent.get_topic_distribution(text)
        
        # Get dominant topics
        dominant_topics = agent.get_dominant_topics(topic_dist)
        
        # Build response
        response = {
            "topic_dist": topic_dist.tolist(),
            "dominant_topics": [
                {
                    "topic_id": topic_id,
                    "weight": float(weight),
                    "words": [
                        {
                            "word": word,
                            "weight": float(w)
                        }
                        for word, w in agent.get_topic_words(topic_id, top_k=5)
                    ]
                }
                for topic_id, weight in dominant_topics
            ]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main function
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port)
