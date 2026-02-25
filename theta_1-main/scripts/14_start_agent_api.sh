#!/bin/bash
# =============================================================================
# THETA Agent API Server Script
# =============================================================================
# Start the FastAPI server for AI agent interactions
#
# The Agent API provides:
#   - Natural language Q&A about topic modeling results
#   - Automated result interpretation
#   - Visualization generation via API
#   - Multi-turn conversation support
#
# API Endpoints:
#   POST /chat              - Chat with the agent
#   POST /analyze           - Analyze topic modeling results
#   GET  /models            - List available models
#   GET  /datasets          - List available datasets
#   GET  /docs              - OpenAPI documentation
#
# Environment Variables (from .env):
#   DEEPSEEK_API_KEY        - DeepSeek API key for LLM
#   QWEN_VISION_API_KEY     - Qwen Vision API key (optional)
#
# Usage:
#   ./07_start_agent_api.sh [--port 8000]
# =============================================================================

set -e

# Default values
HOST="0.0.0.0"
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host    Host address (default: 0.0.0.0)"
            echo "  --port    Port number (default: 8000)"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --port 8080"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "THETA Agent API Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

# Load environment variables
if [ -f /root/autodl-tmp/agent/.env ]; then
    echo "Loading environment from .env..."
    export $(grep -v '^#' /root/autodl-tmp/agent/.env | xargs)
fi

cd /root/autodl-tmp

# Start the API server
echo "Starting API server..."
echo "API Documentation: http://$HOST:$PORT/docs"
echo ""

python -m agent.api
