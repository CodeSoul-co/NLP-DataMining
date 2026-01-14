#!/bin/bash
# THETA - Start Script
# Starts both backend and frontend servers

set -e

echo "=========================================="
echo "  THETA - ETM Topic Model Agent System"
echo "=========================================="

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000
GPU_ID=1

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

# Activate conda environment
source activate jiqun

# Start backend
echo ""
echo "Starting backend server on port $BACKEND_PORT..."
cd /root/autodl-tmp/langgraph_agent/backend
python run.py --port $BACKEND_PORT &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend failed to start"
    exit 1
fi

echo ""
echo "=========================================="
echo "  THETA is running!"
echo "=========================================="
echo ""
echo "  Backend API:  http://localhost:$BACKEND_PORT"
echo "  API Docs:     http://localhost:$BACKEND_PORT/docs"
echo "  WebSocket:    ws://localhost:$BACKEND_PORT/api/ws"
echo ""
echo "  To start frontend:"
echo "    cd /root/autodl-tmp/langgraph_agent/frontend"
echo "    npm install && npm run dev"
echo ""
echo "  Press Ctrl+C to stop"
echo "=========================================="

# Wait for backend
wait $BACKEND_PID
