#!/bin/bash
# THETA Agent Test Script
# Test LLM connection and agent functionality

set -e

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"

echo "=========================================="
echo "THETA Agent Test Suite"
echo "=========================================="

cd "$AGENT_DIR"

# Run tests
python tests/test_llm.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
