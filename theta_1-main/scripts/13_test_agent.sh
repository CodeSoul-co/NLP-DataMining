#!/bin/bash
# THETA Agent Test Script
# Test LLM connection and agent functionality

set -e

echo "=========================================="
echo "THETA Agent Test Suite"
echo "=========================================="

cd /root/autodl-tmp/agent

# Run tests
python tests/test_llm.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
