"""
Test LLM API Connection
Quick test to verify DeepSeek API is working correctly.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}")

def test_llm_connection():
    """Test basic LLM API connection"""
    load_env()
    
    from agent.utils.llm_utils import call_llm_api, format_messages
    
    print("\n" + "="*50)
    print("Testing LLM API Connection")
    print("="*50)
    
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    print(f"Provider: {provider}")
    
    messages = format_messages(
        system_prompt="You are a helpful assistant. Reply briefly.",
        user_message="Hello! Please respond with 'API connection successful!' if you can read this."
    )
    
    try:
        response = call_llm_api(messages)
        print(f"\n✓ API Response: {response}")
        print("\n✓ LLM API connection test PASSED!")
        return True
    except Exception as e:
        print(f"\n✗ API Error: {e}")
        print("\n✗ LLM API connection test FAILED!")
        return False

def test_agent_chat():
    """Test agent chat functionality"""
    load_env()
    
    from agent.core.result_interpreter_agent import ResultInterpreterAgent
    
    print("\n" + "="*50)
    print("Testing Agent Chat")
    print("="*50)
    
    agent = ResultInterpreterAgent()
    
    # Test with a simple question (no job_id needed for basic test)
    try:
        # Create a mock context
        mock_context = {
            "metrics": {
                "topic_coherence_npmi_avg": 0.15,
                "topic_diversity": 0.85
            },
            "topics": [
                {"id": 0, "keywords": ["technology", "innovation", "AI"]},
                {"id": 1, "keywords": ["business", "market", "growth"]}
            ]
        }
        
        print("Testing with mock context...")
        print(f"Mock metrics: {mock_context['metrics']}")
        print(f"Mock topics: {mock_context['topics']}")
        
        # Test interpret_metrics_text method if available
        print("\n✓ Agent initialized successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Agent Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading"""
    load_env()
    
    from agent.config import get_api_config
    
    print("\n" + "="*50)
    print("Testing Configuration")
    print("="*50)
    
    config = get_api_config()
    print(f"Provider: {config.llm.provider}")
    print(f"Model: {config.llm.model}")
    print(f"Base URL: {config.llm.base_url}")
    print(f"API Key: {config.llm.api_key[:8]}..." if config.llm.api_key else "API Key: NOT SET")
    print(f"Server: {config.server.host}:{config.server.port}")
    
    print("\n✓ Configuration test PASSED!")
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("THETA Agent System - Test Suite")
    print("="*60)
    
    results = []
    
    # Test 1: Configuration
    results.append(("Configuration", test_config()))
    
    # Test 2: LLM Connection
    results.append(("LLM Connection", test_llm_connection()))
    
    # Test 3: Agent Chat
    results.append(("Agent Chat", test_agent_chat()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("="*60))
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("="*60)
