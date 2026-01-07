"""
LLM Client

Provides interface for interacting with large language models, supporting different models and APIs.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union


class LLMClient:
    """
    Large language model client that provides interface for interacting with LLMs.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        dev_mode: bool = False
    ):
        """
        Initialize large language model client.
        
        Args:
            model_name: Model name
            api_key: API key
            api_base: API base URL
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            dev_mode: Whether to enable development mode (print debug information)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dev_mode = dev_mode
        
        # Check API key
        if not self.api_key and not self.dev_mode:
            print("[WARNING] No API key provided. Using mock responses.")
        
        if self.dev_mode:
            print(f"[LLMClient] Initialized with model: {model_name}")
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate text response.
        
        Args:
            prompt: Prompt text
            context: Context information
            tools: List of available tools
            
        Returns:
            Dictionary containing response content
        """
        if not self.api_key:
            return self._mock_response(prompt, context, tools)
        
        try:
            # Build messages
            messages = self._build_messages(prompt, context)
            
            # Build request
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Add tools (if provided)
            if tools:
                request_data["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }
                    for tool in tools
                ]
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            api_url = f"{self.api_base}/v1/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data
            )
            
            # Parse response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract content
                message = response_data["choices"][0]["message"]
                content = message.get("content", "")
                
                # Extract tool calls
                tool_calls = message.get("tool_calls", [])
                
                result = {
                    "content": content
                }
                
                if tool_calls:
                    result["tool_calls"] = [
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"])
                        }
                        for tool_call in tool_calls
                    ]
                
                return result
            else:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                if self.dev_mode:
                    print(f"[LLMClient] {error_message}")
                
                return {
                    "content": f"抱歉，我遇到了一个问题：{error_message}",
                    "error": error_message
                }
        
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "content": "抱歉，我遇到了一个问题，无法生成响应。",
                "error": error_message
            }
    
    def generate_with_tool_results(
        self,
        original_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate final response using tool results.
        
        Args:
            original_response: Original response
            tool_results: Tool call results
            context: Context information
            
        Returns:
            Final response
        """
        if not self.api_key:
            return self._mock_tool_response(original_response, tool_results)
        
        try:
            # Build messages
            messages = self._build_messages("", context)
            
            # Add original response
            messages.append({
                "role": "assistant",
                "content": original_response.get("content", ""),
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"])
                        }
                    }
                    for i, tool_call in enumerate(original_response.get("tool_calls", []))
                ]
            })
            
            # Add tool results
            for i, result in enumerate(tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "name": result["name"],
                    "content": json.dumps(result["result"])
                })
            
            # Build request
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Send request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            api_url = f"{self.api_base}/v1/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data
            )
            
            # Parse response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract content
                message = response_data["choices"][0]["message"]
                content = message.get("content", "")
                
                return {
                    "content": content
                }
            else:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                if self.dev_mode:
                    print(f"[LLMClient] {error_message}")
                
                return {
                    "content": f"Sorry, I encountered a problem: {error_message}",
                    "error": error_message
                }
        
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "content": "Sorry, I encountered a problem and cannot generate a response.",
                "error": error_message
            }
    
    def generate_plan(
        self,
        planning_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate plan.
        
        Args:
            planning_prompt: Planning prompt
            context: Context information
            
        Returns:
            Plan
        """
        if not self.api_key:
            return self._mock_plan(planning_prompt)
        
        try:
            # Build messages
            messages = self._build_messages(planning_prompt, context)
            
            # Add system prompt
            messages.insert(0, {
                "role": "system",
                "content": "You are a professional planning assistant. Please create a detailed response plan based on the user's request."
            })
            
            # Build request
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.5,  # Lower temperature for more deterministic planning
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"}  # Request JSON format response
            }
            
            # Send request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            api_url = f"{self.api_base}/v1/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data
            )
            
            # Parse response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract content
                content = response_data["choices"][0]["message"]["content"]
                
                # Parse JSON
                try:
                    plan = json.loads(content)
                    return plan
                except json.JSONDecodeError:
                    if self.dev_mode:
                        print(f"[LLMClient] Failed to parse JSON: {content}")
                    
                    return {
                        "steps": [],
                        "error": "Failed to parse JSON response"
                    }
            else:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                if self.dev_mode:
                    print(f"[LLMClient] {error_message}")
                
                return {
                    "steps": [],
                    "error": error_message
                }
        
        except Exception as e:
            error_message = f"Error generating plan: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "steps": [],
                "error": error_message
            }
    
    def generate_from_plan_results(
        self,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate final response based on plan execution results.
        
        Args:
            plan: Plan
            results: Execution results
            context: Context information
            
        Returns:
            Final response
        """
        if not self.api_key:
            return self._mock_plan_response(plan, results)
        
        try:
            # Build prompt
            prompt = "Please generate a final response based on the following plan and execution results:\n\n"
            
            # Add plan
            prompt += "Plan:\n"
            prompt += json.dumps(plan, ensure_ascii=False, indent=2)
            prompt += "\n\n"
            
            # Add execution results
            prompt += "Execution results:\n"
            prompt += json.dumps(results, ensure_ascii=False, indent=2)
            prompt += "\n\n"
            
            # Add instruction
            prompt += "Please generate a comprehensive response combining the above information."
            
            # Generate response
            return self.generate(prompt, context)
        
        except Exception as e:
            error_message = f"Error generating response from plan results: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "content": "Sorry, I encountered a problem and cannot generate a response.",
                "error": error_message
            }
    
    def _build_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build message list.
        
        Args:
            prompt: Prompt text
            context: Context information
            
        Returns:
            Message list
        """
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": "You are an intelligent assistant based on topic models, capable of understanding user's topic interests and providing relevant answers."
        })
        
        # Add conversation history
        if context and "conversation_history" in context:
            for entry in context["conversation_history"]:
                messages.append({
                    "role": "user",
                    "content": entry["user_input"]
                })
                
                messages.append({
                    "role": "assistant",
                    "content": entry["response"]
                })
        
        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _mock_response(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate mock response (for development mode).
        
        Args:
            prompt: Prompt text
            context: Context information
            tools: List of available tools
            
        Returns:
            Mock response
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock response for prompt: {prompt[:50]}...")
        
        # Simple mock response
        response = {
            "content": f"This is a mock response for '{prompt[:20]}...'. In actual application, this would return the real response from the large language model."
        }
        
        # If tools are provided, randomly select a tool call
        if tools and "search" in prompt.lower():
            response["tool_calls"] = [
                {
                    "name": "search",
                    "arguments": {
                        "query": prompt[:30]
                    }
                }
            ]
        
        return response
    
    def _mock_tool_response(
        self,
        original_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate mock tool response (for development mode).
        
        Args:
            original_response: Original response
            tool_results: Tool call results
            
        Returns:
            Mock response
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock tool response")
        
        # Simple mock response
        tool_names = [result["name"] for result in tool_results]
        
        return {
            "content": f"Based on the results of tools {', '.join(tool_names)}, I can provide the following information:\n\n"
                      f"This is a mock response. In actual application, this would return the real response from the large language model based on tool results."
        }
    
    def _mock_plan(
        self,
        planning_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate mock plan (for development mode).
        
        Args:
            planning_prompt: Planning prompt
            
        Returns:
            Mock plan
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock plan for prompt: {planning_prompt[:50]}...")
        
        # Simple mock plan
        return {
            "user_intent": "User wants to learn about a topic",
            "steps": [
                {
                    "type": "tool_call",
                    "params": {
                        "name": "search",
                        "arguments": {
                            "query": "mock search query"
                        }
                    }
                },
                {
                    "type": "llm_call",
                    "params": {
                        "prompt": "Generate response based on search results"
                    }
                }
            ]
        }
    
    def _mock_plan_response(
        self,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate mock plan response (for development mode).
        
        Args:
            plan: Plan
            results: Execution results
            
        Returns:
            Mock response
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock plan response")
        
        # Simple mock response
        return {
            "content": "This is a mock response based on the execution plan. In actual application, this would return the real response from the large language model based on plan execution results."
        }
