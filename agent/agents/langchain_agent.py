"""
LangChain Topic Agent - 智能调度Agent

基于LangChain框架，通过Qwen LLM智能调度各个处理模块
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.data_tools import DataCleaningTool, DocxConverterTool
from tools.bow_tools import BowGeneratorTool
from tools.embedding_tools import EmbeddingGeneratorTool
from tools.etm_tools import ETMTrainerTool
from tools.visualization_tools import VisualizationTool
from tools.report_tools import ReportGeneratorTool
from tools.qa_tools import TextQATool, VisionQATool


# System Prompt
SYSTEM_PROMPT = """你是一个专业的主题模型分析Agent。你的任务是根据用户需求，智能调度各个处理模块完成主题分析任务。

你可以使用以下工具：
1. docx_converter - 将Word文档转换为CSV格式，输入格式: "docx路径,job_id"
2. data_cleaning - 验证和清洗数据，输入: job_id
3. bow_generator - 生成词袋表示，输入: job_id
4. embedding_generator - 生成嵌入向量，输入: job_id
5. etm_trainer - 训练ETM主题模型，输入: job_id
6. visualization_generator - 生成可视化图表，输入: job_id
7. report_generator - 生成Word报告，输入: job_id
8. text_qa - 回答主题分析问题，输入格式: "job_id,问题"
9. vision_qa - 回答可视化相关问题，输入格式: "job_id,问题"

**重要规则：**
1. 分析用户意图，决定需要调用哪些工具
2. 按照正确的依赖顺序调用工具
3. 如果用户只是提问，直接使用text_qa或vision_qa
4. 如果用户要求完整分析，需要按顺序调用所有工具

请以JSON格式返回你的决策：
{"tool": "工具名称", "input": "工具输入参数"}

如果任务完成，返回：
{"tool": "final_answer", "input": "最终回答内容"}"""


class TopicModelAgent:
    """
    主题模型分析Agent
    
    基于LangChain框架，智能调度各个处理模块
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.tools = self._init_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.conversation_history = []
    
    def _init_tools(self) -> List:
        """初始化所有工具"""
        return [
            DocxConverterTool(base_dir=self.base_dir),
            DataCleaningTool(base_dir=self.base_dir),
            BowGeneratorTool(base_dir=self.base_dir),
            EmbeddingGeneratorTool(base_dir=self.base_dir),
            ETMTrainerTool(base_dir=self.base_dir),
            VisualizationTool(base_dir=self.base_dir),
            ReportGeneratorTool(base_dir=self.base_dir),
            TextQATool(base_dir=self.base_dir),
            VisionQATool(base_dir=self.base_dir)
        ]
    
    def _call_llm(self, messages: List[Dict]) -> str:
        """调用Qwen LLM"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-ca1e46556f584e50aa74a2f6ff5659f0")
        base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.environ.get("QWEN_MODEL", "qwen-plus")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.1
        }
        
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def run(self, user_input: str) -> str:
        """
        运行Agent处理用户请求
        
        Args:
            user_input: 用户输入的请求
            
        Returns:
            Agent的最终回答
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        max_iterations = 15
        results = []
        
        for i in range(max_iterations):
            try:
                # 调用LLM获取决策
                response = self._call_llm(messages)
                print(f"  [Step {i+1}] LLM决策: {response[:100]}...")
                
                # 解析JSON响应
                try:
                    # 尝试提取JSON
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        decision = json.loads(json_str)
                    else:
                        # 没有找到JSON，视为最终回答
                        return response
                except json.JSONDecodeError:
                    return response
                
                tool_name = decision.get("tool", "")
                tool_input = decision.get("input", "")
                
                # 检查是否完成
                if tool_name == "final_answer":
                    return tool_input
                
                # 执行工具
                if tool_name in self.tool_map:
                    print(f"  [Step {i+1}] 执行工具: {tool_name}({tool_input})")
                    tool = self.tool_map[tool_name]
                    result = tool._run(tool_input)
                    results.append(f"{tool_name}: {result}")
                    print(f"  [Step {i+1}] 工具结果: {result[:100]}...")
                    
                    # 将结果添加到对话历史
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"工具执行结果: {result}\n\n请继续下一步，或返回final_answer完成任务。"})
                else:
                    return f"未知工具: {tool_name}"
                    
            except Exception as e:
                return f"Agent执行失败: {str(e)}"
        
        return "达到最大迭代次数。执行结果:\n" + "\n".join(results)
    
    def chat(self, message: str, job_id: str = None) -> str:
        """
        交互式对话
        
        Args:
            message: 用户消息
            job_id: 可选的任务ID
            
        Returns:
            Agent的回答
        """
        if job_id:
            message = f"[任务ID: {job_id}] {message}"
        return self.run(message)


def create_agent(base_dir: str = ".") -> TopicModelAgent:
    """创建Agent实例"""
    return TopicModelAgent(base_dir)


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LangChain Topic Model Agent')
    parser.add_argument('--base_dir', '-d', default='.', help='项目根目录')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互模式')
    
    args = parser.parse_args()
    
    agent = create_agent(args.base_dir)
    
    if args.interactive:
        print("=" * 60)
        print("🤖 LangChain Topic Model Agent")
        print("   输入你的请求，Agent会智能调度各个模块")
        print("   输入 'quit' 退出")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n🙋 你: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见!")
                    break
                if not user_input:
                    continue
                
                print("\n🤔 Agent思考中...")
                response = agent.run(user_input)
                print(f"\n🤖 Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
    else:
        print("使用 --interactive 或 -i 进入交互模式")
