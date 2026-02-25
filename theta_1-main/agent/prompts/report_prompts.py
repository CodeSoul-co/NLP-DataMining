"""
Report Generation Prompts
报告生成提示词模板
"""

REPORT_GENERATION_PROMPTS = {
    "zh": {
        "summary": """请根据以下主题模型分析结果，生成一份简洁的分析摘要：

**评估指标：**
{metrics_text}

**主题概览：**
{topics_text}

要求：
1. 概述分析的整体质量
2. 总结发现的主要主题
3. 给出关键洞察
4. 控制在200字以内""",
        
        "executive_summary": """请为以下分析结果生成一份执行摘要：

{analysis_text}

要求：
1. 面向业务决策者
2. 突出关键发现
3. 提供可行建议
4. 控制在300字以内"""
    },
    
    "en": {
        "summary": """Please generate a concise analysis summary based on the following topic model results:

**Evaluation Metrics:**
{metrics_text}

**Topic Overview:**
{topics_text}

Requirements:
1. Overview the overall quality of the analysis
2. Summarize the main topics discovered
3. Provide key insights
4. Keep it under 200 words""",
        
        "executive_summary": """Please generate an executive summary for the following analysis results:

{analysis_text}

Requirements:
1. Target business decision makers
2. Highlight key findings
3. Provide actionable recommendations
4. Keep it under 300 words"""
    }
}
