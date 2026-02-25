"""
Topic Analysis Prompts
主题分析提示词模板
"""

TOPIC_ANALYSIS_PROMPTS = {
    "zh": {
        "system": """你是一位专业的主题语义分析专家。你的任务是根据主题关键词推断主题的内容领域和业务含义。""",
        
        "single_topic": """请根据以下主题关键词，用一句话概括这个主题讨论的内容领域：

主题{topic_id}的关键词：{keywords}

要求：直接描述主题内容，使用业务化语言，控制在30字以内。""",
        
        "all_topics": """请分析以下所有主题，给出整体内容结构：

{topics_text}

要求：概述主要内容领域、识别核心主题、分析主题关联、总结讨论焦点。"""
    },
    
    "en": {
        "system": """You are a professional topic semantic analysis expert. Your task is to infer the content domain and business meaning of topics based on keywords.""",
        
        "single_topic": """Based on the following topic keywords, summarize the content area in one sentence:

Topic {topic_id} keywords: {keywords}

Requirements: Directly describe the topic content, use business language, keep it under 30 words.""",
        
        "all_topics": """Please analyze all the following topics and provide an overall content structure:

{topics_text}

Requirements: Overview main content domains, identify core topics, analyze relationships, summarize focus."""
    }
}
