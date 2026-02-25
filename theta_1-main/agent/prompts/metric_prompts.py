"""
Metric Interpretation Prompts
指标解读提示词模板
"""

# 指标解读模板
METRIC_INTERPRETATION_PROMPTS = {
    "zh": {
        "system": """你是一位专业的主题模型评估专家。你的任务是将技术指标转化为业务可理解的解读。

**解读原则：**
- 使用通俗易懂的语言，避免技术术语
- 结合具体数值给出评价
- 提供改进建议（如果指标较差）
- 关注业务价值而非算法细节""",
        
        "user_template": """请解读以下主题模型评估指标：

{metrics_text}

要求：
1. 对每个指标给出质量评价（优秀/良好/一般/较差）
2. 用一句话解释该指标的业务含义
3. 如果指标较差，给出改进建议
4. 最后给出整体评价"""
    },
    
    "en": {
        "system": """You are a professional topic model evaluation expert. Your task is to translate technical metrics into business-understandable interpretations.

**Interpretation Principles:**
- Use plain language, avoid technical jargon
- Provide evaluation based on specific values
- Offer improvement suggestions if metrics are poor
- Focus on business value rather than algorithm details""",
        
        "user_template": """Please interpret the following topic model evaluation metrics:

{metrics_text}

Requirements:
1. Give quality assessment for each metric (Excellent/Good/Fair/Poor)
2. Explain the business meaning of each metric in one sentence
3. Provide improvement suggestions if metrics are poor
4. Give an overall assessment at the end"""
    }
}

# 单个指标解读模板
SINGLE_METRIC_PROMPTS = {
    "topic_coherence_npmi": {
        "zh": "NPMI连贯性为{value:.4f}，{quality_text}。这表示主题内词语的语义关联{relation_text}。",
        "en": "NPMI coherence is {value:.4f}, {quality_text}. This indicates {relation_text} semantic association between words within topics."
    },
    "topic_diversity_td": {
        "zh": "主题多样性(TD)为{value:.4f}，{quality_text}。这表示不同主题之间的区分度{relation_text}。",
        "en": "Topic diversity (TD) is {value:.4f}, {quality_text}. This indicates {relation_text} distinction between different topics."
    },
    "topic_diversity_irbo": {
        "zh": "排名多样性(iRBO)为{value:.4f}，{quality_text}。这表示主题词排名的差异程度{relation_text}。",
        "en": "Ranking diversity (iRBO) is {value:.4f}, {quality_text}. This indicates {relation_text} difference in topic word rankings."
    },
    "perplexity": {
        "zh": "困惑度为{value:.2f}，{quality_text}。这表示模型对文档的预测能力{relation_text}。",
        "en": "Perplexity is {value:.2f}, {quality_text}. This indicates {relation_text} model prediction capability."
    }
}

# 质量等级文本
QUALITY_TEXT = {
    "excellent": {"zh": "表现优秀", "en": "excellent"},
    "good": {"zh": "表现良好", "en": "good"},
    "fair": {"zh": "表现一般", "en": "fair"},
    "poor": {"zh": "表现较差", "en": "poor"}
}

# 关系文本
RELATION_TEXT = {
    "excellent": {"zh": "非常强", "en": "very strong"},
    "good": {"zh": "较强", "en": "strong"},
    "fair": {"zh": "一般", "en": "moderate"},
    "poor": {"zh": "较弱", "en": "weak"}
}
