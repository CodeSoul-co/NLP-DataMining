"""
Result Interpreter Agent
Transforms technical metrics into business insights.

Responsibilities:
1. Load analysis results (metrics, topics, theta, beta)
2. Call LLM to generate business-friendly interpretations
3. Answer user questions about results
4. Generate analysis summaries

Core Functions:
- interpret_metrics(): Interpret evaluation metrics
- interpret_topics(): Interpret topic content
- generate_summary(): Generate analysis summary
- answer_question(): Answer user questions
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..config.llm_config import LLMConfigManager, LLMConfig


class ResultInterpreterAgent:
    """
    Result Interpreter Agent
    
    Transforms topic model technical outputs into business-understandable insights.
    """
    
    # Metric interpretation templates
    METRIC_TEMPLATES = {
        "topic_coherence_npmi_avg": {
            "name": "主题连贯性 (NPMI)",
            "name_en": "Topic Coherence (NPMI)",
            "range": "[-1, 1]",
            "good_threshold": 0.1,
            "excellent_threshold": 0.2,
            "interpretation": {
                "zh": "衡量主题内词语的语义关联程度。值越高表示主题内的词语在原始文档中共现频率越高，主题越连贯。",
                "en": "Measures semantic association between words within a topic. Higher values indicate more coherent topics."
            },
            "business_meaning": {
                "zh": "高连贯性意味着识别出的主题更有意义，词语之间有明确的语义联系。",
                "en": "High coherence means identified topics are more meaningful with clear semantic connections."
            }
        },
        "topic_coherence_cv_avg": {
            "name": "主题连贯性 (C_V)",
            "name_en": "Topic Coherence (C_V)",
            "range": "[0, 1]",
            "good_threshold": 0.4,
            "excellent_threshold": 0.6,
            "interpretation": {
                "zh": "基于滑动窗口的连贯性指标，考虑词语在文档中的上下文共现。",
                "en": "Sliding window based coherence metric considering contextual co-occurrence."
            },
            "business_meaning": {
                "zh": "反映主题词在实际文本中的语境关联强度。",
                "en": "Reflects contextual association strength of topic words in actual text."
            }
        },
        "topic_coherence_umass_avg": {
            "name": "主题连贯性 (UMass)",
            "name_en": "Topic Coherence (UMass)",
            "range": "(-∞, 0]",
            "good_threshold": -2.0,
            "excellent_threshold": -1.0,
            "interpretation": {
                "zh": "基于文档内词语共现的连贯性指标，值越接近0越好。",
                "en": "Document-based co-occurrence coherence. Values closer to 0 are better."
            },
            "business_meaning": {
                "zh": "衡量主题词在同一文档中出现的频率。",
                "en": "Measures how often topic words appear together in the same document."
            }
        },
        "topic_diversity_td": {
            "name": "主题多样性 (TD)",
            "name_en": "Topic Diversity (TD)",
            "range": "[0, 1]",
            "good_threshold": 0.7,
            "excellent_threshold": 0.85,
            "interpretation": {
                "zh": "衡量不同主题之间的区分度。值越高表示主题之间重叠词越少，区分度越高。",
                "en": "Measures distinction between topics. Higher values indicate less word overlap."
            },
            "business_meaning": {
                "zh": "高多样性意味着模型识别出了不同的讨论领域，而非重复的主题。",
                "en": "High diversity means the model identified distinct discussion areas, not repetitive topics."
            }
        },
        "topic_diversity_irbo": {
            "name": "主题多样性 (iRBO)",
            "name_en": "Topic Diversity (iRBO)",
            "range": "[0, 1]",
            "good_threshold": 0.7,
            "excellent_threshold": 0.85,
            "interpretation": {
                "zh": "基于排名的主题多样性指标，考虑词语排名顺序的差异。",
                "en": "Rank-based diversity metric considering word ranking order differences."
            },
            "business_meaning": {
                "zh": "反映主题之间不仅词语不同，而且重要性排序也不同。",
                "en": "Reflects that topics differ not only in words but also in importance ranking."
            }
        },
        "topic_exclusivity_avg": {
            "name": "主题排他性",
            "name_en": "Topic Exclusivity",
            "range": "[0, 1]",
            "good_threshold": 0.3,
            "excellent_threshold": 0.5,
            "interpretation": {
                "zh": "衡量每个主题的特征词是否专属于该主题。值越高表示主题特征越鲜明。",
                "en": "Measures whether topic words are exclusive to that topic. Higher values indicate more distinctive topics."
            },
            "business_meaning": {
                "zh": "高排他性意味着每个主题都有独特的标志性词语，便于理解和命名。",
                "en": "High exclusivity means each topic has unique signature words, easier to understand and name."
            }
        },
        "topic_significance_avg": {
            "name": "主题显著性",
            "name_en": "Topic Significance",
            "range": "[0, 1]",
            "good_threshold": 0.03,
            "excellent_threshold": 0.05,
            "interpretation": {
                "zh": "衡量每个主题在文档集中的重要程度。值越高表示该主题覆盖的文档越多。",
                "en": "Measures importance of each topic in the document collection."
            },
            "business_meaning": {
                "zh": "反映主题在整体讨论中的权重分布。",
                "en": "Reflects weight distribution of topics in overall discussion."
            }
        },
        "perplexity": {
            "name": "困惑度",
            "name_en": "Perplexity",
            "range": "[1, +∞)",
            "good_threshold": 500,
            "excellent_threshold": 200,
            "interpretation": {
                "zh": "衡量模型对文档的预测能力。值越低表示模型拟合越好。",
                "en": "Measures model's prediction capability. Lower values indicate better fit."
            },
            "business_meaning": {
                "zh": "低困惑度意味着模型能够很好地解释文档的词语分布。",
                "en": "Low perplexity means the model explains document word distribution well."
            },
            "lower_is_better": True
        }
    }
    
    # 质量等级定义
    QUALITY_LEVELS = {
        "excellent": {"zh": "优秀", "en": "Excellent", "emoji": "🌟"},
        "good": {"zh": "良好", "en": "Good", "emoji": "✅"},
        "fair": {"zh": "一般", "en": "Fair", "emoji": "⚠️"},
        "poor": {"zh": "较差", "en": "Poor", "emoji": "❌"}
    }
    
    def __init__(
        self, 
        base_dir: str = "/root/autodl-tmp",
        llm_config: Optional[Dict] = None
    ):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
        
        # 初始化LLM配置
        if llm_config:
            self.llm_config = LLMConfigManager.get_config(
                llm_config.get("provider", "qwen"),
                llm_config
            )
        else:
            self.llm_config = LLMConfigManager.get_qwen_config()
        
        # 对话历史（用于多轮对话）
        self.conversation_history: Dict[str, List[Dict]] = {}
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"ResultInterpreterAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def interpret_metrics(
        self, 
        job_id: str,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        解读评估指标
        
        Args:
            job_id: 任务ID
            language: 语言 (zh/en)
            
        Returns:
            包含指标解读的字典
        """
        try:
            # 加载指标数据
            metrics = self._load_metrics(job_id)
            
            interpretations = []
            overall_quality = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
            
            for metric_key, template in self.METRIC_TEMPLATES.items():
                if metric_key in metrics and metrics[metric_key] is not None:
                    value = metrics[metric_key]
                    
                    # 评估质量等级
                    quality = self._evaluate_metric_quality(metric_key, value, template)
                    overall_quality[quality] += 1
                    
                    interpretation = {
                        "metric": metric_key,
                        "name": template["name"] if language == "zh" else template["name_en"],
                        "value": value,
                        "range": template["range"],
                        "quality": quality,
                        "quality_label": self.QUALITY_LEVELS[quality][language],
                        "quality_emoji": self.QUALITY_LEVELS[quality]["emoji"],
                        "interpretation": template["interpretation"][language],
                        "business_meaning": template["business_meaning"][language]
                    }
                    interpretations.append(interpretation)
            
            # 生成总体评估
            overall_assessment = self._generate_overall_assessment(
                overall_quality, language
            )
            
            return {
                "status": "success",
                "job_id": job_id,
                "metrics_count": len(interpretations),
                "interpretations": interpretations,
                "overall_quality": overall_quality,
                "overall_assessment": overall_assessment
            }
            
        except Exception as e:
            self.logger.error(f"Failed to interpret metrics: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e)
            }
    
    def interpret_topics(
        self, 
        job_id: str,
        language: str = "zh",
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        解读主题内容
        
        Args:
            job_id: 任务ID
            language: 语言 (zh/en)
            use_llm: 是否使用LLM生成深度解读
            
        Returns:
            包含主题解读的字典
        """
        try:
            # 加载主题数据
            topics = self._load_topics(job_id)
            analysis_result = self._load_analysis_result(job_id)
            
            topic_interpretations = []
            
            for topic in topics:
                topic_id = topic.get("id", topic.get("topic_id", 0))
                keywords = topic.get("keywords", topic.get("words", []))
                proportion = topic.get("proportion", 0)
                
                # 基础解读
                interpretation = {
                    "topic_id": topic_id,
                    "keywords": keywords[:10] if isinstance(keywords, list) else keywords,
                    "proportion": proportion,
                    "proportion_percent": f"{proportion * 100:.1f}%"
                }
                
                # 使用LLM生成语义解读
                if use_llm and keywords:
                    semantic_interpretation = self._generate_topic_interpretation(
                        topic_id, keywords, language
                    )
                    interpretation["semantic_interpretation"] = semantic_interpretation
                
                topic_interpretations.append(interpretation)
            
            # 按比例排序
            topic_interpretations.sort(
                key=lambda x: x.get("proportion", 0), 
                reverse=True
            )
            
            return {
                "status": "success",
                "job_id": job_id,
                "topics_count": len(topic_interpretations),
                "topics": topic_interpretations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to interpret topics: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e)
            }
    
    def generate_summary(
        self, 
        job_id: str,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        生成分析摘要
        
        Args:
            job_id: 任务ID
            language: 语言 (zh/en)
            
        Returns:
            包含分析摘要的字典
        """
        try:
            # 加载所有数据
            metrics = self._load_metrics(job_id)
            topics = self._load_topics(job_id)
            analysis_result = self._load_analysis_result(job_id)
            
            # 准备上下文
            context = self._prepare_summary_context(metrics, topics, analysis_result)
            
            # 使用LLM生成摘要
            summary = self._generate_llm_summary(context, language)
            
            return {
                "status": "success",
                "job_id": job_id,
                "summary": summary,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e)
            }
    
    def answer_question(
        self, 
        job_id: str,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        回答用户问题（支持多轮对话）
        
        Args:
            job_id: 任务ID
            question: 用户问题
            session_id: 会话ID（用于多轮对话）
            
        Returns:
            包含回答的字典
        """
        try:
            # 加载分析数据
            metrics = self._load_metrics(job_id)
            topics = self._load_topics(job_id)
            analysis_result = self._load_analysis_result(job_id)
            
            # 准备上下文
            context = self._prepare_qa_context(metrics, topics, analysis_result)
            
            # 获取对话历史
            if session_id:
                history = self.conversation_history.get(session_id, [])
            else:
                session_id = f"{job_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                history = []
            
            # 生成回答
            answer = self._generate_qa_answer(question, context, history)
            
            # 更新对话历史
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            self.conversation_history[session_id] = history[-10:]  # 保留最近10轮
            
            return {
                "status": "success",
                "job_id": job_id,
                "session_id": session_id,
                "question": question,
                "answer": answer
            }
            
        except Exception as e:
            self.logger.error(f"Failed to answer question: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "question": question,
                "error": str(e)
            }
    
    def _evaluate_metric_quality(
        self, 
        metric_key: str, 
        value: float,
        template: Dict
    ) -> str:
        """评估指标质量等级"""
        good_threshold = template.get("good_threshold", 0)
        excellent_threshold = template.get("excellent_threshold", 0)
        lower_is_better = template.get("lower_is_better", False)
        
        if lower_is_better:
            if value <= excellent_threshold:
                return "excellent"
            elif value <= good_threshold:
                return "good"
            elif value <= good_threshold * 2:
                return "fair"
            else:
                return "poor"
        else:
            if value >= excellent_threshold:
                return "excellent"
            elif value >= good_threshold:
                return "good"
            elif value >= good_threshold * 0.5:
                return "fair"
            else:
                return "poor"
    
    def _generate_overall_assessment(
        self, 
        quality_counts: Dict[str, int],
        language: str
    ) -> str:
        """生成总体评估"""
        total = sum(quality_counts.values())
        if total == 0:
            return "无法评估" if language == "zh" else "Unable to assess"
        
        excellent_ratio = quality_counts["excellent"] / total
        good_ratio = (quality_counts["excellent"] + quality_counts["good"]) / total
        
        if excellent_ratio >= 0.5:
            if language == "zh":
                return "🌟 整体表现优秀，主题模型质量很高，结果可信度强。"
            return "🌟 Excellent overall performance. High quality topic model with strong reliability."
        elif good_ratio >= 0.6:
            if language == "zh":
                return "✅ 整体表现良好，主题模型质量较好，结果具有参考价值。"
            return "✅ Good overall performance. Quality topic model with valuable results."
        elif good_ratio >= 0.4:
            if language == "zh":
                return "⚠️ 整体表现一般，建议调整参数或增加数据量以提升质量。"
            return "⚠️ Fair overall performance. Consider adjusting parameters or increasing data."
        else:
            if language == "zh":
                return "❌ 整体表现较差，建议重新调整模型参数或检查数据质量。"
            return "❌ Poor overall performance. Recommend adjusting model parameters or checking data quality."
    
    def _generate_topic_interpretation(
        self, 
        topic_id: int,
        keywords: List,
        language: str
    ) -> str:
        """使用LLM生成主题语义解读"""
        # 处理keywords格式
        if isinstance(keywords, list) and len(keywords) > 0:
            if isinstance(keywords[0], dict):
                keyword_str = ", ".join([k.get("word", str(k)) for k in keywords[:10]])
            else:
                keyword_str = ", ".join([str(k) for k in keywords[:10]])
        else:
            keyword_str = str(keywords)
        
        if language == "zh":
            prompt = f"""请根据以下主题关键词，用一句话概括这个主题讨论的内容领域：

主题{topic_id}的关键词：{keyword_str}

要求：
1. 直接描述主题内容，不要说"这个主题是关于..."
2. 使用业务化语言，避免技术术语
3. 控制在30字以内"""
        else:
            prompt = f"""Based on the following topic keywords, summarize the content area in one sentence:

Topic {topic_id} keywords: {keyword_str}

Requirements:
1. Directly describe the topic content
2. Use business language, avoid technical terms
3. Keep it under 30 words"""
        
        try:
            return self._call_llm(prompt)
        except Exception as e:
            self.logger.warning(f"LLM call failed for topic interpretation: {e}")
            return f"主题{topic_id}相关内容" if language == "zh" else f"Topic {topic_id} related content"
    
    def _prepare_summary_context(
        self, 
        metrics: Dict,
        topics: List,
        analysis_result: Dict
    ) -> str:
        """准备摘要生成的上下文"""
        parts = []
        
        # 添加指标信息
        parts.append("=== 评估指标 ===")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not key.endswith("_per_topic"):
                parts.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        # 添加主题信息
        parts.append("\n=== 主题列表 ===")
        for topic in topics[:10]:  # 最多10个主题
            topic_id = topic.get("id", topic.get("topic_id", 0))
            keywords = topic.get("keywords", topic.get("words", []))
            if isinstance(keywords, list) and len(keywords) > 0:
                if isinstance(keywords[0], dict):
                    kw_str = ", ".join([k.get("word", str(k)) for k in keywords[:5]])
                else:
                    kw_str = ", ".join([str(k) for k in keywords[:5]])
            else:
                kw_str = str(keywords)
            proportion = topic.get("proportion", 0)
            parts.append(f"主题{topic_id} ({proportion*100:.1f}%): {kw_str}")
        
        return "\n".join(parts)
    
    def _prepare_qa_context(
        self, 
        metrics: Dict,
        topics: List,
        analysis_result: Dict
    ) -> str:
        """准备问答的上下文"""
        return self._prepare_summary_context(metrics, topics, analysis_result)
    
    def _generate_llm_summary(self, context: str, language: str) -> str:
        """使用LLM生成摘要"""
        if language == "zh":
            prompt = f"""请基于以下主题模型分析结果，生成一份简洁的分析摘要：

{context}

要求：
1. 概述主要发现（2-3句话）
2. 指出最重要的主题及其含义
3. 给出质量评估结论
4. 提供1-2条建议
5. 总字数控制在200字以内"""
        else:
            prompt = f"""Based on the following topic model analysis results, generate a concise summary:

{context}

Requirements:
1. Overview of main findings (2-3 sentences)
2. Highlight the most important topics and their meanings
3. Quality assessment conclusion
4. 1-2 recommendations
5. Keep total under 200 words"""
        
        try:
            return self._call_llm(prompt)
        except Exception as e:
            self.logger.error(f"LLM summary generation failed: {e}")
            return "摘要生成失败，请检查LLM配置。" if language == "zh" else "Summary generation failed. Please check LLM configuration."
    
    def _generate_qa_answer(
        self, 
        question: str,
        context: str,
        history: List[Dict]
    ) -> str:
        """使用LLM生成问答回答"""
        system_prompt = """你是一位专业的主题模型分析专家。你的任务是帮助用户理解主题模型的分析结果。

分析原则：
- 从内容语义角度解读，而非算法角度
- 使用业务化语言，避免技术术语
- 结合具体数据给出解释
- 如果用户问中文问题，用中文回答；英文问题用英文回答"""
        
        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史对话
        for msg in history[-6:]:  # 最近3轮对话
            messages.append(msg)
        
        # 添加当前问题
        user_message = f"分析数据：\n{context}\n\n用户问题：{question}"
        messages.append({"role": "user", "content": user_message})
        
        try:
            return self._call_llm_with_messages(messages)
        except Exception as e:
            self.logger.error(f"LLM QA generation failed: {e}")
            return f"回答生成失败：{str(e)}"
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM API（单轮）"""
        messages = [
            {"role": "system", "content": "你是一位专业的主题模型分析专家。"},
            {"role": "user", "content": prompt}
        ]
        return self._call_llm_with_messages(messages)
    
    def _call_llm_with_messages(self, messages: List[Dict]) -> str:
        """调用LLM API（多轮）"""
        import requests
        
        is_valid, error = LLMConfigManager.validate_config(self.llm_config)
        if not is_valid:
            return error
        
        try:
            # 使用OpenAI兼容接口
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url
            )
            
            response = client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                temperature=self.llm_config.temperature,
                top_p=self.llm_config.top_p,
                max_tokens=self.llm_config.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            # 回退到requests
            return self._call_llm_with_requests(messages)
        except Exception as e:
            self.logger.error(f"OpenAI client error: {e}")
            return self._call_llm_with_requests(messages)
    
    def _call_llm_with_requests(self, messages: List[Dict]) -> str:
        """使用requests调用LLM API"""
        import requests
        
        url = f"{self.llm_config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.llm_config.model,
            "messages": messages,
            "temperature": self.llm_config.temperature,
            "top_p": self.llm_config.top_p,
            "max_tokens": self.llm_config.max_tokens
        }
        
        resp = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            timeout=self.llm_config.timeout
        )
        
        if resp.status_code != 200:
            return f"LLM API error: HTTP {resp.status_code}"
        
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    
    # ==================== 数据加载方法 ====================
    
    def _load_metrics(self, job_id: str) -> Dict[str, Any]:
        """加载指标数据"""
        # 尝试多个可能的路径
        possible_paths = [
            self.base_dir / "result" / job_id / "metrics.json",
            self.base_dir / "result" / job_id / "evaluation" / "metrics.json",
            self.base_dir / "result" / job_id / "analysis_result.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 如果是analysis_result，提取metrics部分
                    if "metrics" in data:
                        return data["metrics"]
                    return data
        
        return {}
    
    def _load_topics(self, job_id: str) -> List[Dict]:
        """加载主题数据"""
        possible_paths = [
            self.base_dir / "result" / job_id / "topics.json",
            self.base_dir / "result" / job_id / "topic_words.json",
            self.base_dir / "result" / job_id / "analysis_result.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "topics" in data:
                        return data["topics"]
                    if isinstance(data, list):
                        return data
        
        return []
    
    def _load_analysis_result(self, job_id: str) -> Dict[str, Any]:
        """加载完整分析结果"""
        path = self.base_dir / "result" / job_id / "analysis_result.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
