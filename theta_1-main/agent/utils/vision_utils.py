"""
Vision Utils
Utility functions for Qwen3 Vision API calls.
Supports image analysis using Qwen3-VL models.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


def get_vision_client():
    """
    Get OpenAI-compatible client for Qwen3 Vision API.
    
    Returns:
        OpenAI client instance configured for Qwen3 Vision
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    api_key = os.environ.get("QWEN_VISION_API_KEY", "")
    base_url = os.environ.get("QWEN_VISION_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    
    if not api_key:
        raise ValueError("QWEN_VISION_API_KEY not set in environment")
    
    return OpenAI(api_key=api_key, base_url=base_url)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """
    Get MIME type based on file extension.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string
    """
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }
    return mime_types.get(ext, "image/jpeg")


def analyze_image(
    image_source: str,
    question: str,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    max_tokens: int = 1000
) -> str:
    """
    Analyze an image using Qwen3 Vision API.
    
    Args:
        image_source: URL or local file path to the image
        question: Question about the image
        model: Model name (default from env)
        api_key: API key (default from env)
        base_url: API base URL (default from env)
        max_tokens: Maximum tokens in response
        
    Returns:
        Analysis result text
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    # Get configuration from environment
    api_key = api_key or os.environ.get("QWEN_VISION_API_KEY", "")
    base_url = base_url or os.environ.get("QWEN_VISION_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    model = model or os.environ.get("QWEN_VISION_MODEL", "qwen3-vl-plus")
    
    if not api_key:
        raise ValueError("QWEN_VISION_API_KEY not set")
    
    # Prepare image content
    if image_source.startswith(("http://", "https://")):
        # URL-based image
        image_content = {
            "type": "image_url",
            "image_url": {"url": image_source}
        }
    else:
        # Local file - encode to base64
        base64_image = encode_image_to_base64(image_source)
        mime_type = get_image_mime_type(image_source)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
        }
    
    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Call API
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Vision API error: {e}")
        raise Exception(f"Vision API error: {e}")


def analyze_multiple_images(
    images: List[str],
    question: str,
    model: str = None,
    max_tokens: int = 2000
) -> str:
    """
    Analyze multiple images together.
    
    Args:
        images: List of image URLs or local file paths
        question: Question about the images
        model: Model name (default from env)
        max_tokens: Maximum tokens in response
        
    Returns:
        Analysis result text
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    api_key = os.environ.get("QWEN_VISION_API_KEY", "")
    base_url = os.environ.get("QWEN_VISION_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    model = model or os.environ.get("QWEN_VISION_MODEL", "qwen3-vl-plus")
    
    if not api_key:
        raise ValueError("QWEN_VISION_API_KEY not set")
    
    # Build content list with all images
    content = []
    for image_source in images:
        if image_source.startswith(("http://", "https://")):
            content.append({
                "type": "image_url",
                "image_url": {"url": image_source}
            })
        else:
            base64_image = encode_image_to_base64(image_source)
            mime_type = get_image_mime_type(image_source)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
            })
    
    # Add question
    content.append({"type": "text", "text": question})
    
    messages = [{"role": "user", "content": content}]
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Vision API error: {e}")
        raise Exception(f"Vision API error: {e}")


def analyze_chart(
    chart_path: str,
    analysis_type: str = "general",
    language: str = "zh"
) -> Dict[str, Any]:
    """
    Analyze a chart image (word cloud, distribution, etc.).
    
    Args:
        chart_path: Path to the chart image
        analysis_type: Type of analysis (general, wordcloud, distribution, heatmap)
        language: Response language (zh/en)
        
    Returns:
        Dictionary with analysis results
    """
    prompts = {
        "general": {
            "zh": "请详细分析这张图表，描述其主要内容、关键发现和可能的业务洞察。",
            "en": "Please analyze this chart in detail, describing its main content, key findings, and potential business insights."
        },
        "wordcloud": {
            "zh": "这是一个词云图。请分析其中的关键词，识别主要主题，并解释这些词语之间的关联。",
            "en": "This is a word cloud. Please analyze the keywords, identify main themes, and explain the relationships between these words."
        },
        "distribution": {
            "zh": "这是一个分布图。请分析数据分布特征，识别主要模式和异常点。",
            "en": "This is a distribution chart. Please analyze the data distribution characteristics, identify main patterns and outliers."
        },
        "heatmap": {
            "zh": "这是一个热力图。请分析其中的相关性模式，识别强相关和弱相关的区域。",
            "en": "This is a heatmap. Please analyze the correlation patterns, identify areas of strong and weak correlation."
        }
    }
    
    prompt = prompts.get(analysis_type, prompts["general"]).get(language, prompts["general"]["en"])
    
    try:
        result = analyze_image(chart_path, prompt)
        return {
            "success": True,
            "analysis_type": analysis_type,
            "chart_path": chart_path,
            "analysis": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chart_path": chart_path
        }
