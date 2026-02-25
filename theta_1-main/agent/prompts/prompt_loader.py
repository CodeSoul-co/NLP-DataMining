"""
Prompt Loader
提示词加载器：从 txt 文件加载提示词模板

使用方式：
    from agent.prompts import PromptLoader
    
    loader = PromptLoader()
    system_prompt = loader.get("qa_system", language="zh")
"""

import os
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache


class PromptLoader:
    """
    提示词加载器
    
    从 templates/ 目录加载 txt 格式的提示词文件
    支持多语言和缓存
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化加载器
        
        Args:
            templates_dir: 模板目录路径，默认为当前目录下的 templates/
        """
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            self.templates_dir = Path(__file__).parent / "templates"
        
        self._cache: Dict[str, str] = {}
    
    def get(self, name: str, language: str = "zh", use_cache: bool = True) -> str:
        """
        获取提示词
        
        Args:
            name: 提示词名称（不含语言后缀和扩展名）
            language: 语言代码 (zh/en)
            use_cache: 是否使用缓存
            
        Returns:
            提示词内容
            
        Example:
            loader.get("qa_system", "zh")  # 加载 qa_system_zh.txt
        """
        cache_key = f"{name}_{language}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 构建文件路径
        filename = f"{name}_{language}.txt"
        filepath = self.templates_dir / filename
        
        if not filepath.exists():
            # 尝试不带语言后缀的文件
            filepath = self.templates_dir / f"{name}.txt"
            if not filepath.exists():
                raise FileNotFoundError(f"Prompt template not found: {filename}")
        
        # 读取文件
        content = filepath.read_text(encoding="utf-8").strip()
        
        if use_cache:
            self._cache[cache_key] = content
        
        return content
    
    def get_with_fallback(self, name: str, language: str = "zh", fallback: str = "") -> str:
        """
        获取提示词，如果不存在则返回默认值
        
        Args:
            name: 提示词名称
            language: 语言代码
            fallback: 默认值
            
        Returns:
            提示词内容或默认值
        """
        try:
            return self.get(name, language)
        except FileNotFoundError:
            return fallback
    
    def reload(self, name: str = None, language: str = None):
        """
        重新加载提示词（清除缓存）
        
        Args:
            name: 指定名称，None 表示清除所有
            language: 指定语言，None 表示所有语言
        """
        if name is None:
            self._cache.clear()
        elif language is None:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{name}_")]
            for k in keys_to_remove:
                del self._cache[k]
        else:
            cache_key = f"{name}_{language}"
            if cache_key in self._cache:
                del self._cache[cache_key]
    
    def list_templates(self) -> list:
        """
        列出所有可用的模板
        
        Returns:
            模板名称列表
        """
        if not self.templates_dir.exists():
            return []
        
        templates = []
        for f in self.templates_dir.glob("*.txt"):
            templates.append(f.stem)
        return sorted(templates)
    
    def save(self, name: str, content: str, language: str = "zh"):
        """
        保存提示词到文件
        
        Args:
            name: 提示词名称
            content: 提示词内容
            language: 语言代码
        """
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{name}_{language}.txt"
        filepath = self.templates_dir / filename
        filepath.write_text(content, encoding="utf-8")
        
        # 更新缓存
        cache_key = f"{name}_{language}"
        self._cache[cache_key] = content


# 全局加载器实例
_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """获取全局提示词加载器实例"""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


def load_prompt(name: str, language: str = "zh") -> str:
    """
    便捷函数：加载提示词
    
    Args:
        name: 提示词名称
        language: 语言代码
        
    Returns:
        提示词内容
    """
    return get_prompt_loader().get(name, language)


def reload_prompts():
    """便捷函数：重新加载所有提示词"""
    get_prompt_loader().reload()
