"""
File Utils
文件操作工具函数
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        path: 文件路径
        
    Returns:
        JSON内容字典，文件不存在返回空字典
    """
    path = Path(path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2):
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        path: 文件路径
        indent: 缩进空格数
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        目录Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Union[str, Path], default: str = "") -> str:
    """
    读取文本文件
    
    Args:
        path: 文件路径
        default: 文件不存在时的默认值
        
    Returns:
        文件内容
    """
    path = Path(path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return default


def write_text(content: str, path: Union[str, Path]):
    """
    写入文本文件
    
    Args:
        content: 文件内容
        path: 文件路径
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def list_files(directory: Union[str, Path], pattern: str = "*") -> list:
    """
    列出目录下的文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        
    Returns:
        文件路径列表
    """
    directory = Path(directory)
    if directory.exists():
        return list(directory.glob(pattern))
    return []
