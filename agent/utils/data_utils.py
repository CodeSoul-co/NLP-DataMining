"""
Data Utils
数据处理工具函数
"""

import numpy as np
from typing import Dict, Any, List, Union


def convert_to_serializable(obj: Any) -> Any:
    """
    将对象转换为JSON可序列化格式
    
    Args:
        obj: 任意对象
        
    Returns:
        可序列化的对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    return obj


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    扁平化嵌套字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键前缀
        sep: 分隔符
        
    Returns:
        扁平化的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '_') -> Dict[str, Any]:
    """
    还原扁平化的字典
    
    Args:
        d: 扁平化的字典
        sep: 分隔符
        
    Returns:
        嵌套字典
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def format_number(value: float, precision: int = 4) -> str:
    """
    格式化数字显示
    
    Args:
        value: 数值
        precision: 小数精度
        
    Returns:
        格式化的字符串
    """
    if abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        a: 被除数
        b: 除数
        default: 除数为零时的默认值
        
    Returns:
        除法结果
    """
    if b == 0:
        return default
    return a / b


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分块
    
    Args:
        lst: 原始列表
        chunk_size: 块大小
        
    Returns:
        分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
