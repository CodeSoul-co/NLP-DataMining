"""
下载 Qwen3-Embedding-0.6B 模型到本地
模型是免费开源的，无需付费

下载来源：ModelScope (阿里云魔搭社区)
模型大小：约 1.2GB
"""

import os
import sys

# 下载目录
DOWNLOAD_DIR = r"d:\chalotte\OneDrive\桌面\qwen3_embedding_0.6B"

def download_from_modelscope():
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        
        print("=" * 60)
        print("开始从 ModelScope 下载 Qwen3-Embedding-0.6B")
        print("模型是免费开源的，无需付费")
        print("=" * 60)
        print(f"\n下载目录: {DOWNLOAD_DIR}")
        print("预计大小: ~1.2GB")
        print("请耐心等待...\n")
        
        # 下载模型
        model_dir = snapshot_download(
            model_id='Qwen/Qwen3-Embedding-0.6B',
            cache_dir=DOWNLOAD_DIR,
            revision='master'
        )
        
        print("\n" + "=" * 60)
        print("下载完成!")
        print(f"模型路径: {model_dir}")
        print("=" * 60)
        print("\n下一步: 将此文件夹上传到 OSS 的 embedding_models/ 目录")
        
        return model_dir
        
    except ImportError:
        print("错误: 未安装 modelscope")
        print("请运行: pip install modelscope")
        return None
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def download_from_huggingface():
    """从 HuggingFace 下载模型（备用方案）"""
    try:
        from huggingface_hub import snapshot_download
        
        print("=" * 60)
        print("开始从 HuggingFace 下载 Qwen3-Embedding-0.6B")
        print("=" * 60)
        
        model_dir = snapshot_download(
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            local_dir=DOWNLOAD_DIR,
            local_dir_use_symlinks=False
        )
        
        print(f"\n下载完成! 模型路径: {model_dir}")
        return model_dir
        
    except ImportError:
        print("错误: 未安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"下载失败: {e}")
        return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Qwen3-Embedding-0.6B 模型下载工具")
    print("=" * 60)
    print("\n模型信息:")
    print("  - 名称: Qwen3-Embedding-0.6B")
    print("  - 来源: 阿里云通义千问")
    print("  - 许可: Apache 2.0 (免费开源)")
    print("  - 大小: ~1.2GB")
    print()
    
    # 优先使用 ModelScope
    result = download_from_modelscope()
    
    if result is None:
        print("\nModelScope 下载失败，尝试 HuggingFace...")
        result = download_from_huggingface()
    
    if result:
        print("\n" + "=" * 60)
        print("成功! 模型已下载到:")
        print(f"  {DOWNLOAD_DIR}")
        print("\n请将此文件夹上传到 OSS:")
        print("  oss://theta-prod-20260123/embedding_models/qwen3_embedding_0.6B/")
        print("=" * 60)
