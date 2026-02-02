#!/usr/bin/env python3
"""
统一入口脚本：联合THETA-main和topic_agent实现完整流程

支持输入：
- Word文档 (.docx)
- 文本文件 (.txt)
- CSV文件 (.csv)

完整流程：
1. 数据清洗（使用THETA-main的DataClean模块）
2. 格式统一转换为CSV
3. BOW生成
4. Embedding生成
5. ETM训练
6. 可视化输出
7. Word报告生成
8. 交互式问答
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
TOPIC_AGENT_ROOT = Path(__file__).parent
THETA_ROOT = TOPIC_AGENT_ROOT.parent  # topic_agent现在在THETA-main目录下

sys.path.insert(0, str(TOPIC_AGENT_ROOT))
sys.path.insert(0, str(THETA_ROOT))

from docx import Document
import pandas as pd


def extract_text_from_docx(docx_path: str) -> list:
    """从Word文档中提取文本段落"""
    doc = Document(docx_path)
    
    paragraphs = []
    current_section = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # 检测是否是新章节标题
            if len(text) < 50 and (
                text.startswith('主题') or 
                text.startswith('LDA') or 
                text.startswith('ETM') or 
                text.startswith('BERTopic') or
                text.startswith('Top2Vec') or 
                text.endswith('模型') or
                text.startswith('传统') or 
                text.startswith('神经') or
                text.startswith('应用') or 
                '.' in text[:5] or
                text[0].isdigit()
            ):
                if current_section:
                    paragraphs.append(' '.join(current_section))
                current_section = [text]
            else:
                current_section.append(text)
    
    if current_section:
        paragraphs.append(' '.join(current_section))
    
    # 过滤太短的段落
    paragraphs = [p for p in paragraphs if len(p) > 50]
    
    return paragraphs


def convert_to_csv(input_path: str, output_path: str) -> int:
    """将输入文件转换为CSV格式"""
    input_path = Path(input_path)
    
    if input_path.suffix.lower() == '.docx':
        paragraphs = extract_text_from_docx(str(input_path))
        df = pd.DataFrame({'text': paragraphs})
    elif input_path.suffix.lower() == '.txt':
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({'text': lines})
    elif input_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path)
        if 'text' not in df.columns:
            # 尝试找到文本列
            text_cols = [c for c in df.columns if 'text' in c.lower() or 'content' in c.lower()]
            if text_cols:
                df = df.rename(columns={text_cols[0]: 'text'})
            else:
                df = df.rename(columns={df.columns[0]: 'text'})
    else:
        raise ValueError(f"不支持的文件格式: {input_path.suffix}")
    
    # 保存CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    return len(df)


def run_full_pipeline(input_file: str, job_id: str = None):
    """运行完整的分析流程"""
    
    # 生成job_id
    if not job_id:
        job_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"=" * 60)
    print(f"🚀 开始完整分析流程")
    print(f"   输入文件: {input_file}")
    print(f"   任务ID: {job_id}")
    print(f"=" * 60)
    
    # Step 1: 数据准备
    print("\n📁 Step 1: 数据准备...")
    data_dir = TOPIC_AGENT_ROOT / "data" / job_id
    data_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = data_dir / "data.csv"
    doc_count = convert_to_csv(input_file, str(csv_path))
    print(f"   ✅ 转换完成: {doc_count} 个文档段落")
    
    # Step 2-7: 运行topic_agent完整流程
    print("\n🔄 Step 2-7: 运行主题分析流程...")
    from app.agent_integration import AgentIntegration
    
    integration = AgentIntegration(base_dir=str(TOPIC_AGENT_ROOT))
    result = integration.run_full_analysis(job_id)
    
    if result.get('status') == 'success':
        print(f"   ✅ 分析完成!")
        print(f"\n📊 生成的文件:")
        
        result_dir = TOPIC_AGENT_ROOT / "result" / job_id
        if result_dir.exists():
            for f in result_dir.iterdir():
                print(f"   - {f.name}")
        
        print(f"\n🎯 现在可以开始交互式提问了!")
        print(f"   使用命令: python interactive_qa.py {job_id}")
        
        return job_id, integration
    else:
        print(f"   ❌ 分析失败: {result.get('error')}")
        return None, None


def interactive_qa(job_id: str, integration=None):
    """交互式问答"""
    if integration is None:
        from app.agent_integration import AgentIntegration
        integration = AgentIntegration(base_dir=str(TOPIC_AGENT_ROOT))
    
    print(f"\n" + "=" * 60)
    print(f"💬 交互式问答模式 (任务: {job_id})")
    print(f"   输入问题进行提问，输入 'quit' 或 'exit' 退出")
    print(f"=" * 60)
    
    while True:
        try:
            question = input("\n🙋 你的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break
            
            print("\n🤔 正在分析...")
            result = integration.handle_query(job_id, question)
            
            if result.get('status') == 'success':
                print(f"\n📝 回答:\n{result.get('answer')}")
            else:
                print(f"\n❌ 错误: {result.get('error')}")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='联合THETA-main和topic_agent的完整分析流程')
    parser.add_argument('input_file', help='输入文件路径 (.docx, .txt, .csv)')
    parser.add_argument('--job_id', '-j', help='任务ID (可选，默认自动生成)')
    parser.add_argument('--interactive', '-i', action='store_true', help='分析完成后进入交互式问答')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input_file).exists():
        print(f"❌ 文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 运行完整流程
    job_id, integration = run_full_pipeline(args.input_file, args.job_id)
    
    # 进入交互式问答
    if job_id and args.interactive:
        interactive_qa(job_id, integration)


if __name__ == "__main__":
    main()
