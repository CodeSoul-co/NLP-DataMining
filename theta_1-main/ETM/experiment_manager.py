#!/usr/bin/env python
"""
Experiment Manager - 实验管理模块

提供实验列表、查询、选择等功能，支持：
1. API 调用（供前端使用）
2. 命令行交互式选择（供测试使用）

Usage:
    # 列出数据实验
    python experiment_manager.py --list-data --dataset edu_data
    
    # 列出模型实验
    python experiment_manager.py --list-models --dataset edu_data --model lda
    
    # 交互式选择数据实验
    python experiment_manager.py --select-data --dataset edu_data
    
    # 交互式选择模型实验
    python experiment_manager.py --select-model --dataset edu_data --model lda
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

# Default paths
RESULT_DIR = '/root/autodl-tmp/result'


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, result_dir: str = RESULT_DIR):
        self.result_dir = Path(result_dir)
    
    def list_data_experiments(self, dataset: str, model_type: str = 'baseline') -> List[Dict[str, Any]]:
        """
        列出数据预处理实验
        
        Args:
            dataset: 数据集名称
            model_type: 模型类型 ('baseline' 或 'theta')
        
        Returns:
            实验列表，每个实验包含 exp_id, created_at, config 等信息
        """
        if model_type == 'theta':
            data_dir = self.result_dir / 'theta' / dataset / 'data'
        else:
            data_dir = self.result_dir / 'baseline' / dataset / 'data'
        
        if not data_dir.exists():
            return []
        
        experiments = []
        for exp_dir in sorted(data_dir.iterdir()):
            if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                exp_info = self._load_experiment_info(exp_dir)
                experiments.append(exp_info)
        
        # 按创建时间倒序排列（最新的在前）
        experiments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return experiments
    
    def list_model_experiments(self, dataset: str, model: str, model_type: str = 'baseline') -> List[Dict[str, Any]]:
        """
        列出模型训练实验
        
        Args:
            dataset: 数据集名称
            model: 模型名称 (lda, hdp, etc.)
            model_type: 模型类型 ('baseline' 或 'theta')
        
        Returns:
            实验列表
        """
        if model_type == 'theta':
            # THETA 结构: theta/{dataset}/models/{model_size}_{mode}/{exp_id}/
            models_dir = self.result_dir / 'theta' / dataset / 'models'
        else:
            models_dir = self.result_dir / 'baseline' / dataset / 'models' / model
        
        if not models_dir.exists():
            return []
        
        experiments = []
        
        if model_type == 'theta':
            # 遍历所有 model_size_mode 目录
            for size_mode_dir in models_dir.iterdir():
                if size_mode_dir.is_dir():
                    for exp_dir in size_mode_dir.iterdir():
                        if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                            exp_info = self._load_experiment_info(exp_dir)
                            exp_info['model_config'] = size_mode_dir.name
                            experiments.append(exp_info)
        else:
            for exp_dir in sorted(models_dir.iterdir()):
                if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                    exp_info = self._load_experiment_info(exp_dir)
                    experiments.append(exp_info)
        
        # 按创建时间倒序排列
        experiments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return experiments
    
    def _load_experiment_info(self, exp_dir: Path) -> Dict[str, Any]:
        """加载实验信息"""
        exp_info = {
            'exp_id': exp_dir.name,
            'path': str(exp_dir),
            'created_at': None,
            'config': {}
        }
        
        # 尝试从 config.json 加载
        config_path = exp_dir / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                exp_info['config'] = config
                exp_info['created_at'] = config.get('created_at')
            except:
                pass
        
        # 如果没有 created_at，从目录名解析
        if not exp_info['created_at']:
            # exp_20260205_171229_vocab3500 -> 2026-02-05 17:12:29
            parts = exp_dir.name.split('_')
            if len(parts) >= 3:
                try:
                    date_str = parts[1]  # 20260205
                    time_str = parts[2]  # 171229
                    exp_info['created_at'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                except:
                    pass
        
        # 提取标签（exp_id 中时间戳之后的部分）
        parts = exp_dir.name.split('_')
        if len(parts) > 3:
            exp_info['label'] = '_'.join(parts[3:])
        else:
            exp_info['label'] = None
        
        return exp_info
    
    def get_latest_data_experiment(self, dataset: str, model_type: str = 'baseline') -> Optional[str]:
        """获取最新的数据实验 ID"""
        experiments = self.list_data_experiments(dataset, model_type)
        if experiments:
            return experiments[0]['exp_id']
        return None
    
    def get_latest_model_experiment(self, dataset: str, model: str, model_type: str = 'baseline') -> Optional[str]:
        """获取最新的模型实验 ID"""
        experiments = self.list_model_experiments(dataset, model, model_type)
        if experiments:
            return experiments[0]['exp_id']
        return None
    
    def find_data_experiment(self, dataset: str, query: str, model_type: str = 'baseline') -> Optional[str]:
        """
        模糊查找数据实验
        
        Args:
            dataset: 数据集名称
            query: 查询字符串（可以是完整 exp_id、标签、或 'latest'）
        
        Returns:
            匹配的实验路径，或 None
        """
        if query == 'latest':
            exp_id = self.get_latest_data_experiment(dataset, model_type)
            if exp_id:
                if model_type == 'theta':
                    return str(self.result_dir / 'theta' / dataset / 'data' / exp_id)
                return str(self.result_dir / 'baseline' / dataset / 'data' / exp_id)
            return None
        
        experiments = self.list_data_experiments(dataset, model_type)
        
        # 精确匹配 exp_id
        for exp in experiments:
            if exp['exp_id'] == query:
                return exp['path']
        
        # 模糊匹配（包含查询字符串）
        for exp in experiments:
            if query in exp['exp_id']:
                return exp['path']
            if exp.get('label') and query in exp['label']:
                return exp['path']
        
        return None
    
    def find_model_experiment(self, dataset: str, model: str, query: str, model_type: str = 'baseline') -> Optional[str]:
        """
        模糊查找模型实验
        
        Args:
            dataset: 数据集名称
            model: 模型名称
            query: 查询字符串（可以是完整 exp_id、标签、或 'latest'）
        
        Returns:
            匹配的实验路径，或 None
        """
        if query == 'latest':
            exp_id = self.get_latest_model_experiment(dataset, model, model_type)
            if exp_id:
                if model_type == 'theta':
                    return str(self.result_dir / 'theta' / dataset / 'models' / model / exp_id)
                return str(self.result_dir / 'baseline' / dataset / 'models' / model / exp_id)
            return None
        
        experiments = self.list_model_experiments(dataset, model, model_type)
        
        # 精确匹配 exp_id
        for exp in experiments:
            if exp['exp_id'] == query:
                return exp['path']
        
        # 模糊匹配（包含查询字符串）
        for exp in experiments:
            if query in exp['exp_id']:
                return exp['path']
            if exp.get('label') and query in exp['label']:
                return exp['path']
        
        return None


def interactive_select_data(manager: ExperimentManager, dataset: str, model_type: str = 'baseline') -> Optional[str]:
    """交互式选择数据实验"""
    experiments = manager.list_data_experiments(dataset, model_type)
    
    if not experiments:
        print(f"没有找到数据实验: {dataset}")
        return None
    
    print(f"\n{'='*60}")
    print(f"选择数据预处理实验 ({dataset})")
    print(f"{'='*60}")
    
    for i, exp in enumerate(experiments):
        config = exp.get('config', {})
        vocab_size = config.get('vocab_size', '?')
        created = exp.get('created_at', '?')
        label = exp.get('label', '')
        
        # 标记最新的
        latest_mark = " [最新]" if i == 0 else ""
        label_str = f" ({label})" if label else ""
        
        print(f"  [{i+1}] {exp['exp_id']}{latest_mark}")
        print(f"      vocab_size={vocab_size}, created={created}{label_str}")
    
    print(f"\n  [0] 取消")
    print(f"{'='*60}")
    
    while True:
        try:
            choice = input("请选择 (输入数字，回车选择最新): ").strip()
            if choice == '':
                choice = 1  # 默认选择最新
            else:
                choice = int(choice)
            
            if choice == 0:
                return None
            if 1 <= choice <= len(experiments):
                selected = experiments[choice - 1]
                print(f"\n已选择: {selected['exp_id']}")
                return selected['path']
            print("无效选择，请重新输入")
        except ValueError:
            print("请输入数字")
        except KeyboardInterrupt:
            print("\n取消选择")
            return None


def interactive_select_model(manager: ExperimentManager, dataset: str, model: str, model_type: str = 'baseline') -> Optional[str]:
    """交互式选择模型实验"""
    experiments = manager.list_model_experiments(dataset, model, model_type)
    
    if not experiments:
        print(f"没有找到模型实验: {dataset}/{model}")
        return None
    
    print(f"\n{'='*60}")
    print(f"选择模型训练实验 ({dataset}/{model})")
    print(f"{'='*60}")
    
    for i, exp in enumerate(experiments):
        config = exp.get('config', {})
        num_topics = config.get('num_topics', '?')
        data_exp = config.get('data_exp', '?')
        created = exp.get('created_at', '?')
        label = exp.get('label', '')
        
        latest_mark = " [最新]" if i == 0 else ""
        label_str = f" ({label})" if label else ""
        
        print(f"  [{i+1}] {exp['exp_id']}{latest_mark}")
        print(f"      num_topics={num_topics}, data_exp={data_exp[:20]}...{label_str}")
    
    print(f"\n  [0] 取消")
    print(f"{'='*60}")
    
    while True:
        try:
            choice = input("请选择 (输入数字，回车选择最新): ").strip()
            if choice == '':
                choice = 1
            else:
                choice = int(choice)
            
            if choice == 0:
                return None
            if 1 <= choice <= len(experiments):
                selected = experiments[choice - 1]
                print(f"\n已选择: {selected['exp_id']}")
                return selected['path']
            print("无效选择，请重新输入")
        except ValueError:
            print("请输入数字")
        except KeyboardInterrupt:
            print("\n取消选择")
            return None


def print_experiments_json(experiments: List[Dict], pretty: bool = True):
    """以 JSON 格式打印实验列表（供 API 使用）"""
    if pretty:
        print(json.dumps(experiments, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(experiments, ensure_ascii=False))


def print_experiments_table(experiments: List[Dict], exp_type: str = 'data'):
    """以表格格式打印实验列表"""
    if not experiments:
        print("没有找到实验")
        return
    
    print(f"\n{'='*80}")
    if exp_type == 'data':
        print(f"{'序号':<4} {'实验ID':<35} {'vocab_size':<12} {'创建时间':<20}")
    else:
        print(f"{'序号':<4} {'实验ID':<35} {'num_topics':<12} {'创建时间':<20}")
    print(f"{'='*80}")
    
    for i, exp in enumerate(experiments):
        config = exp.get('config', {})
        if exp_type == 'data':
            param = config.get('vocab_size', '?')
        else:
            param = config.get('num_topics', '?')
        created = exp.get('created_at', '?')[:19] if exp.get('created_at') else '?'
        
        print(f"{i+1:<4} {exp['exp_id']:<35} {str(param):<12} {created:<20}")
    
    print(f"{'='*80}")
    print(f"共 {len(experiments)} 个实验")


def main():
    parser = argparse.ArgumentParser(description='实验管理器')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--model-type', type=str, default='baseline', choices=['baseline', 'theta'])
    
    # 操作类型
    parser.add_argument('--list-data', action='store_true', help='列出数据实验')
    parser.add_argument('--list-models', action='store_true', help='列出模型实验')
    parser.add_argument('--select-data', action='store_true', help='交互式选择数据实验')
    parser.add_argument('--select-model', action='store_true', help='交互式选择模型实验')
    parser.add_argument('--find-data', type=str, help='查找数据实验（支持模糊匹配）')
    parser.add_argument('--find-model', type=str, help='查找模型实验（支持模糊匹配）')
    
    # 输出格式
    parser.add_argument('--json', action='store_true', help='以 JSON 格式输出')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.list_data:
        if not args.dataset:
            print("错误: --list-data 需要 --dataset 参数")
            return
        experiments = manager.list_data_experiments(args.dataset, args.model_type)
        if args.json:
            print_experiments_json(experiments)
        else:
            print_experiments_table(experiments, 'data')
    
    elif args.list_models:
        if not args.dataset or not args.model:
            print("错误: --list-models 需要 --dataset 和 --model 参数")
            return
        experiments = manager.list_model_experiments(args.dataset, args.model, args.model_type)
        if args.json:
            print_experiments_json(experiments)
        else:
            print_experiments_table(experiments, 'model')
    
    elif args.select_data:
        if not args.dataset:
            print("错误: --select-data 需要 --dataset 参数")
            return
        result = interactive_select_data(manager, args.dataset, args.model_type)
        if result:
            print(f"\n输出: {result}")
    
    elif args.select_model:
        if not args.dataset or not args.model:
            print("错误: --select-model 需要 --dataset 和 --model 参数")
            return
        result = interactive_select_model(manager, args.dataset, args.model, args.model_type)
        if result:
            print(f"\n输出: {result}")
    
    elif args.find_data:
        if not args.dataset:
            print("错误: --find-data 需要 --dataset 参数")
            return
        result = manager.find_data_experiment(args.dataset, args.find_data, args.model_type)
        if result:
            print(result)
        else:
            print(f"未找到匹配的实验: {args.find_data}")
    
    elif args.find_model:
        if not args.dataset or not args.model:
            print("错误: --find-model 需要 --dataset 和 --model 参数")
            return
        result = manager.find_model_experiment(args.dataset, args.model, args.find_model, args.model_type)
        if result:
            print(result)
        else:
            print(f"未找到匹配的实验: {args.find_model}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
