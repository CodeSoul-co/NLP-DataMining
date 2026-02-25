"""
THETA DLC 本地客户端
直接调用阿里云 PAI-DLC API 提交训练任务

使用方法:
    python dlc_client.py submit --dataset data --num_topics 20 --epochs 50 --mode zero_shot
    python dlc_client.py status --job_id <job_id>
    python dlc_client.py list
    python dlc_client.py stop --job_id <job_id>
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 阿里云 SDK
try:
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_pai_dlc20201203 import models as dlc_models
    from alibabacloud_tea_openapi import models as open_api_models
    DLC_AVAILABLE = True
except ImportError:
    DLC_AVAILABLE = False
    print("[WARNING] 阿里云 DLC SDK 未安装，请运行: pip install alibabacloud_pai_dlc20201203")


# ============ 配置 ============
# 请填写你的阿里云凭证
ACCESS_KEY_ID = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
ACCESS_KEY_SECRET = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
REGION_ID = "cn-shanghai"
WORKSPACE_ID = "464377"

# OSS 配置
OSS_BUCKET = "theta-prod-20260123"
OSS_DATASET_ID = "d-cvx2t6q7t8w3bnrvgl"  # PAI 数据集 ID

# DLC 配置
DEFAULT_IMAGE = "registry.cn-shanghai.aliyuncs.com/pai-dlc/pytorch-training:2.1-gpu-py310-cu121-ubuntu22.04"
DEFAULT_INSTANCE_TYPE = "ecs.gn7i-c8g1.2xlarge"  # A10 GPU 24GB
# ==============================


def create_dlc_client():
    """创建 DLC 客户端"""
    if not DLC_AVAILABLE:
        raise RuntimeError("阿里云 DLC SDK 未安装")
    
    if not ACCESS_KEY_ID or not ACCESS_KEY_SECRET:
        raise RuntimeError("请设置环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    
    config = open_api_models.Config(
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        region_id=REGION_ID,
        endpoint=f"pai-dlc.{REGION_ID}.aliyuncs.com"
    )
    return Client(config)


def submit_job(dataset: str, num_topics: int, epochs: int, mode: str, model_size: str = "0.6B"):
    """提交 DLC 训练任务"""
    client = create_dlc_client()
    
    # 生成任务名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"THETA-{dataset}-{mode}_{timestamp}"
    
    # 构建训练命令 - 适配直接 OSS 挂载方式 (/mnt/)
    oss_base = f"oss://{OSS_BUCKET}"
    command = f"""
# 设置环境变量 - 使用 /mnt/ 挂载路径
export THETA_BASE="/mnt"
export THETA_MODEL_SIZE="{model_size}"
export THETA_DATA_DIR="/mnt/data"
export THETA_RESULT_DIR="/mnt/result"
export THETA_MODEL_DIR="/mnt/embedding_models"

# 安装依赖
pip install transformers torch numpy scipy scikit-learn tqdm jieba pandas

# 检查目录结构
echo "=== Directory structure ==="
ls -la /mnt/
ls -la /mnt/code/ || true
ls -la /mnt/data/ || true
ls -la /mnt/data/{dataset}/ || true
ls -la /mnt/data/{dataset}/data/ || true

# 检查数据文件位置
echo "=== Looking for data files ==="
find /mnt/data -name "*.csv" | head -20

# 合并各省份数据文件（如果需要）
echo "=== Merging data files ==="
cd /mnt/data/{dataset}
if [ -d "data" ]; then
    echo "Found data subdirectory, merging CSV files..."
    # 获取第一个 CSV 的表头
    first_csv=$(find data -name "*.csv" | head -1)
    if [ -n "$first_csv" ]; then
        head -1 "$first_csv" > {dataset}_text_only.csv
        # 合并所有 CSV（跳过表头）
        for f in data/*.csv data/*/*.csv; do
            if [ -f "$f" ]; then
                tail -n +2 "$f" >> {dataset}_text_only.csv
            fi
        done
        echo "Merged data saved to {dataset}_text_only.csv"
        wc -l {dataset}_text_only.csv
    fi
fi

# 运行训练 - 使用 2.12 的 scripts 目录结构
cd /mnt/code/ETM

# 方式1: 使用 scripts 脚本运行 (推荐)
# bash /mnt/code/scripts/04_train_theta.sh --dataset {dataset} --model_size {model_size} --mode {mode} --num_topics {num_topics} --epochs {epochs}

# 方式2: 直接运行 run_pipeline.py
python run_pipeline.py --dataset {dataset} --models theta --model_size {model_size} --mode {mode} --num_topics {num_topics} --epochs {epochs} --gpu 0
"""
    
    print(f"[INFO] 提交任务: {job_name}")
    print(f"[INFO] 镜像: {DEFAULT_IMAGE}")
    print(f"[INFO] 实例类型: {DEFAULT_INSTANCE_TYPE}")
    print(f"[INFO] 数据集 ID: {OSS_DATASET_ID}")
    
    # 创建任务请求
    create_job_request = dlc_models.CreateJobRequest(
        workspace_id=WORKSPACE_ID,
        display_name=job_name,
        job_type="PyTorchJob",
        job_specs=[
            dlc_models.JobSpec(
                type="Worker",
                image=DEFAULT_IMAGE,
                pod_count=1,
                ecs_spec=DEFAULT_INSTANCE_TYPE
            )
        ],
        user_command=command,
        data_sources=[
            dlc_models.DataSourceItem(
                data_source_type="Dataset",
                data_source_id=OSS_DATASET_ID,
                mount_path="/workspace"
            )
        ],
        envs={
            "THETA_DATASET": dataset,
            "THETA_NUM_TOPICS": str(num_topics),
            "THETA_EPOCHS": str(epochs),
            "THETA_MODE": mode,
            "THETA_MODEL_SIZE": model_size
        }
    )
    
    # 提交任务
    response = client.create_job(create_job_request)
    job_id = response.body.job_id
    
    print(f"[SUCCESS] 任务提交成功!")
    print(f"[INFO] Job ID: {job_id}")
    print(f"[INFO] 任务名称: {job_name}")
    print(f"[INFO] 查看状态: python dlc_client.py status --job_id {job_id}")
    
    return job_id


def get_job_status(job_id: str):
    """获取任务状态"""
    client = create_dlc_client()
    
    request = dlc_models.GetJobRequest()
    response = client.get_job(job_id, request)
    
    job = response.body
    print(f"[INFO] Job ID: {job_id}")
    print(f"[INFO] 任务名称: {job.display_name}")
    print(f"[INFO] 状态: {job.status}")
    print(f"[INFO] 创建时间: {job.gmt_create_time}")
    
    if hasattr(job, 'reason_message') and job.reason_message:
        print(f"[INFO] 消息: {job.reason_message}")
    
    return job.status


def list_jobs(limit: int = 10):
    """列出最近的任务"""
    client = create_dlc_client()
    
    request = dlc_models.ListJobsRequest(
        workspace_id=WORKSPACE_ID,
        page_size=limit,
        page_number=1
    )
    response = client.list_jobs(request)
    
    print(f"[INFO] 最近 {limit} 个任务:")
    print("-" * 80)
    
    for job in response.body.jobs:
        print(f"  ID: {job.job_id}")
        print(f"  名称: {job.display_name}")
        print(f"  状态: {job.status}")
        print(f"  创建时间: {job.gmt_create_time}")
        print("-" * 80)


def stop_job(job_id: str):
    """停止任务"""
    client = create_dlc_client()
    
    request = dlc_models.StopJobRequest()
    client.stop_job(job_id, request)
    
    print(f"[SUCCESS] 任务 {job_id} 已停止")


def main():
    parser = argparse.ArgumentParser(description="THETA DLC 本地客户端")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # submit 命令
    submit_parser = subparsers.add_parser("submit", help="提交训练任务")
    submit_parser.add_argument("--dataset", type=str, default="data", help="数据集名称")
    submit_parser.add_argument("--num_topics", type=int, default=20, help="主题数量")
    submit_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    submit_parser.add_argument("--mode", type=str, default="zero_shot", help="训练模式")
    submit_parser.add_argument("--model_size", type=str, default="0.6B", help="模型大小")
    
    # status 命令
    status_parser = subparsers.add_parser("status", help="查看任务状态")
    status_parser.add_argument("--job_id", type=str, required=True, help="任务 ID")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出最近任务")
    list_parser.add_argument("--limit", type=int, default=10, help="显示数量")
    
    # stop 命令
    stop_parser = subparsers.add_parser("stop", help="停止任务")
    stop_parser.add_argument("--job_id", type=str, required=True, help="任务 ID")
    
    args = parser.parse_args()
    
    if args.command == "submit":
        submit_job(args.dataset, args.num_topics, args.epochs, args.mode, args.model_size)
    elif args.command == "status":
        get_job_status(args.job_id)
    elif args.command == "list":
        list_jobs(args.limit)
    elif args.command == "stop":
        stop_job(args.job_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
