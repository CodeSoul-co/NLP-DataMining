#!/usr/bin/env python
"""
THETA 任务调度器 - 管理多用户并发任务
支持任务队列、GPU资源分配和状态跟踪
"""

import os
import sys
import json
import time
import uuid
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)

# 全局配置 - 使用环境变量支持 DLC 环境
_DEFAULT_BASE = "/root/autodl-tmp" if os.path.exists("/root/autodl-tmp") else str(Path.home())
CONFIG = {
    "base_dir": os.environ.get("THETA_BASE", _DEFAULT_BASE),
    "gpu_count": int(os.environ.get("THETA_GPU_COUNT", "1")),  # 可用GPU数量
    "queue_file": "job_queue.json",  # 任务队列文件
    "status_dir": "job_status",  # 任务状态目录
    "avg_task_time": 600,  # 平均任务时间（秒）
    "check_interval": 5,  # 检查间隔（秒）
    "pipeline_script": "run_full_pipeline.sh"  # 处理流程脚本
}


class JobQueue:
    """任务队列管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_dir = Path(config["base_dir"])
        self.queue_file = self.base_dir / config["queue_file"]
        self.status_dir = self.base_dir / config["status_dir"]
        self.status_dir.mkdir(exist_ok=True)
        
        # 初始化队列
        if not self.queue_file.exists():
            self._save_queue([])
    
    def _load_queue(self) -> List[Dict[str, Any]]:
        """加载任务队列"""
        if not self.queue_file.exists():
            return []
        
        try:
            with open(self.queue_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载队列失败: {e}")
            return []
    
    def _save_queue(self, queue: List[Dict[str, Any]]) -> None:
        """保存任务队列"""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            logging.error(f"保存队列失败: {e}")
    
    def add_job(self, job_data: Dict[str, Any]) -> str:
        """添加任务到队列"""
        queue = self._load_queue()
        
        # 生成任务ID（如果没有提供）
        if 'job_id' not in job_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = uuid.uuid4().hex[:4]
            job_data['job_id'] = f"job_{timestamp}_{random_suffix}"
        
        # 添加任务元数据
        job_data['status'] = 'queued'
        job_data['submitted_at'] = datetime.now().isoformat()
        job_data['started_at'] = None
        job_data['completed_at'] = None
        job_data['gpu_id'] = None
        
        # 添加到队列
        queue.append(job_data)
        self._save_queue(queue)
        
        # 创建任务状态文件
        self._update_job_status(job_data['job_id'], job_data)
        
        logging.info(f"任务已添加到队列: {job_data['job_id']}")
        return job_data['job_id']
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """获取队列中下一个待处理任务"""
        queue = self._load_queue()
        
        # 查找第一个排队中的任务
        for job in queue:
            if job['status'] == 'queued':
                return job
        
        return None
    
    def start_job(self, job_id: str, gpu_id: int) -> bool:
        """标记任务为运行中"""
        queue = self._load_queue()
        
        for i, job in enumerate(queue):
            if job['job_id'] == job_id:
                queue[i]['status'] = 'running'
                queue[i]['started_at'] = datetime.now().isoformat()
                queue[i]['gpu_id'] = gpu_id
                self._save_queue(queue)
                
                # 更新任务状态文件
                self._update_job_status(job_id, queue[i])
                return True
        
        return False
    
    def complete_job(self, job_id: str, success: bool = True) -> bool:
        """标记任务为已完成"""
        queue = self._load_queue()
        
        for i, job in enumerate(queue):
            if job['job_id'] == job_id:
                queue[i]['status'] = 'completed' if success else 'failed'
                queue[i]['completed_at'] = datetime.now().isoformat()
                self._save_queue(queue)
                
                # 更新任务状态文件
                self._update_job_status(job_id, queue[i])
                return True
        
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        status_file = self.status_dir / f"{job_id}.json"
        
        if not status_file.exists():
            return None
        
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"读取任务状态失败 {job_id}: {e}")
            return None
    
    def _update_job_status(self, job_id: str, status_data: Dict[str, Any]) -> None:
        """更新任务状态文件"""
        status_file = self.status_dir / f"{job_id}.json"
        
        try:
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logging.error(f"更新任务状态失败 {job_id}: {e}")
    
    def get_queue_position(self, job_id: str) -> int:
        """获取任务在队列中的位置"""
        queue = self._load_queue()
        position = 0
        
        for job in queue:
            if job['status'] == 'queued':
                position += 1
                if job['job_id'] == job_id:
                    return position
        
        return 0  # 不在队列中或已在运行
    
    def estimate_wait_time(self, position: int) -> int:
        """估算等待时间（秒）"""
        if position <= 0:
            return 0
        
        return position * self.config["avg_task_time"]
    
    def clean_old_jobs(self, days: int = 30) -> int:
        """清理指定天数前的已完成任务"""
        queue = self._load_queue()
        now = datetime.now()
        count = 0
        
        new_queue = []
        for job in queue:
            if job['status'] in ['completed', 'failed']:
                completed_at = datetime.fromisoformat(job['completed_at'])
                days_old = (now - completed_at).days
                
                if days_old >= days:
                    # 删除状态文件
                    status_file = self.status_dir / f"{job['job_id']}.json"
                    if status_file.exists():
                        status_file.unlink()
                    count += 1
                    continue
            
            new_queue.append(job)
        
        if count > 0:
            self._save_queue(new_queue)
            logging.info(f"已清理 {count} 个旧任务")
        
        return count


class GPUManager:
    """GPU资源管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_count = config["gpu_count"]
        self.running_jobs = {}  # gpu_id -> job_id
    
    def get_available_gpu(self) -> Optional[int]:
        """获取可用的GPU ID"""
        for gpu_id in range(self.gpu_count):
            if gpu_id not in self.running_jobs:
                return gpu_id
        
        return None
    
    def allocate_gpu(self, job_id: str, gpu_id: int) -> None:
        """分配GPU给任务"""
        self.running_jobs[gpu_id] = job_id
        logging.info(f"已分配 GPU {gpu_id} 给任务 {job_id}")
    
    def release_gpu(self, gpu_id: int) -> None:
        """释放GPU资源"""
        if gpu_id in self.running_jobs:
            job_id = self.running_jobs[gpu_id]
            del self.running_jobs[gpu_id]
            logging.info(f"已释放 GPU {gpu_id} (任务 {job_id})")


class Scheduler:
    """任务调度器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_dir = Path(config["base_dir"])
        self.job_queue = JobQueue(config)
        self.gpu_manager = GPUManager(config)
        self.running_processes = {}  # job_id -> process
    
    def run_job(self, job: Dict[str, Any], gpu_id: int) -> None:
        """运行任务"""
        job_id = job['job_id']
        dataset = job.get('dataset', 'default')
        
        # 准备命令行参数
        cmd = [
            '/bin/bash',
            str(self.base_dir / 'scripts' / self.config["pipeline_script"]),
            job_id,
            dataset
        ]
        
        # 添加可选参数
        if 'text_col' in job:
            cmd.append(job['text_col'])
        else:
            cmd.append('')
        
        if 'time_col' in job:
            cmd.append(job['time_col'])
        else:
            cmd.append('')
        
        if 'num_topics' in job:
            cmd.append(str(job['num_topics']))
        else:
            cmd.append('0')
        
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 启动进程
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            self.running_processes[job_id] = process
            logging.info(f"已启动任务 {job_id} 在 GPU {gpu_id}")
            
            # 标记任务为运行中
            self.job_queue.start_job(job_id, gpu_id)
            self.gpu_manager.allocate_gpu(job_id, gpu_id)
            
            # 异步监控进程
            self._monitor_process(job_id, gpu_id)
            
        except Exception as e:
            logging.error(f"启动任务失败 {job_id}: {e}")
            self.job_queue.complete_job(job_id, success=False)
    
    def _monitor_process(self, job_id: str, gpu_id: int) -> None:
        """监控进程（非阻塞）"""
        def _monitor():
            process = self.running_processes[job_id]
            
            # 收集输出
            output = []
            for line in process.stdout:
                output.append(line)
                logging.debug(f"[{job_id}] {line.strip()}")
            
            # 等待进程结束
            return_code = process.wait()
            
            # 更新任务状态
            success = (return_code == 0)
            self.job_queue.complete_job(job_id, success=success)
            
            # 释放GPU
            self.gpu_manager.release_gpu(gpu_id)
            
            # 清理
            if job_id in self.running_processes:
                del self.running_processes[job_id]
            
            if success:
                logging.info(f"任务完成 {job_id}")
            else:
                logging.error(f"任务失败 {job_id}, 返回码: {return_code}")
        
        # 启动监控线程
        import threading
        thread = threading.Thread(target=_monitor)
        thread.daemon = True
        thread.start()
    
    def schedule_loop(self) -> None:
        """主调度循环"""
        logging.info("调度器已启动")
        
        while True:
            try:
                # 检查可用GPU
                gpu_id = self.gpu_manager.get_available_gpu()
                
                if gpu_id is not None:
                    # 获取下一个任务
                    next_job = self.job_queue.get_next_job()
                    
                    if next_job:
                        # 运行任务
                        self.run_job(next_job, gpu_id)
                
                # 定期清理旧任务
                if datetime.now().hour == 3:  # 每天凌晨3点
                    self.job_queue.clean_old_jobs()
                
            except Exception as e:
                logging.error(f"调度循环异常: {e}")
            
            # 等待下一次检查
            time.sleep(self.config["check_interval"])


def parse_args():
    parser = argparse.ArgumentParser(description='THETA 任务调度器')
    parser.add_argument('--base-dir', type=str, help='基础目录')
    parser.add_argument('--gpu-count', type=int, default=1, help='可用GPU数量')
    parser.add_argument('--daemon', action='store_true', help='以守护进程运行')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 更新配置
    if args.base_dir:
        CONFIG["base_dir"] = args.base_dir
    
    if args.gpu_count:
        CONFIG["gpu_count"] = args.gpu_count
    
    # 创建调度器
    scheduler = Scheduler(CONFIG)
    
    if args.daemon:
        # 以守护进程运行
        import daemon
        with daemon.DaemonContext():
            scheduler.schedule_loop()
    else:
        # 直接运行
        scheduler.schedule_loop()


if __name__ == '__main__':
    main()
