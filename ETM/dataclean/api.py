"""
DataClean API Service

提供 RESTful API 接口，用于前端调用数据清洗和文件转换功能。
"""

import os
import sys
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.converter import TextConverter
from src.cleaner import TextCleaner
from src.consolidator import DataConsolidator

# 创建 FastAPI 应用
app = FastAPI(
    title="DataClean API",
    description="文本文件清洗和转换 API 服务",
    version="1.0.0"
)

# 从环境变量读取 CORS 配置
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",")] if cors_origins_env != "*" else ["*"]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建临时目录存储上传和处理后的文件
TEMP_DIR = Path(tempfile.gettempdir()) / "dataclean_api"
TEMP_DIR.mkdir(exist_ok=True)

# 全局组件实例
converter = TextConverter()
consolidator = DataConsolidator()


# 请求/响应模型
class CleanTextRequest(BaseModel):
    """文本清洗请求模型"""
    text: str
    language: str = "chinese"
    operations: Optional[List[str]] = None


class CleanTextResponse(BaseModel):
    """文本清洗响应模型"""
    cleaned_text: str
    original_length: int
    cleaned_length: int


class ProcessFileRequest(BaseModel):
    """文件处理请求模型"""
    language: str = "chinese"
    clean: bool = True
    operations: Optional[List[str]] = None


class ProcessFileResponse(BaseModel):
    """文件处理响应模型"""
    task_id: str
    status: str
    message: str
    file_count: Optional[int] = None


class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    message: str
    result_file: Optional[str] = None
    error: Optional[str] = None


# 任务存储（生产环境应使用 Redis 或数据库）
tasks: Dict[str, Dict[str, Any]] = {}


# 健康检查端点
@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "DataClean API"
    }


# 获取支持的文件格式
@app.get("/api/formats")
def get_supported_formats():
    """获取支持的文件格式列表"""
    return {
        "formats": converter.supported_formats,
        "count": len(converter.supported_formats)
    }


# 文本清洗端点
@app.post("/api/clean/text", response_model=CleanTextResponse)
def clean_text(request: CleanTextRequest):
    """清洗文本内容"""
    try:
        cleaner = TextCleaner(language=request.language)
        
        # 执行清洗
        cleaned_text = cleaner.clean_text(
            request.text,
            operations=request.operations
        )
        
        return CleanTextResponse(
            cleaned_text=cleaned_text,
            original_length=len(request.text),
            cleaned_length=len(cleaned_text)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 单文件上传和处理
@app.post("/api/upload/process")
async def upload_and_process(
    file: UploadFile = File(...),
    language: str = Form("chinese"),
    clean: bool = Form(True),
    operations: Optional[str] = Form(None)
):
    """上传文件并处理"""
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 检查文件格式
        file_ext = Path(file.filename).suffix.lower()
        if not converter.is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}"
            )
        
        # 保存上传的文件
        task_dir = TEMP_DIR / task_id
        task_dir.mkdir(exist_ok=True)
        input_file = task_dir / file.filename
        
        with open(input_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 解析清洗操作
        ops = None
        if operations:
            ops = operations.split(",") if isinstance(operations, str) else operations
        
        # 初始化任务状态
        tasks[task_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "开始处理文件...",
            "input_file": str(input_file),
            "language": language,
            "clean": clean,
            "operations": ops
        }
        
        # 处理文件
        try:
            cleaner = TextCleaner(language=language)
            output_file = task_dir / "result.csv"
            
            # 提取文本
            text = converter.extract_text(str(input_file))
            
            # 清洗文本（如果需要）
            if clean:
                text = cleaner.clean_text(text, operations=ops)
            
            # 创建CSV
            consolidator.create_oneline_csv(
                [str(input_file)],
                str(output_file),
                converter.extract_text,
                lambda t: cleaner.clean_text(t, operations=ops) if clean else t
            )
            
            # 更新任务状态
            tasks[task_id].update({
                "status": "completed",
                "progress": 100.0,
                "message": "处理完成",
                "result_file": str(output_file)
            })
            
        except Exception as e:
            tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "message": f"处理失败: {str(e)}"
            })
            raise HTTPException(status_code=500, detail=str(e))
        
        return ProcessFileResponse(
            task_id=task_id,
            status="completed",
            message="文件处理完成",
            file_count=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 批量文件上传和处理
@app.post("/api/upload/batch")
async def upload_batch_and_process(
    files: List[UploadFile] = File(...),
    language: str = Form("chinese"),
    clean: bool = Form(True),
    operations: Optional[str] = Form(None)
):
    """批量上传文件并处理"""
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 保存上传的文件
        task_dir = TEMP_DIR / task_id
        task_dir.mkdir(exist_ok=True)
        input_files = []
        
        for file in files:
            # 检查文件格式
            if not converter.is_supported(file.filename):
                continue  # 跳过不支持的文件
            
            input_file = task_dir / file.filename
            with open(input_file, "wb") as f:
                shutil.copyfileobj(file.file, f)
            input_files.append(str(input_file))
        
        if not input_files:
            raise HTTPException(
                status_code=400,
                detail="没有支持的文件格式"
            )
        
        # 解析清洗操作
        ops = None
        if operations:
            ops = operations.split(",") if isinstance(operations, str) else operations
        
        # 初始化任务状态
        tasks[task_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": f"开始处理 {len(input_files)} 个文件...",
            "input_files": input_files,
            "language": language,
            "clean": clean,
            "operations": ops
        }
        
        # 处理文件
        try:
            cleaner = TextCleaner(language=language)
            output_file = task_dir / "result.csv"
            
            # 创建CSV
            consolidator.create_oneline_csv(
                input_files,
                str(output_file),
                converter.extract_text,
                lambda t: cleaner.clean_text(t, operations=ops) if clean else t
            )
            
            # 更新任务状态
            tasks[task_id].update({
                "status": "completed",
                "progress": 100.0,
                "message": "批量处理完成",
                "result_file": str(output_file)
            })
            
        except Exception as e:
            tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "message": f"处理失败: {str(e)}"
            })
            raise HTTPException(status_code=500, detail=str(e))
        
        return ProcessFileResponse(
            task_id=task_id,
            status="completed",
            message=f"成功处理 {len(input_files)} 个文件",
            file_count=len(input_files)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 获取任务状态
@app.get("/api/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """获取任务处理状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    
    # 生成下载URL（如果完成）
    result_file = None
    if task["status"] == "completed" and "result_file" in task:
        result_file = f"/api/download/{task_id}"
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0.0),
        message=task.get("message", ""),
        result_file=result_file,
        error=task.get("error")
    )


# 下载处理结果
@app.get("/api/download/{task_id}")
def download_result(task_id: str):
    """下载处理结果文件"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    if "result_file" not in task:
        raise HTTPException(status_code=404, detail="结果文件不存在")
    
    result_file = Path(task["result_file"])
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="结果文件已过期")
    
    return FileResponse(
        path=str(result_file),
        filename="result.csv",
        media_type="text/csv"
    )


# 清理过期任务（可选，用于定期清理）
@app.delete("/api/task/{task_id}")
def delete_task(task_id: str):
    """删除任务及其文件"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_dir = TEMP_DIR / task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    
    del tasks[task_id]
    
    return {"message": "任务已删除"}


# 主函数
if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取端口
    port = int(os.environ.get("PORT", 8001))
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
