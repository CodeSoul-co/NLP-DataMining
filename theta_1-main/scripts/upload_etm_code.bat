@echo off
REM ============================================================
REM 2.12 代码完整上传脚本
REM 将 2.12 目录上传到 OSS 替换现有 THETA-main 代码
REM ============================================================

echo ============================================================
echo 2.12 代码上传脚本 (替换 THETA-main)
echo ============================================================

REM 设置变量
set OSS_BUCKET=oss://theta-prod-20260123
set LOCAL_BASE_DIR=d:\chalotte\OneDrive\桌面\2.12
set LOCAL_ETM_DIR=%LOCAL_BASE_DIR%\ETM
set LOCAL_SCRIPTS_DIR=%LOCAL_BASE_DIR%\scripts
set LOCAL_API_DIR=%LOCAL_BASE_DIR%\api
set OSS_ETM_DIR=%OSS_BUCKET%/code/ETM
set OSS_SCRIPTS_DIR=%OSS_BUCKET%/code/scripts
set OSS_API_DIR=%OSS_BUCKET%/code/api

echo.
echo [Step 1] 删除 OSS 上旧的 ETM 代码...
echo 执行: ossutil rm -rf %OSS_ETM_DIR%/
ossutil rm -rf %OSS_ETM_DIR%/

echo.
echo [Step 2] 上传新的 ETM 代码（不包含 __pycache__ 和 .ipynb_checkpoints）...

REM 上传主要 Python 文件
echo   上传根目录文件...
ossutil cp "%LOCAL_ETM_DIR%\main.py" %OSS_ETM_DIR%/main.py
ossutil cp "%LOCAL_ETM_DIR%\config.py" %OSS_ETM_DIR%/config.py
ossutil cp "%LOCAL_ETM_DIR%\prepare_data.py" %OSS_ETM_DIR%/prepare_data.py
ossutil cp "%LOCAL_ETM_DIR%\run_pipeline.py" %OSS_ETM_DIR%/run_pipeline.py
ossutil cp "%LOCAL_ETM_DIR%\requirements.txt" %OSS_ETM_DIR%/requirements.txt
ossutil cp "%LOCAL_ETM_DIR%\__init__.py" %OSS_ETM_DIR%/__init__.py
ossutil cp "%LOCAL_ETM_DIR%\README.md" %OSS_ETM_DIR%/README.md

REM 上传 bow 目录
echo   上传 bow 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\bow" %OSS_ETM_DIR%/bow/ --exclude "*.pyc" --exclude "__pycache__/*"

REM 上传 model 目录（包含 SBERT）
echo   上传 model 目录（包含 SBERT 模型）...
ossutil cp -r "%LOCAL_ETM_DIR%\model" %OSS_ETM_DIR%/model/ --exclude "*.pyc" --exclude "__pycache__/*" --exclude ".ipynb_checkpoints/*"

REM 上传 dataclean 目录
echo   上传 dataclean 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\dataclean" %OSS_ETM_DIR%/dataclean/ --exclude "*.pyc" --exclude "__pycache__/*"

REM 上传 data 目录
echo   上传 data 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\data" %OSS_ETM_DIR%/data/ --exclude "*.pyc" --exclude "__pycache__/*"

REM 上传 evaluation 目录
echo   上传 evaluation 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\evaluation" %OSS_ETM_DIR%/evaluation/ --exclude "*.pyc" --exclude "__pycache__/*"

REM 上传 visualization 目录
echo   上传 visualization 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\visualization" %OSS_ETM_DIR%/visualization/ --exclude "*.pyc" --exclude "__pycache__/*"

REM 上传 preprocessing 目录
echo   上传 preprocessing 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\preprocessing" %OSS_ETM_DIR%/preprocessing/ --exclude "*.pyc" --exclude "__pycache__/*"

REM 上传 models_config 目录
echo   上传 models_config 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\models_config" %OSS_ETM_DIR%/models_config/ --exclude "*.pyc" --exclude "__pycache__/*"

REM 上传 utils 目录
echo   上传 utils 目录...
ossutil cp -r "%LOCAL_ETM_DIR%\utils" %OSS_ETM_DIR%/utils/ --exclude "*.pyc" --exclude "__pycache__/*"

echo.
echo [Step 3] 删除 OSS 上旧的 scripts 代码...
ossutil rm -rf %OSS_SCRIPTS_DIR%/

echo.
echo [Step 4] 上传 scripts 目录...
ossutil cp -r "%LOCAL_SCRIPTS_DIR%" %OSS_SCRIPTS_DIR%/ --exclude "*.pyc" --exclude "__pycache__/*"

echo.
echo [Step 5] 删除 OSS 上旧的 api 代码...
ossutil rm -rf %OSS_API_DIR%/

echo.
echo [Step 6] 上传 api 目录...
ossutil cp -r "%LOCAL_API_DIR%" %OSS_API_DIR%/ --exclude "*.pyc" --exclude "__pycache__/*"

echo.
echo ============================================================
echo [完成] 2.12 代码已上传到 OSS (替换 THETA-main)
echo ============================================================
echo.
echo 上传内容:
echo   ETM/:
echo   - main.py, config.py, prepare_data.py, run_pipeline.py (支持 --job_id)
echo   - bow/, model/, dataclean/, data/, evaluation/
echo   - visualization/, preprocessing/, models_config/, utils/
echo   scripts/:
echo   - 所有 bash 脚本 (01_setup.sh ~ 14_start_agent_api.sh)
echo   - dlc_client.py (DLC 任务提交客户端)
echo   - 11 个新增批量训练脚本
echo   api/:
echo   - main.py (FastAPI 入口)
echo   - data_api.py (前端直传 OSS + DLC 自启动)
echo.
echo OSS 目录结构:
echo   /code/ETM/      - 训练代码
echo   /code/scripts/  - 脚本
echo   /code/api/      - API 服务
echo   /data/{job_id}/ - 用户数据 (按 job_id 隔离)
echo   /result/{job_id}/ - 训练结果 (按 job_id 隔离)
echo.
echo 下一步:
echo   1. 部署 API 服务到 ECS
echo   2. 前端调用 /api/data/presigned-url 获取上传链接
echo   3. 前端上传完成后调用 /api/data/upload-complete 触发 DLC 训练
echo   4. 前端轮询 /api/data/jobs/{job_id}/status 查看状态
echo   5. 训练完成后调用 /api/data/jobs/{job_id}/results 获取结果
echo.

pause
