# THETA Core Code Packaging Script
# Pack core code to desktop for OSS upload

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

# Config - use resolved paths to handle Chinese characters
$SourceDir = (Resolve-Path "$PSScriptRoot").Path
$DesktopDir = (Resolve-Path "$PSScriptRoot\..").Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PackageName = "THETA_core_$Timestamp"
$TempDir = "C:\temp\$PackageName"
$ZipPath = "$DesktopDir\$PackageName.zip"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "THETA 核心代码打包脚本" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# 创建临时目录
Write-Host "`n[1/5] 创建临时目录..." -ForegroundColor Yellow
if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
New-Item -ItemType Directory -Path $TempDir | Out-Null

# 定义核心文件列表
$CoreFiles = @(
    # ETM 核心
    "ETM\config.py",
    "ETM\main.py",
    "ETM\run_pipeline.py",
    "ETM\prepare_data.py",
    "ETM\requirements.txt",
    
    # ETM 模型
    "ETM\model\__init__.py",
    "ETM\model\etm.py",
    "ETM\model\base.py",
    "ETM\model\trainer.py",
    "ETM\model\vocab_embedder.py",
    "ETM\model\baseline_trainer.py",
    "ETM\model\baseline_evaluator.py",
    "ETM\model\baseline_data.py",
    
    # ETM 预处理
    "ETM\preprocessing\__init__.py",
    "ETM\preprocessing\embedding_processor.py",
    
    # ETM 可视化
    "ETM\visualization\__init__.py",
    "ETM\visualization\run_visualization.py",
    
    # Embedding 模块
    "embedding\__init__.py",
    "embedding\main.py",
    "embedding\trainer.py",
    "embedding\trainer_v2.py",
    "embedding\embedder.py",
    "embedding\data_loader.py",
    "embedding\registry.py",
    
    # 脚本
    "scripts\run_full_pipeline.sh",
    "scripts\scheduler.py"
)

# 定义核心目录（整个目录复制）
$CoreDirs = @(
    "ETM\dataclean",
    "ETM\evaluation"
)

# 复制核心文件
Write-Host "`n[2/5] 复制核心文件..." -ForegroundColor Yellow
$CopiedCount = 0
foreach ($file in $CoreFiles) {
    $srcPath = Join-Path $SourceDir $file
    $dstPath = Join-Path $TempDir $file
    
    if (Test-Path $srcPath) {
        $dstDir = Split-Path $dstPath -Parent
        if (-not (Test-Path $dstDir)) {
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        }
        Copy-Item $srcPath $dstPath -Force
        $CopiedCount++
        Write-Host "  + $file" -ForegroundColor Green
    } else {
        Write-Host "  - $file (不存在)" -ForegroundColor DarkGray
    }
}

# 复制核心目录
Write-Host "`n[3/5] 复制核心目录..." -ForegroundColor Yellow
foreach ($dir in $CoreDirs) {
    $srcPath = Join-Path $SourceDir $dir
    $dstPath = Join-Path $TempDir $dir
    
    if (Test-Path $srcPath) {
        Copy-Item $srcPath $dstPath -Recurse -Force
        # 删除 __pycache__ 和 .ipynb_checkpoints
        Get-ChildItem -Path $dstPath -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path $dstPath -Recurse -Directory -Filter ".ipynb_checkpoints" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  + $dir/" -ForegroundColor Green
    } else {
        Write-Host "  - $dir/ (不存在)" -ForegroundColor DarkGray
    }
}

# 清理临时目录中的缓存
Write-Host "`n[4/5] 清理缓存文件..." -ForegroundColor Yellow
Get-ChildItem -Path $TempDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $TempDir -Recurse -Directory -Filter ".ipynb_checkpoints" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $TempDir -Recurse -File -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

# 压缩
Write-Host "`n[5/5] 压缩到桌面..." -ForegroundColor Yellow
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path "$TempDir\*" -DestinationPath $ZipPath -CompressionLevel Optimal

# 清理临时目录
Remove-Item -Recurse -Force $TempDir

# 统计
$ZipSize = (Get-Item $ZipPath).Length / 1MB
Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "打包完成!" -ForegroundColor Green
Write-Host "  文件数: $CopiedCount" -ForegroundColor White
Write-Host "  压缩包: $ZipPath" -ForegroundColor White
Write-Host "  大小: $([math]::Round($ZipSize, 2)) MB" -ForegroundColor White
Write-Host "=" * 60 -ForegroundColor Cyan
