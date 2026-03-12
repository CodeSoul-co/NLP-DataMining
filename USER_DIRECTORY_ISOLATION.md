# 用户目录隔离 — 前后端与 DLC 模块分工协作说明

> 目标：每个用户（如 `000001`）在 OSS/DLC 挂载卷上均拥有独立的数据目录和结果目录，
> 上传、训练、下载全程在自己的隔离命名空间内完成，多用户不互相干扰。

---

## 一、目录结构约定（双方共同遵守）

```
/mnt/
├── data/
│   └── {user_id}/                        ← 用户数据根目录
│       └── {job_id}/                     ← 上传的原始文件（临时）
│           └── xxx.csv / xxx.pdf / ...
│
├── result/
│   └── {user_id}/                        ← 用户结果根目录
│       └── {dataset}/                    ← 按数据集组织
│           ├── {mode}/                   ← zero_shot / few_shot 等
│           │   └── theta / ctm / lda ... ← 各模型输出
│           └── baseline/
│               └── bow/                  ← BOW 矩阵
│                   ├── bow_matrix.npz
│                   ├── vocab.txt
│                   └── vocab_embeddings.npy
│
└── code/ETM/                             ← 代码（只读挂载，不写入）
```

**关键变量（通过环境变量传递，DLC 容器启动时注入）：**

| 环境变量 | 值示例 | 用途 |
|---------|--------|------|
| `THETA_USER_ID` | `000001` | 用户 ID |
| `THETA_JOB_ID` | `abc123` | 本次任务 ID |
| `ETM_USER_DATA_DIR` | `/mnt/data/000001` | ETM 代码读取数据的根目录 |
| `ETM_USER_RESULT_DIR` | `/mnt/result/000001` | ETM 代码写入结果的根目录 |

> `config.py` 中 `DATA_DIR` 和 `RESULT_DIR` 已从这两个环境变量读取，
> **只要 DLC 容器启动时注入正确，ETM 代码无需硬编码路径**。

---

## 二、我需要做的（后端接入层 `data_api.py`）

### 2.1 修复 Step 1：改用命令行调用，绕开函数签名问题

**现在的错误写法（TypeError）：**
```python
python -c "
from prepare_data import prepare_baseline_data
prepare_baseline_data(dataset='{dataset}', vocab_size=5000, ...)  # ← 错！函数接受 args，不接受 kwargs
"
```

**改为命令行调用方式（稳定，不依赖内部函数签名）：**
```bash
# Step 1: 数据预处理
python prepare_data.py \
  --dataset {dataset} \
  --vocab_size 5000 \
  --language zh \
  --prepare          # 只做 prepare，不训练
```

> 好处：接口是命令行参数（稳定契约），双方解耦；ETM 内部改动不影响接入层。

### 2.2 修复 Step 2：embedding 合并进 Step 1，或等 DLC 同学提供新函数

**目前情况：** `generate_embeddings_for_baseline` 函数不存在，Step 2 必然失败。

**临时方案：** 删除 Step 2 的独立 Python 调用，embedding 由 `prepare_data.py` 在 Step 1 内部自动生成（它本来就会调用 `generate_sbert_embeddings`）。

**长期方案：** 等 DLC 同学提供 `generate_embeddings_for_baseline` 函数后补充。

### 2.3 修复 user_id 路径断言

在 `_submit_dlc_job` 函数顶部加断言，防止 `user_id` 为空导致路径无效：

```python
def _submit_dlc_job(job_id: str, user_id: int, dataset: str, ...):
    # 安全断言
    if not user_id:
        raise ValueError(f"user_id 为空，无法构建隔离目录！job_id={job_id}")
    if not dataset:
        raise ValueError(f"dataset 为空，无法构建结果目录！job_id={job_id}")
    
    # 调试：在 shell 脚本开头打印关键变量
    # echo "DEBUG: user_id={user_id}, dataset={dataset}, job_id={job_id}"
```

shell 脚本中同样加调试输出：
```bash
echo "=== KEY PATHS DEBUG ==="
echo "  USER_ID  : {user_id}"
echo "  JOB_ID   : {job_id}"
echo "  DATASET  : {dataset}"
echo "  DATA DIR : /mnt/data/{user_id}/{job_id}/"
echo "  RESULT   : /mnt/result/{user_id}/{dataset}/"
ls -la /mnt/data/{user_id}/{job_id}/ 2>&1 || echo "  [WARN] data dir not found"
```

### 2.4 结果下载 API（待补充）

用户下载结果时，API 必须从该用户自己的结果目录读取：

```python
@router.get("/result/{dataset}/download")
async def download_result(dataset: str, current_user: User = Depends(get_current_active_user)):
    # 用 current_user.id 构建路径，不允许用户访问他人目录
    result_path = Path(f"/mnt/result/{current_user.id}/{dataset}/")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="结果不存在")
    # ... 打包 zip 返回
```

> 注意：路径必须用 `current_user.id`（从 JWT token 解析出的），不可接受前端传入的 user_id 参数（防止越权访问）。

---

## 三、DLC 模块同学需要做的（`ETM/prepare_data.py` + `run_pipeline.py`）

### 3.1 新增 kwargs 包装函数（让接入层可以用 Python 直接调用）

在 `prepare_data.py` 末尾新增：

```python
def prepare_baseline_data_kwargs(
    dataset: str,
    vocab_size: int = 5000,
    language: str = 'zh',
    data_dir: str = None,
    result_dir: str = None,
    job_id: str = None,
    bow_only: bool = False,
    batch_size: int = 32,
):
    """
    包装函数：接受 keyword arguments，内部构建 argparse.Namespace 后调用原函数。
    接入层（data_api.py）通过该函数调用，不直接依赖 argparse。
    """
    import argparse
    args = argparse.Namespace(
        dataset=dataset,
        vocab_size=vocab_size,
        language=language,
        bow_only=bow_only,
        batch_size=batch_size,
    )
    # 如果传入了外部路径，临时覆盖全局变量
    if data_dir or result_dir:
        import ETM.config as cfg
        if data_dir:
            cfg.DATA_DIR = Path(data_dir)
        if result_dir:
            cfg.RESULT_DIR = Path(result_dir)
    return prepare_baseline_data(args)
```

### 3.2 新增 `generate_embeddings_for_baseline` 函数（Step 2 需要）

当前 prepare_baseline_data 内部会调用 `generate_sbert_embeddings`，
**如果接入层需要单独调用 embedding 生成**，请新增：

```python
def generate_embeddings_for_baseline(
    dataset: str,
    result_dir: str = None,
    job_id: str = None,
):
    """
    为 baseline 生成 SBERT 嵌入（单独步骤版本）。
    result_dir: 如果指定，写入该目录而非全局 RESULT_DIR。
    """
    effective_result_dir = Path(result_dir) / 'baseline' / dataset if result_dir \
        else Path(RESULT_DIR) / 'baseline' / dataset
    
    data_path = find_data_file(dataset)
    if data_path is None:
        raise FileNotFoundError(f"找不到数据文件: {dataset}")
    
    texts, _ = load_texts(data_path)
    generate_sbert_embeddings(texts, effective_result_dir)
    print(f"[Embedding] 生成完成 → {effective_result_dir}")
```

### 3.3 BOW 文件路径对齐（Bug 4）

**现状：** Step 1 把 BOW 文件写入 `/mnt/data/{dataset}/bow/`，
`run_pipeline.py` 从 `/mnt/result/{user_id}/{dataset}/baseline/bow/` 读取。

**需要统一为：**

| 步骤 | 当前路径 | 应改为 |
|-----|---------|--------|
| Step 1 写入 | `/mnt/data/{dataset}/bow/` | `/mnt/result/{user_id}/{dataset}/baseline/bow/` |
| run_pipeline 读取 | `/mnt/result/{user_id}/{dataset}/baseline/bow/` | 保持不变 |

修改 `prepare_baseline_data` 中 `result_dir` 的组装方式，确保 BOW 文件写到 `RESULT_DIR/baseline/{dataset}/bow/`，即当前 `config.py` 调的 `RESULT_DIR`（已经是带 user_id 的）。

### 3.4 确认 `run_pipeline.py` 路径通过环境变量注入正常工作

目前 `config.py` 已正确从环境变量读取：
```python
RESULT_DIR = Path(os.environ.get("ETM_USER_RESULT_DIR", ...))
DATA_DIR   = Path(os.environ.get("ETM_USER_DATA_DIR", ...))
```

**需要确认：** `run_pipeline.py` 导入 `config` 时，这两行是否在容器启动时（env var 注入后）才执行？

**注意：** 如果 `config.py` 被打包成 `.pyc` 或提前 import，env var 可能被忽略。
推荐在 `run_pipeline.py` 开头加检查：
```python
print(f"[CONFIG] DATA_DIR={DATA_DIR}, RESULT_DIR={RESULT_DIR}")
assert str(DATA_DIR) != "/root/autodl-tmp/data", \
    "ETM_USER_DATA_DIR 未注入，路径未隔离！"
```

---

## 四、接口约定（双方对齐的契约）

### DLC 容器启动时，后端注入的环境变量（由我负责）

```bash
export THETA_USER_ID="{user_id}"          # 必须非空
export THETA_JOB_ID="{job_id}"            # 必须非空
export ETM_USER_DATA_DIR="/mnt/data/{user_id}"
export ETM_USER_RESULT_DIR="/mnt/result/{user_id}"
```

### DLC 容器中，ETM代码读写路径规范（由 DLC 同学负责）

| 文件类型 | 路径 |
|---------|------|
| 上传原始文件 | `/mnt/data/{user_id}/{job_id}/` |
| 清洗后数据 | `/mnt/data/{user_id}/{dataset}/` |
| BOW 矩阵 | `/mnt/result/{user_id}/{dataset}/baseline/bow/` |
| 训练结果 | `/mnt/result/{user_id}/{dataset}/{mode}/{model}/` |
| 可视化图表 | `/mnt/result/{user_id}/{dataset}/{mode}/viz/` |

### 结果回传（OSS 同步）

训练完成后，DLC 容器需将 `/mnt/result/{user_id}/` 同步到 OSS：
```
OSS 路径：result/{user_id}/{dataset}/{mode}/
```
后端从 OSS 路径生成签名 URL 供用户下载，路径带 user_id，实现隔离。

---

## 五、优先级排序

| 优先级 | 事项 | 负责方 |
|--------|------|--------|
| P0 | 修复 Step 1 改用命令行调用（绕开 TypeError） | 我 |
| P0 | 删除 Step 2 独立调用（合并到 Step 1）或新增函数 | 我 + DLC同学 |
| P0 | 在 `_submit_dlc_job` 加 user_id 非空断言 + 调试日志 | 我 |
| P1 | 新增 `prepare_baseline_data_kwargs` 包装函数 | DLC同学 |
| P1 | 统一 BOW 写入路径 → `RESULT_DIR/baseline/{dataset}/bow/` | DLC同学 |
| P1 | 在 `run_pipeline.py` 加路径注入检查 assert | DLC同学 |
| P2 | 结果下载 API 用 `current_user.id` 路径隔离 | 我 |
| P2 | DLC 完成后将结果同步到 OSS `result/{user_id}/` | DLC同学 |
