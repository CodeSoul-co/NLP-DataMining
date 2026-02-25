# 代码仓库上传指南

## 概述

本指南说明如何将 THETA 训练代码上传到 codesoul 组织的匿名仓库。

---

## 1. 上传前检查清单

### 1.1 需要上传的目录/文件

```
✅ ETM/                    # 核心训练代码
✅ api/                    # API 服务代码
✅ scripts/                # 训练脚本
✅ embedding/              # 嵌入处理代码
✅ agent/                  # Agent 代码（如需要）
✅ README.md               # 项目说明
✅ FRONTEND_INTEGRATION.md # 前端对接文档
✅ .gitignore              # Git 忽略规则
✅ .env.example            # 环境变量示例
```

### 1.2 不要上传的目录/文件

```
❌ data/                   # 原始数据（存储在 OSS）
❌ embedding_models/       # 预训练模型（存储在 OSS）
❌ result/                 # 训练结果（存储在 OSS）
❌ logs/                   # 日志文件
❌ *.npy, *.pkl, *.pt      # 二进制数据文件
❌ .env                    # 包含密钥的环境变量
❌ __pycache__/            # Python 缓存
```

---

## 2. 上传步骤

### 2.1 初始化 Git 仓库

```bash
cd d:\chalotte\OneDrive\桌面\2.12

# 初始化 Git
git init

# 添加远程仓库（替换为实际的 codesoul 仓库地址）
git remote add origin https://github.com/codesoul-org/theta-anonymous.git
```

### 2.2 检查 .gitignore 是否生效

```bash
# 查看将被追踪的文件
git status

# 确认 data/ 和其他大文件不在列表中
```

### 2.3 提交代码

```bash
# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 检查将要提交的文件
git status

# 提交
git commit -m "Initial commit: THETA topic model training code"

# 推送到远程仓库
git push -u origin main
```

---

## 3. 协作者下载说明

### 3.1 克隆仓库

```bash
git clone https://github.com/codesoul-org/theta-anonymous.git
cd theta-anonymous
```

### 3.2 安装依赖

```bash
# API 依赖
pip install -r api/requirements.txt

# 训练代码依赖
pip install -r ETM/requirements.txt
```

### 3.3 配置环境变量

```bash
# 复制环境变量示例
cp .env.example .env

# 编辑 .env 文件，填入实际的阿里云凭证
```

### 3.4 启动 API 服务

```bash
cd api
python main.py
```

---

## 4. OSS 数据和模型访问

### 4.1 OSS Bucket 信息

| 配置项 | 值 |
|--------|-----|
| Bucket 名称 | `theta-prod-20260123` |
| 区域 | 华东2（上海） |
| Endpoint | `oss-cn-shanghai.aliyuncs.com` |

### 4.2 目录结构

```
oss://theta-prod-20260123/
├── code/                          # 训练代码（DLC 挂载）
│   └── ETM/
├── data/                          # 用户数据
│   └── {job_id}/
├── result/                        # 训练结果
│   └── {job_id}/
├── embedding_models/              # 预训练嵌入模型
│   └── sbert/
│       └── all-MiniLM-L6-v2/
└── sbert/                         # SBERT 模型（备用路径）
```

### 4.3 DLC 挂载配置

在 DLC 任务中，OSS 数据集会挂载到 `/mnt` 目录：

| OSS 路径 | DLC 挂载路径 |
|----------|--------------|
| `oss://theta-prod-20260123/code/` | `/mnt/code/` |
| `oss://theta-prod-20260123/data/` | `/mnt/data/` |
| `oss://theta-prod-20260123/result/` | `/mnt/result/` |
| `oss://theta-prod-20260123/embedding_models/` | `/mnt/embedding_models/` |
| `oss://theta-prod-20260123/sbert/` | `/mnt/sbert/` |

---

## 5. 测试账号

| 字段 | 值 |
|------|-----|
| 用户名 | `admin` |
| 密码 | `admin123` |

---

## 6. 数据预处理流程

完整的数据处理流程包含以下步骤：

### 6.1 流程概览

```
原始 CSV → 数据清洗 → 分词 → 构建词汇表 → 生成 BOW → 生成嵌入 → 训练模型
```

### 6.2 脚本调用顺序

```bash
# 1. 数据清洗（可选，如果数据已清洗）
bash scripts/02_clean_data.sh --dataset my_data --language zh

# 2. 数据预处理（构建词汇表、BOW 矩阵）
bash scripts/03_prepare_data.sh --dataset my_data --vocab_size 5000 --language zh

# 3. 生成嵌入（SBERT + Word2Vec）
bash scripts/02_generate_embeddings.sh --dataset my_data --mode zero_shot

# 4. 训练 THETA 模型
bash scripts/04_train_theta.sh --dataset my_data --num_topics 20 --epochs 100

# 5. 训练基线模型
bash scripts/05_train_baseline.sh --dataset my_data --models lda,etm,ctm --num_topics 20
```

### 6.3 API 自动化流程

通过 API 上传数据后，系统会自动执行完整的预处理和训练流程：

1. 用户上传 CSV 文件到 OSS
2. API 触发 DLC 训练任务
3. DLC 任务自动执行：
   - 数据清洗
   - 分词和词汇表构建
   - BOW 矩阵生成
   - SBERT/Word2Vec 嵌入生成
   - 模型训练
   - 评估和可视化
4. 结果保存到 OSS
5. 用户通过 API 下载结果

---

## 7. 常见问题

### Q: 前端如何获取训练进度？

A: 通过轮询 `/api/data/jobs/{job_id}/status` 接口获取任务状态。

### Q: 如何支持新的数据格式？

A: 修改 `ETM/dataclean/` 目录下的数据清洗代码。

### Q: 如何添加新的模型？

A: 在 `ETM/model/` 目录下添加模型实现，并在 `ETM/model/registry.py` 中注册。

---

## 8. 联系方式

如有问题，请联系后端开发团队。
