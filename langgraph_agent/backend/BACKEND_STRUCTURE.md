# 后端服务器目录结构

> 本文档描述 THETA 项目后端服务器的完整目录结构和各模块功能

**最后更新时间**: 2025-01-17

---

## 📁 目录树

```
langgraph_agent/backend/
│
├── app/                          # 主应用目录
│   ├── __init__.py
│   ├── main.py                   # FastAPI 应用入口
│   │
│   ├── api/                      # API 路由模块
│   │   ├── __init__.py
│   │   ├── auth.py              # 认证路由 (登录、注册、用户管理)
│   │   ├── routes.py            # 主路由 (数据集、任务、结果等)
│   │   ├── oss.py               # OSS 接口 (presign、列举，来自 theta-backend 合并)
│   │   ├── scripts.py           # 脚本执行相关路由
│   │   └── websocket.py         # WebSocket 实时通信
│   │
│   ├── agents/                   # LangGraph 代理模块
│   │   ├── __init__.py
│   │   ├── etm_agent.py         # ETM 代理主逻辑 (LangGraph 图构建)
│   │   └── nodes.py             # 节点实现 (预处理、嵌入、训练、评估、可视化)
│   │
│   ├── core/                     # 核心配置和工具
│   │   ├── __init__.py
│   │   ├── config.py            # 应用配置 (路径、GPU、环境变量等)
│   │   └── logging.py           # 日志配置
│   │
│   ├── models/                   # 数据模型
│   │   ├── __init__.py
│   │   └── user.py              # 用户模型 (SQLite 数据库操作)
│   │
│   ├── schemas/                  # Pydantic 数据模式
│   │   ├── __init__.py
│   │   ├── agent.py             # 代理相关请求/响应模型
│   │   ├── auth.py              # 认证相关请求/响应模型
│   │   └── data.py              # 数据集、结果等数据模型
│   │
│   ├── services/                 # 业务服务层
│   │   ├── __init__.py
│   │   ├── auth_service.py      # 认证服务 (JWT token 生成/验证)
│   │   ├── chat_service.py      # 聊天服务 (Qwen API 集成)
│   │   ├── oss_service.py       # OSS 服务 (presign、列举，来自 theta-backend 合并)
│   │   └── script_service.py    # 脚本执行服务
│   │
│   └── static/                   # 静态文件
│       └── index.html           # 前端 HTML (可选)
│
├── run.py                        # 启动脚本
├── requirements.txt              # Python 依赖
├── railway.json                  # Railway 部署配置
└── QWEN_API_SETUP.md            # Qwen API 设置文档
```

---

## 📋 模块详细说明

### 🔷 入口文件

#### `app/main.py`
- **功能**: FastAPI 应用初始化、路由注册、中间件配置
- **职责**:
  - 创建 FastAPI 应用实例
  - 注册 API 路由 (`/api/auth`, `/api`)
  - 配置 CORS 中间件
  - 挂载静态文件目录
  - 应用生命周期管理 (lifespan)

#### `run.py`
- **功能**: 启动脚本
- **使用**: `python run.py` 或 `uvicorn app.main:app --reload`

---

### 🔷 API 路由模块 (`app/api/`)

#### `routes.py` - 核心业务路由
**主要端点**:
- `GET /api/` - 健康检查
- `GET /api/health` - 详细健康检查 (GPU、目录状态)
- `GET /api/project` - 项目概览信息

**数据集管理**:
- `GET /api/datasets` - 列出所有数据集
- `POST /api/datasets/upload` - 上传数据集
- `DELETE /api/datasets/{dataset_name}` - 删除数据集

**训练任务**:
- `POST /api/tasks` - 创建训练任务
- `GET /api/tasks` - 列出所有任务
- `GET /api/tasks/{task_id}` - 获取任务状态
- `DELETE /api/tasks/{task_id}` - 取消任务

**结果查询**:
- `GET /api/results` - 列出所有结果
- `GET /api/results/{dataset}/{mode}/metrics` - 获取评估指标
- `GET /api/results/{dataset}/{mode}/topic-words` - 获取主题词
- `GET /api/results/{dataset}/{mode}/visualizations` - 列出可视化
- `GET /api/results/{dataset}/{mode}/visualization-data` - 获取可视化数据

**向量化预处理**:
- `POST /api/preprocessing/start` - 开始向量化
- `GET /api/preprocessing/check/{dataset}` - 检查向量化状态

**其他**:
- `POST /api/chat` - AI 助手聊天
- `POST /api/chat/suggestions` - 获取智能建议
- `POST /api/restart` - 重启后端服务

#### `auth.py` - 认证路由
**端点**:
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 登录 (OAuth2 form)
- `POST /api/auth/login-json` - 登录 (JSON)
- `GET /api/auth/me` - 获取当前用户信息
- `GET /api/auth/verify` - 验证 token
- `PUT /api/auth/profile` - 更新用户资料
- `POST /api/auth/change-password` - 修改密码

#### `oss.py` - OSS 接口（由 theta-backend 合并，可选）
- **端点**:
  - `POST /api/upload/presign` - 获取 OSS 直传预签名 URL（需登录，需配置 OSS）
  - `GET /api/oss/datasets` - 列举当前用户在 OSS 下的数据集（需登录，需配置 OSS）
- **配置**: `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_BUCKET_NAME`, `OSS_ENDPOINT`

#### `websocket.py` - WebSocket 实时通信
- **端点**: `/api/ws`
- **功能**: 实时推送任务进度更新
- **消息类型**: `step_update`, `task_update`

#### `scripts.py` - 脚本执行路由 (如已实现)
- 脚本列表、执行、任务管理等相关端点

---

### 🔷 LangGraph 代理模块 (`app/agents/`)

#### `etm_agent.py` - ETM 代理主逻辑
**功能**:
- 创建 LangGraph 状态图 (StateGraph)
- 定义工作流: `preprocess → embedding → training → evaluation → visualization`
- 任务状态管理 (内存存储)
- 任务执行入口 (`run_pipeline`)

**关键类/函数**:
- `ETMAgent` - 代理主类
- `create_etm_graph()` - 创建工作流图
- `create_initial_state()` - 创建初始状态

#### `nodes.py` - 节点实现
**节点列表**:

1. **`preprocess_node`** - 数据预处理
   - 构建词汇表 (`VocabBuilder`)
   - 生成 BOW 矩阵 (`BOWGenerator`)
   - 生成词嵌入 (`VocabEmbedder`, 使用 Qwen 模型)
   - 保存到 `result/{dataset}/{mode}/bow/`

2. **`embedding_node`** - 文档嵌入加载
   - 加载预计算的文档嵌入
   - 验证嵌入文件存在

3. **`training_node`** - ETM 模型训练
   - 创建 ETM 模型实例 (`engine_c.etm.ETM`)
   - PyTorch 训练循环 (epochs, loss, optimizer)
   - 保存模型参数 (theta, beta 矩阵)
   - 保存到 `result/{dataset}/{mode}/model/`

4. **`evaluation_node`** - 评估指标计算
   - 主题一致性 (Topic Coherence)
   - 主题多样性 (Topic Diversity)
   - 困惑度 (Perplexity)
   - 保存到 `result/{dataset}/{mode}/evaluation/`

5. **`visualization_node`** - 可视化生成
   - 主题词云
   - 主题分布图
   - 热力图等
   - 保存到 `result/{dataset}/{mode}/visualization/`

---

### 🔷 核心配置 (`app/core/`)

#### `config.py` - 应用配置
**配置项**:
- **路径配置**:
  - `BASE_DIR` - 项目根目录 (支持环境变量 `THETA_PROJECT_ROOT`, AutoDL 检测)
  - `ETM_DIR` - ETM 代码目录
  - `DATA_DIR` - 数据目录
  - `RESULT_DIR` - 结果目录
  - `QWEN_MODEL_PATH` - Qwen 模型路径

- **GPU 配置**:
  - `GPU_ID` - GPU 设备 ID
  - `DEVICE` - 设备类型 ("cuda"/"cpu")

- **服务器配置**:
  - `HOST` - 绑定地址
  - `PORT` - 端口号
  - `CORS_ORIGINS` - 允许的跨域来源

- **功能开关**:
  - `SIMULATION_MODE` - 模拟模式 (False=真实训练, True=模拟演示)

- **认证配置**:
  - `SECRET_KEY` - JWT 密钥
  - `ACCESS_TOKEN_EXPIRE_DAYS` - Token 过期天数

#### `logging.py` - 日志配置
- 配置日志格式、级别、输出目标

---

### 🔷 数据模型 (`app/models/`)

#### `user.py` - 用户模型
**功能**:
- SQLite 数据库操作
- 用户 CRUD 操作
- 密码哈希 (bcrypt)
- 用户认证 (`authenticate_user`)
- 用户信息更新 (`update_user`)
- 密码修改 (`change_password`)

**数据库表结构**:
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name TEXT,
    created_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
)
```

---

### 🔷 数据模式 (`app/schemas/`)

#### `agent.py` - 代理相关模式
**主要模型**:
- `TaskRequest` - 创建任务请求
- `TaskResponse` - 任务响应
- `AgentState` - 代理状态 (LangGraph)
- `ChatRequest` / `ChatResponse` - 聊天请求/响应
- `SuggestionsRequest` / `SuggestionsResponse` - 智能建议

#### `auth.py` - 认证相关模式
**主要模型**:
- `UserRegister` - 注册请求
- `UserLogin` - 登录请求
- `UserResponse` - 用户信息响应
- `UserUpdate` - 更新用户请求
- `PasswordChange` - 修改密码请求
- `Token` - Token 响应

#### `data.py` - 数据相关模式
**主要模型**:
- `DatasetInfo` - 数据集信息
- `ResultInfo` - 结果信息
- `VisualizationInfo` - 可视化信息
- `MetricsResponse` - 评估指标响应
- `ProjectInfo` - 项目概览信息

---

### 🔷 业务服务层 (`app/services/`)

#### `auth_service.py` - 认证服务
**功能**:
- JWT token 生成 (`create_access_token`)
- Token 验证 (`verify_token`)
- 获取当前用户 (`get_current_user`, `get_current_active_user`)
- OAuth2 密码流程支持

#### `chat_service.py` - 聊天服务
**功能**:
- 集成千问 (Qwen) API
- 消息处理和响应生成
- 上下文管理
- 操作执行 (创建任务、切换页面等)

#### `script_service.py` - 脚本执行服务
**功能**:
- 脚本管理和执行
- 任务状态跟踪

---

### 🔷 配置文件

#### `requirements.txt`
Python 依赖包列表，包括:
- `fastapi` - Web 框架
- `uvicorn` - ASGI 服务器
- `langgraph` - 工作流图框架
- `torch` - PyTorch 深度学习框架
- `pydantic` - 数据验证
- `python-jose` - JWT 处理
- `passlib` - 密码哈希
- 其他依赖...

#### `railway.json`
Railway 平台部署配置

#### `QWEN_API_SETUP.md`
Qwen API 设置和配置说明

---

## 🏗️ 架构设计

### 分层架构

```
┌─────────────────────────────────────┐
│      API Layer (FastAPI Routes)     │  ← HTTP/WebSocket 接口
├─────────────────────────────────────┤
│      Service Layer                  │  ← 业务逻辑
├─────────────────────────────────────┤
│      Agent Layer (LangGraph)        │  ← 工作流编排
├─────────────────────────────────────┤
│      Data Layer (Models/Schemas)    │  ← 数据访问
└─────────────────────────────────────┘
```

### 数据流

1. **客户端请求** → API 路由 (`routes.py`, `auth.py`)
2. **路由处理** → 服务层 (`services/`) 或 代理层 (`agents/`)
3. **业务逻辑** → LangGraph 节点执行 (`nodes.py`)
4. **数据访问** → 模型/模式层 (`models/`, `schemas/`)
5. **响应返回** → API 路由 → 客户端

### 关键特性

- **真实训练模式**: `SIMULATION_MODE=False` 时执行真实的 ETM 模型训练
- **模拟模式**: `SIMULATION_MODE=True` 时仅模拟训练流程（用于演示）
- **异步处理**: 使用 FastAPI 的 `BackgroundTasks` 处理长时间运行的任务
- **实时通信**: WebSocket 推送任务进度更新
- **认证授权**: JWT token 认证，支持用户注册、登录、资料管理

---

## 📝 注意事项

1. **路径配置**: 
   - 支持环境变量 `THETA_PROJECT_ROOT` 自定义项目根目录
   - 自动检测 AutoDL 服务器环境 (`/root/autodl-tmp`)

2. **模型文件**:
   - Qwen 嵌入模型路径: `{BASE_DIR}/qwen3_embedding_0.6B`
   - ETM 代码路径: `{BASE_DIR}/ETM`

3. **数据库**:
   - 用户数据库: SQLite (`{DATA_DIR}/../users.db`)

4. **结果存储**:
   - 结果目录结构: `result/{dataset}/{mode}/{step}/`
   - 步骤: `bow/`, `embeddings/`, `model/`, `evaluation/`, `visualization/`

---

## 🔗 相关文档

- [架构设计文档](../ARCHITECTURE.md)
- [开发进度文档](../DEVELOPMENT_PROGRESS.md)
- [设置指南](../SETUP_GUIDE.md)
- [Qwen API 设置](QWEN_API_SETUP.md)
