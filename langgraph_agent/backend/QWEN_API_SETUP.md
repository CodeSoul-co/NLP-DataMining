# 千问 API 接入说明

## ✅ 已完成

1. ✅ 添加了 `dashscope` 依赖到 `requirements.txt`
2. ✅ 在 `config.py` 中配置了 Qwen API 相关设置
3. ✅ 更新了 `ChatService` 以支持千问 API
4. ✅ 添加了 `/api/chat` 端点到 `routes.py`

## 📦 安装依赖

```bash
cd langgraph_agent/backend
pip install dashscope>=1.17.0
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 🔑 API Key 配置

API Key 已配置在 `app/core/config.py` 中：

```python
QWEN_API_KEY: Optional[str] = Field(
    default="sk-ca1e46556f584e50aa74a2f6ff5659f0",
    description="Qwen API Key for chat service"
)
QWEN_MODEL: str = "qwen-turbo"  # qwen-turbo, qwen-plus, qwen-max
```

### 通过环境变量配置（推荐）

创建 `.env` 文件：

```bash
QWEN_API_KEY=sk-你的DashScope密钥
QWEN_MODEL=qwen-turbo
```

## 🚀 使用方式

### API 端点

**POST** `/api/chat`

**请求体：**
```json
{
  "message": "训练 socialTwitter 数据集",
  "context": {}
}
```

**响应：**
```json
{
  "message": "好的，我将为您启动训练任务...",
  "action": "start_task",
  "data": {
    "dataset": "socialTwitter",
    "mode": "zero_shot",
    "num_topics": 20
  }
}
```

### 前端调用

前端已经配置好了 API 调用：

```typescript
const response = await ETMAgentAPI.chat("训练 socialTwitter 数据集");
```

## 🔄 工作流程

1. **用户发送消息** → 前端调用 `/api/chat`
2. **ChatService 处理**：
   - 如果 `dashscope` 已安装且 `QWEN_API_KEY` 配置 → 使用千问 API
   - 否则 → 使用规则匹配回退模式
3. **千问 API 处理**：
   - 构建系统提示词（包含数据集、任务等上下文）
   - 调用千问 API 获取 AI 回复
   - 尝试从 AI 回复中提取 JSON 格式的操作数据
   - 如果没有 JSON，则使用规则匹配提取意图
4. **返回响应** → 前端根据 `action` 和 `data` 执行相应操作

## 🎯 支持的模型

- `qwen-turbo` - 快速响应（默认）
- `qwen-plus` - 更好的质量
- `qwen-max` - 最佳质量

在 `config.py` 或 `.env` 文件中修改 `QWEN_MODEL` 来切换模型。

## 🛠️ 故障排除

### 1. dashscope 未安装

**错误**：`dashscope not installed`

**解决**：
```bash
pip install dashscope>=1.17.0
```

### 2. API Key 未配置

**错误**：Chat service 使用规则匹配回退模式

**解决**：确保 `QWEN_API_KEY` 在配置文件中设置

### 3. API 调用失败

**错误**：`Qwen API error: 401` 或 `403`

**解决**：检查 API Key 是否正确，是否有足够的配额

## 📝 系统提示词

ChatService 会自动构建包含以下信息的系统提示词：

- 可用数据集列表
- 训练任务参数
- 支持的操作（训练、查看状态、查看结果等）
- 响应格式要求（JSON 操作数据）

系统提示词会根据当前系统状态动态更新。

## 🔍 测试

### 使用 curl 测试

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "列出所有数据集"}'
```

### 使用 Python 测试

```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "训练 socialTwitter 数据集"}
)
print(response.json())
```

## 📚 相关文档

- [DashScope Python SDK](https://help.aliyun.com/zh/dashscope/developer-reference/api-details)
- [Qwen API 文档](https://help.aliyun.com/zh/dashscope/developer-reference/api-details-9)
