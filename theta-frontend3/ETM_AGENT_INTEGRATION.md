# ETM Agent 前端集成指南

## 概述

已成功将 THETA_wait 中的 LangGraph Agent 前端功能集成到 Next.js 项目中。集成的功能包括：

1. **训练任务管理** (`/app/training`) - 通过对话或表单创建和管理 ETM 训练任务
2. **结果查看** (`/app/results`) - 查看训练结果、评估指标和主题词
3. **可视化展示** (`/app/visualizations`) - 查看训练结果的可视化图表

## 新增文件

### API 客户端
- `lib/api/etm-agent.ts` - ETM Agent API 客户端，提供与后端通信的接口

### Hooks
- `hooks/use-etm-websocket.ts` - WebSocket hook，用于实时接收训练进度更新

### 页面组件
- `app/training/page.tsx` - 训练任务管理页面
- `app/results/page.tsx` - 结果查看页面
- `app/visualizations/page.tsx` - 可视化展示页面

## 环境变量配置

在 `.env.local` 文件中添加以下配置：

```bash
# ETM Agent API 配置
NEXT_PUBLIC_ETM_AGENT_API_URL=http://localhost:8000
NEXT_PUBLIC_ETM_AGENT_WS_URL=ws://localhost:8000
```

如果后端部署在其他地址，请修改为实际地址：

```bash
NEXT_PUBLIC_ETM_AGENT_API_URL=https://your-api-domain.com
NEXT_PUBLIC_ETM_AGENT_WS_URL=wss://your-api-domain.com
```

## 启动后端服务

在启动前端之前，确保 LangGraph Agent 后端服务正在运行：

```bash
cd THETA_wait/langgraph_agent/backend
source activate jiqun  # 或您的 conda 环境
pip install -r requirements.txt
python run.py --port 8000
```

后端服务默认运行在 `http://localhost:8000`

## 使用方法

### 1. 训练任务管理页面

访问路径：`http://localhost:3000/training`

功能：
- **对话式创建任务**：输入自然语言命令，例如：
  - "训练 socialTwitter 数据集"
  - "使用 zero_shot 模式训练，20 个主题"
  - "训练 hatespeech，supervised 模式"
- **实时进度更新**：通过 WebSocket 实时接收训练进度
- **任务列表**：查看所有训练任务及其状态
- **任务管理**：可以取消正在运行的任务

### 2. 结果查看页面

访问路径：`http://localhost:3000/results`

功能：
- **结果列表**：查看所有训练结果
- **结果详情**：查看特定结果的基本信息、评估指标和主题词
- **主题词展示**：查看每个主题的 top-k 关键词

### 3. 可视化展示页面

访问路径：`http://localhost:3000/visualizations`

功能：
- **结果选择**：选择要查看的可视化结果
- **图表展示**：查看训练生成的各种可视化图表
- **图表下载**：下载可视化图表

## API 接口说明

### ETMAgentAPI 类

主要方法：

```typescript
// 健康检查
await ETMAgentAPI.healthCheck()

// 获取数据集列表
await ETMAgentAPI.getDatasets()

// 获取结果列表
await ETMAgentAPI.getResults()

// 获取特定结果
await ETMAgentAPI.getResult(dataset, mode)

// 获取主题词
await ETMAgentAPI.getTopicWords(dataset, mode, topK)

// 获取可视化列表
await ETMAgentAPI.getVisualizations(dataset, mode)

// 创建训练任务
await ETMAgentAPI.createTask({
  dataset: 'socialTwitter',
  mode: 'zero_shot',
  num_topics: 20,
  epochs: 50
})

// 获取任务状态
await ETMAgentAPI.getTask(taskId)

// 取消任务
await ETMAgentAPI.cancelTask(taskId)
```

### WebSocket Hook

```typescript
import { useETMWebSocket } from '@/hooks/use-etm-websocket'

function MyComponent() {
  const { isConnected, lastMessage, subscribe } = useETMWebSocket()
  
  // 订阅任务更新
  useEffect(() => {
    if (taskId) {
      subscribe(taskId)
    }
  }, [taskId, subscribe])
  
  // 处理消息
  useEffect(() => {
    if (lastMessage?.type === 'step_update') {
      console.log('步骤更新:', lastMessage.step, lastMessage.message)
    }
  }, [lastMessage])
}
```

## 集成到主页面

如果需要将新页面集成到主页面（`app/page.tsx`）的导航中，可以：

1. **添加导航项**：在侧边栏导航中添加新的导航项
2. **使用 Next.js Link**：使用 Next.js 的 `Link` 组件进行路由跳转
3. **或使用状态管理**：保持现有的状态管理方式，添加新的视图类型

示例（使用 Next.js Link）：

```tsx
import Link from 'next/link'

<NavItem
  icon={Zap}
  label="训练任务"
  onClick={() => {}}
  as={Link}
  href="/training"
/>
```

## 故障排查

### 1. API 连接失败

**问题**：无法连接到后端服务

**解决方案**：
- 检查后端服务是否运行：`curl http://localhost:8000/health`
- 确认 `.env.local` 中的 API URL 配置正确
- 检查防火墙设置

### 2. WebSocket 连接失败

**问题**：WebSocket 无法连接

**解决方案**：
- 确认 WebSocket URL 配置正确
- 检查后端是否支持 WebSocket
- 查看浏览器控制台的错误信息

### 3. 任务创建失败

**问题**：创建训练任务时出错

**解决方案**：
- 检查数据集是否存在
- 查看后端服务日志
- 确认参数格式正确

### 4. 结果加载失败

**问题**：无法加载训练结果

**解决方案**：
- 确认训练任务已完成
- 检查结果文件是否存在
- 查看后端服务日志

## 开发建议

1. **错误处理**：所有 API 调用都应包含错误处理
2. **加载状态**：在加载数据时显示加载状态
3. **用户反馈**：使用 Toast 或 Alert 组件提示用户操作结果
4. **实时更新**：利用 WebSocket 实现实时进度更新
5. **缓存策略**：考虑使用 React Query 或 SWR 进行数据缓存

## 相关文档

- [LangGraph Agent README](../../THETA_wait/langgraph_agent/README.md)
- [API 文档](../../THETA_wait/langgraph_agent/API_DOCUMENTATION.md)
- [ETM 项目结构](../../THETA_wait/ETM/PROJECT_STRUCTURE.md)

## 下一步

1. 将新页面集成到主页面导航
2. 添加更多可视化组件
3. 实现任务历史记录
4. 添加导出功能
5. 优化用户体验
