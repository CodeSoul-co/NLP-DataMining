# THETA 前端集成指南

本文档提供THETA系统前端集成所需的关键信息，包括数据流程、接口变更和实现建议。

## 1. 数据流程图

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  用户上传CSV │────>│ 后端生成任务 │────>│ 任务队列处理 │────>│ 结果展示页面 │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 文件预览界面 │     │ 参数配置界面 │     │ 进度跟踪界面 │     │ 可视化展示  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## 2. 接口变更说明

### 2.1 新增接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/data/upload` | POST | 上传数据文件 |
| `/api/analysis/start` | POST | 提交分析任务 |
| `/api/analysis/status/{job_id}` | GET | 查询任务状态 |
| `/api/results/{job_id}` | GET | 获取分析结果 |
| `/api/download/{job_id}/{filename}` | GET | 下载结果文件 |
| `/api/jobs` | GET | 获取任务列表 |

### 2.2 参数变更

所有接口都新增了`job_id`参数，用于支持多用户并发处理。

### 2.3 响应格式变更

分析结果现在使用统一的`analysis_result.json`格式，包含以下主要部分：
- 任务元数据（ID、时间等）
- 主题列表（关键词、占比等）
- 图表URL列表
- 下载文件URL列表

## 3. 前端实现建议

### 3.1 文件上传组件

```jsx
// 示例代码（React）
const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  
  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/data/upload', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    setPreview(result);
    
    // 保存job_id用于后续操作
    localStorage.setItem('currentJobId', result.job_id);
  };
  
  return (
    <div>
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <button onClick={handleUpload}>上传</button>
      
      {preview && (
        <div>
          <h3>文件预览</h3>
          <p>行数: {preview.rows}</p>
          <p>列: {preview.columns.join(', ')}</p>
          <table>
            {/* 显示预览数据 */}
          </table>
        </div>
      )}
    </div>
  );
};
```

### 3.2 任务状态轮询

```jsx
// 示例代码（React）
const JobStatus = ({ jobId }) => {
  const [status, setStatus] = useState(null);
  
  useEffect(() => {
    const pollStatus = async () => {
      const response = await fetch(`/api/analysis/status/${jobId}`);
      const result = await response.json();
      setStatus(result);
      
      if (result.status !== 'completed' && result.status !== 'failed') {
        // 继续轮询
        setTimeout(pollStatus, 5000);
      }
    };
    
    pollStatus();
  }, [jobId]);
  
  return (
    <div>
      <h3>任务状态</h3>
      <p>状态: {status?.status}</p>
      
      {status?.status === 'queued' && (
        <p>队列位置: {status.queue_position}</p>
      )}
      
      {status?.status === 'running' && (
        <div>
          <p>当前阶段: {status.progress.stage_name}</p>
          <progress value={status.progress.percentage} max="100" />
          <p>{status.progress.percentage}%</p>
        </div>
      )}
    </div>
  );
};
```

### 3.3 结果展示组件

```jsx
// 示例代码（React）
const ResultView = ({ jobId }) => {
  const [result, setResult] = useState(null);
  
  useEffect(() => {
    const fetchResult = async () => {
      const response = await fetch(`/api/results/${jobId}`);
      const data = await response.json();
      setResult(data);
    };
    
    fetchResult();
  }, [jobId]);
  
  if (!result) return <div>加载中...</div>;
  
  return (
    <div>
      <h2>分析结果</h2>
      
      <div>
        <h3>主题列表</h3>
        {result.topics.map(topic => (
          <div key={topic.id}>
            <h4>{topic.name}</h4>
            <p>关键词: {topic.keywords.join(', ')}</p>
            <p>占比: {(topic.proportion * 100).toFixed(2)}%</p>
            <img src={topic.wordcloud_url} alt={`主题 ${topic.id} 词云`} />
          </div>
        ))}
      </div>
      
      <div>
        <h3>图表</h3>
        <img src={result.charts.topic_distribution} alt="主题分布" />
        <img src={result.charts.heatmap} alt="热力图" />
      </div>
      
      <div>
        <h3>下载</h3>
        <a href={result.downloads.report} download>下载报告</a>
        <a href={result.downloads.theta_csv} download>下载主题-文档矩阵</a>
      </div>
    </div>
  );
};
```

## 4. 路由设计建议

```jsx
// 示例代码（React Router）
const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<FileUpload />} />
        <Route path="/configure/:jobId" element={<ConfigureJob />} />
        <Route path="/status/:jobId" element={<JobStatus />} />
        <Route path="/results/:jobId" element={<ResultView />} />
        <Route path="/jobs" element={<JobList />} />
      </Routes>
    </Router>
  );
};
```

## 5. 兼容性处理

为了支持平滑迁移，建议实现一个适配层，处理新旧API格式的差异：

```jsx
// 示例代码（适配层）
const fetchResult = async (jobId) => {
  try {
    // 尝试新API
    const response = await fetch(`/api/v2/results/${jobId}`);
    if (response.ok) {
      return await response.json();
    }
    
    // 回退到旧API
    const oldResponse = await fetch(`/api/v1/results?dataset=${jobId}`);
    const oldData = await oldResponse.json();
    
    // 转换为新格式
    return convertToNewFormat(oldData);
  } catch (error) {
    console.error("获取结果失败", error);
    throw error;
  }
};

const convertToNewFormat = (oldData) => {
  // 将旧格式转换为新格式
  return {
    job_id: oldData.dataset,
    status: "completed",
    topics: oldData.topics.map((t, i) => ({
      id: i,
      name: t.name,
      keywords: t.words,
      proportion: t.weight
    })),
    // ...其他转换
  };
};
```

## 6. 测试计划

1. **单元测试**：
   - 测试各组件的渲染和交互
   - 测试API调用和数据处理

2. **集成测试**：
   - 测试完整的用户流程
   - 测试并发任务处理

3. **兼容性测试**：
   - 测试新旧API的兼容性
   - 测试不同浏览器的兼容性

## 7. 部署注意事项

1. **环境变量**：
   - API_BASE_URL: API基础URL
   - VERSION: API版本（v1/v2）

2. **构建配置**：
   - 确保正确处理静态资源路径
   - 配置适当的缓存策略

3. **监控**：
   - 实现前端错误监控
   - 添加用户行为分析

## 8. 联系方式

如有任何问题，请联系后端开发负责人：
- 邮箱：backend@example.com
- 电话：123-456-7890
