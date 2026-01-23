/**
 * ETM Agent API Client
 * 用于与后端 LangGraph Agent 通信
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ==================== 类型定义 ====================

export interface TaskResponse {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_step?: string;
  progress: number;
  message?: string;
  
  // Task configuration
  dataset?: string;
  mode?: string;
  num_topics?: number;
  
  // Results
  metrics?: Record<string, number>;
  topic_words?: Record<string, string[]>;
  visualization_paths?: string[];
  
  // Timing
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  
  // Error
  error_message?: string;
  
  // Deprecated (for backwards compatibility)
  result?: any;
  error?: string;
}

export interface CreateTaskRequest {
  dataset: string;
  mode: 'zero_shot' | 'unsupervised' | 'supervised';
  num_topics?: number;
  vocab_size?: number;
  epochs?: number;
  batch_size?: number;
}

export interface DatasetInfo {
  name: string;
  path: string;
  file_count?: number;
  total_size?: string;
}

export interface ResultInfo {
  dataset: string;
  mode: string;
  timestamp: string;
  path: string;
  
  // Model info
  num_topics?: number;
  vocab_size?: number;
  epochs_trained?: number;
  
  // Metrics
  metrics?: Record<string, number>;
  
  // Available files
  has_model?: boolean;
  has_theta?: boolean;
  has_beta?: boolean;
  has_topic_words?: boolean;
  has_visualizations?: boolean;
}

export interface VisualizationInfo {
  name: string;
  type: string;
  path: string;
  url?: string;
}

export interface TopicWord {
  word: string;
  weight: number;
}

export interface MetricsResponse {
  coherence?: number;
  diversity?: number;
  perplexity?: number;
  [key: string]: any;
}

export interface PreprocessingJob {
  job_id: string;
  dataset: string;
  model: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  created_at?: string;
  bow_file?: string;
  embedding_file?: string;
}

export interface PreprocessingStatus {
  has_bow: boolean;
  has_embeddings: boolean;
  bow_file?: string;
  embedding_file?: string;
}

// ==================== 脚本执行类型 ====================

export interface ScriptParameter {
  name: string;
  type: string;
  required?: boolean;
  default?: string;
  description: string;
}

export interface ScriptInfo {
  id: string;
  name: string;
  description: string;
  parameters: ScriptParameter[];
  category: string;
}

export interface ScriptJob {
  job_id: string;
  script_id: string;
  script_name: string;
  parameters: Record<string, string>;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  logs: string[];
  exit_code?: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  error_message?: string;
}

export interface ExecuteScriptRequest {
  script_id: string;
  parameters: Record<string, string>;
}

export interface ExecuteScriptResponse {
  job_id: string;
  script_id: string;
  script_name: string;
  status: string;
  message: string;
}

// ==================== API 请求函数 ====================

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  // 获取 token（如果存在）
  const token = localStorage.getItem('access_token');
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options?.headers,
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch (error: any) {
    // 网络错误（连接失败、CORS、超时等）
    const errorMessage = error.message || 'Network error';
    if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
      throw new Error('无法连接到后端服务。请检查：\n1. 后端是否在本地运行（如 ./start.sh 或 uvicorn）\n2. NEXT_PUBLIC_API_URL 是否为 http://localhost:8000\n3. 网络连接是否正常');
    }
    throw error;
  }

  if (!response.ok) {
    // 对于 404 错误，提供更详细的错误信息
    if (response.status === 404) {
      const error = await response.json().catch(() => ({ detail: 'Not Found' }));
      const errorMessage = error.detail || 'Not Found';
      throw new Error(
        errorMessage === 'Not Found' 
          ? `API 端点不存在: ${endpoint}。请检查后端服务是否已更新。`
          : errorMessage
      );
    }
    
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// ==================== ETM Agent API ====================

export const ETMAgentAPI = {
  // ========== 健康检查 ==========
  async healthCheck(): Promise<{ status: string; gpu_available: boolean }> {
    return fetchApi('/api/health');
  },

  // ========== 数据集管理 ==========
  async getDatasets(): Promise<DatasetInfo[]> {
    return fetchApi('/api/datasets');
  },

  async getDataset(name: string): Promise<DatasetInfo> {
    return fetchApi(`/api/datasets/${name}`);
  },

  async uploadDataset(
    files: File[],
    datasetName: string,
    onProgress?: (progress: number) => void
  ): Promise<{
    success: boolean;
    message: string;
    dataset_name: string;
    file_count: number;
    total_size: number;
    files: string[];
  }> {
    const formData = new FormData();
    formData.append('dataset_name', datasetName);
    
    // 计算总文件大小（用于真实进度计算）
    let totalFileSize = 0;
    files.forEach(file => {
      formData.append('files', file);
      totalFileSize += file.size;
    });

    console.log(`[Upload] Starting upload: ${files.length} files, total size: ${(totalFileSize / 1024).toFixed(2)} KB`);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      let lastProgress = 0;
      let progressUpdateCount = 0;
      
      // 在上传开始时设置进度为 0%
      if (onProgress) {
        onProgress(0);
        lastProgress = 0;
      }
      
      // 监听上传进度事件（这是浏览器原生事件，显示真实上传进度）
      xhr.upload.addEventListener('progress', (event) => {
        progressUpdateCount++;
        if (onProgress && event.total > 0) {
          // 使用真实的上传进度（event.loaded / event.total）
          const progress = Math.min(Math.round((event.loaded / event.total) * 100), 99);
          
          // 只在进度实际变化时更新，避免频繁更新
          if (progress !== lastProgress) {
            lastProgress = progress;
            console.log(`[Upload Progress] ${progress}% (${(event.loaded / 1024).toFixed(2)} KB / ${(event.total / 1024).toFixed(2)} KB)`);
            onProgress(progress);
          }
        } else if (onProgress && event.loaded > 0) {
          // 如果没有 total 信息但已经有加载数据，使用文件大小估算
          const estimatedProgress = Math.min(Math.round((event.loaded / totalFileSize) * 100), 95);
          if (estimatedProgress !== lastProgress) {
            lastProgress = estimatedProgress;
            console.log(`[Upload Progress] ~${estimatedProgress}% (estimated from ${(event.loaded / 1024).toFixed(2)} KB loaded)`);
            onProgress(estimatedProgress);
          }
        }
      });

      xhr.addEventListener('loadstart', () => {
        console.log('[Upload] Load started');
        if (onProgress && lastProgress === 0) {
          onProgress(1); // 至少显示1%表示已开始
        }
      });

      xhr.addEventListener('load', () => {
        console.log(`[Upload] Load complete, status: ${xhr.status}, readyState: ${xhr.readyState}, progress events: ${progressUpdateCount}`);
        
        // 上传完成，设置进度为 100%
        if (onProgress) {
          onProgress(100);
        }
        
        // status === 0 通常表示网络错误（连接失败、CORS 错误等）
        if (xhr.status === 0) {
          reject(new Error('无法连接到后端服务。请检查：\n1. 后端是否在本地运行（如 ./start.sh 或 uvicorn）\n2. NEXT_PUBLIC_API_URL 是否为 http://localhost:8000\n3. 网络连接是否正常'));
          return;
        }
        
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            console.log('[Upload] Success:', response);
            resolve(response);
          } catch (e) {
            console.error('[Upload] Failed to parse response:', e);
            reject(new Error('Invalid response format'));
          }
        } else if (xhr.status === 405) {
          reject(new Error('上传接口不可用，请确保后端服务已重启并支持文件上传'));
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || `服务器错误 (HTTP ${xhr.status})`));
          } catch {
            reject(new Error(`服务器错误 (HTTP ${xhr.status})`));
          }
        }
      });

      xhr.addEventListener('error', (e) => {
        console.error('[Upload] Network error:', {
          error: e,
          readyState: xhr.readyState,
          status: xhr.status,
          statusText: xhr.statusText,
          responseText: xhr.responseText?.substring(0, 100)
        });
        
        // XMLHttpRequest error 事件通常发生在网络层错误（连接失败、超时等）
        // readyState 通常是 4（完成）或 0（未初始化）
        // status 通常是 0
        const errorMsg = xhr.status === 0 
          ? '无法连接到后端服务。请检查：\n1. 后端是否在本地运行（如 ./start.sh）\n2. 后端服务是否正在运行\n3. 网络连接是否正常'
          : `网络错误 (状态码: ${xhr.status}): 上传过程中连接中断`;
        
        reject(new Error(errorMsg));
      });

      xhr.addEventListener('abort', () => {
        console.warn('[Upload] Upload aborted');
        reject(new Error('Upload aborted'));
      });

      xhr.open('POST', `${API_BASE_URL}/api/datasets/upload`);
      
      // 设置超时时间（30 秒）
      xhr.timeout = 30000;
      
      // 超时处理
      xhr.addEventListener('timeout', () => {
        console.error('[Upload] Timeout:', {
          readyState: xhr.readyState,
          status: xhr.status
        });
        reject(new Error('上传超时（30 秒）。请检查：\n1. 网络连接是否稳定\n2. 文件是否过大\n3. 后端服务是否正常响应'));
      });
      
      // 确保不设置额外的 headers，让浏览器自动处理 Content-Type 和 Content-Length
      // 这对于 FormData 上传很重要
      
      xhr.send(formData);
    });
  },

  async deleteDataset(name: string): Promise<{ success: boolean; message: string }> {
    return fetchApi(`/api/datasets/${name}`, { method: 'DELETE' });
  },

  // ========== 任务管理 (Task Center API) ==========
  
  /**
   * 获取任务列表（支持过滤和分页）
   */
  async getTasks(params?: {
    status?: string;
    dataset?: string;
    limit?: number;
    offset?: number;
  }): Promise<TaskResponse[]> {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set('status', params.status);
    if (params?.dataset) searchParams.set('dataset', params.dataset);
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    if (params?.offset) searchParams.set('offset', params.offset.toString());
    
    const query = searchParams.toString();
    return fetchApi(`/api/tasks${query ? `?${query}` : ''}`);
  },

  /**
   * 获取任务统计信息
   */
  async getTaskStats(): Promise<{
    total: number;
    pending: number;
    running: number;
    completed: number;
    failed: number;
    cancelled: number;
  }> {
    return fetchApi('/api/tasks/stats');
  },

  /**
   * 获取单个任务详情
   */
  async getTask(taskId: string): Promise<TaskResponse> {
    return fetchApi(`/api/tasks/${taskId}`);
  },

  /**
   * 获取任务执行日志
   */
  async getTaskLogs(taskId: string, tail: number = 50): Promise<{
    task_id: string;
    status: string;
    logs: Array<{
      step: string;
      status: string;
      message: string;
      timestamp: string;
    }>;
    total_count: number;
  }> {
    return fetchApi(`/api/tasks/${taskId}/logs?tail=${tail}`);
  },

  /**
   * 创建新任务（Fire-and-Forget 模式）
   * 立即返回 task_id，任务在后台执行
   */
  async createTask(request: CreateTaskRequest): Promise<TaskResponse> {
    return fetchApi('/api/tasks', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  /**
   * 取消正在运行的任务
   */
  async cancelTask(taskId: string): Promise<{ message: string }> {
    return fetchApi(`/api/tasks/${taskId}`, {
      method: 'DELETE',
    });
  },

  /**
   * 轮询任务状态直到完成
   * @param taskId 任务ID
   * @param onProgress 进度回调
   * @param interval 轮询间隔（毫秒）
   * @param timeout 超时时间（毫秒）
   */
  async pollTaskUntilDone(
    taskId: string,
    onProgress?: (task: TaskResponse) => void,
    interval: number = 2000,
    timeout: number = 3600000 // 1 hour default
  ): Promise<TaskResponse> {
    const startTime = Date.now();
    
    while (true) {
      const task = await this.getTask(taskId);
      
      if (onProgress) {
        onProgress(task);
      }
      
      // 任务完成
      if (['completed', 'failed', 'cancelled'].includes(task.status)) {
        return task;
      }
      
      // 超时检查
      if (Date.now() - startTime > timeout) {
        throw new Error(`Task polling timeout after ${timeout / 1000} seconds`);
      }
      
      // 等待下一次轮询
      await new Promise(resolve => setTimeout(resolve, interval));
    }
  },

  // ========== 结果查询 ==========
  async getResults(): Promise<ResultInfo[]> {
    return fetchApi('/api/results');
  },

  async getResult(dataset: string, mode: string): Promise<ResultInfo> {
    return fetchApi(`/api/results/${dataset}/${mode}`);
  },

  async getTopicWords(dataset: string, mode: string, topK: number = 10): Promise<Record<string, string[]>> {
    return fetchApi(`/api/results/${dataset}/${mode}/topic-words?top_k=${topK}`);
  },

  async getMetrics(dataset: string, mode: string): Promise<MetricsResponse> {
    return fetchApi(`/api/results/${dataset}/${mode}/metrics`);
  },

  // ========== 可视化 ==========
  async getVisualizations(dataset: string, mode: string): Promise<VisualizationInfo[]> {
    return fetchApi(`/api/results/${dataset}/${mode}/visualizations`);
  },

  async getVisualizationData(
    dataset: string, 
    mode: string, 
    dataType: 'topic_distribution' | 'doc_topic_distribution' | 'topic_similarity'
  ): Promise<{
    topics?: string[];
    proportions?: number[];
    topic_words?: Record<string, string[]>;
    documents?: string[];
    distributions?: number[][];
    num_topics?: number;
    similarity_matrix?: number[][];
  }> {
    return fetchApi(`/api/results/${dataset}/${mode}/visualization-data?data_type=${dataType}`);
  },

  // ========== 聊天接口 ==========
  async chat(message: string, context?: Record<string, unknown>): Promise<{ 
    message: string; 
    response?: string;  // 兼容旧格式
    action?: string; 
    task_id?: string;
    data?: Record<string, unknown>;
  }> {
    return fetchApi('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message, context }),
    });
  },

  // ========== 对话历史 ==========
  async saveConversationHistory(sessionId: string, messages: Array<{ role: string; content: string }>): Promise<{ message: string; session_id: string; message_count: number }> {
    return fetchApi('/api/chat/history', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, messages }),
    });
  },

  async getConversationHistory(sessionId: string): Promise<{ session_id: string; messages: Array<any>; count: number }> {
    return fetchApi(`/api/chat/history/${sessionId}`);
  },

  async clearConversationHistory(sessionId: string): Promise<{ message: string; session_id: string }> {
    return fetchApi(`/api/chat/history/${sessionId}`, {
      method: 'DELETE',
    });
  },

  // ========== 智能建议 ==========
  async getSuggestions(context?: Record<string, unknown>): Promise<{ suggestions: Array<{ text: string; action: string; description: string; data?: Record<string, unknown> }> }> {
    return fetchApi('/api/chat/suggestions', {
      method: 'POST',
      body: JSON.stringify(context || {}),
    });
  },

  // ========== 预处理 (Embedding) ==========
  async getEmbeddingModels(): Promise<{ models: string[] }> {
    return fetchApi('/api/preprocessing/models');
  },

  async startPreprocessing(params: {
    dataset: string;
    text_column?: string;  // Optional: if not provided, backend will auto-detect
    config?: {
      embedding_model?: string;
      [key: string]: any;
    };
  }): Promise<PreprocessingJob> {
    return fetchApi('/api/preprocessing/start', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },

  async getPreprocessingJob(jobId: string): Promise<PreprocessingJob> {
    return fetchApi(`/api/preprocessing/${jobId}`);
  },

  async getPreprocessingJobs(): Promise<PreprocessingJob[]> {
    return fetchApi('/api/preprocessing');
  },

  async cancelPreprocessingJob(jobId: string): Promise<{ message: string }> {
    return fetchApi(`/api/preprocessing/${jobId}`, {
      method: 'DELETE',
    });
  },

  async checkPreprocessingStatus(dataset: string): Promise<PreprocessingStatus> {
    return fetchApi(`/api/preprocessing/check/${dataset}`);
  },

  // ========== 脚本执行 API ==========
  
  /**
   * 获取所有可用脚本列表
   */
  async getScripts(): Promise<ScriptInfo[]> {
    return fetchApi('/api/scripts');
  },

  /**
   * 获取脚本分类
   */
  async getScriptCategories(): Promise<Record<string, Array<{ id: string; name: string; description: string }>>> {
    return fetchApi('/api/scripts/categories');
  },

  /**
   * 获取指定脚本信息
   */
  async getScript(scriptId: string): Promise<ScriptInfo> {
    return fetchApi(`/api/scripts/${scriptId}`);
  },

  /**
   * 执行脚本
   */
  async executeScript(request: ExecuteScriptRequest): Promise<ExecuteScriptResponse> {
    return fetchApi('/api/scripts/execute', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  /**
   * 获取所有脚本任务列表
   */
  async getScriptJobs(): Promise<ScriptJob[]> {
    return fetchApi('/api/scripts/jobs');
  },

  /**
   * 获取指定任务状态
   */
  async getScriptJob(jobId: string): Promise<ScriptJob> {
    return fetchApi(`/api/scripts/jobs/${jobId}`);
  },

  /**
   * 获取任务日志
   */
  async getScriptJobLogs(jobId: string, tail: number = 100): Promise<{
    job_id: string;
    status: string;
    logs: string[];
    total_lines: number;
  }> {
    return fetchApi(`/api/scripts/jobs/${jobId}/logs?tail=${tail}`);
  },

  /**
   * 取消任务
   */
  async cancelScriptJob(jobId: string): Promise<{ message: string; job_id: string }> {
    return fetchApi(`/api/scripts/jobs/${jobId}`, {
      method: 'DELETE',
    });
  },

  // ========== 便捷脚本执行方法 ==========

  /**
   * 执行 ETM 训练
   */
  async runTraining(params: {
    dataset: string;
    mode?: string;
    num_topics?: number;
    epochs?: number;
    batch_size?: number;
  }): Promise<ExecuteScriptResponse> {
    return fetchApi('/api/scripts/train', {
      method: 'POST',
      body: JSON.stringify({
        dataset: params.dataset,
        mode: params.mode || 'zero_shot',
        num_topics: params.num_topics || 20,
        epochs: params.epochs || 50,
        batch_size: params.batch_size || 64,
      }),
    });
  },

  /**
   * 生成 Embedding
   */
  async runEmbedding(params: {
    dataset: string;
    mode?: string;
    epochs?: number;
    batch_size?: number;
  }): Promise<ExecuteScriptResponse> {
    return fetchApi('/api/scripts/embedding', {
      method: 'POST',
      body: JSON.stringify({
        dataset: params.dataset,
        mode: params.mode || 'zero_shot',
        epochs: params.epochs || 3,
        batch_size: params.batch_size || 16,
      }),
    });
  },

  /**
   * 运行评估
   */
  async runEvaluate(params: {
    dataset: string;
    mode?: string;
  }): Promise<ExecuteScriptResponse> {
    return fetchApi('/api/scripts/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        dataset: params.dataset,
        mode: params.mode || 'zero_shot',
      }),
    });
  },

  /**
   * 生成可视化
   */
  async runVisualize(params: {
    dataset: string;
    mode?: string;
  }): Promise<ExecuteScriptResponse> {
    return fetchApi('/api/scripts/visualize', {
      method: 'POST',
      body: JSON.stringify({
        dataset: params.dataset,
        mode: params.mode || 'zero_shot',
      }),
    });
  },

  /**
   * 运行完整流程
   */
  async runFullPipeline(params: {
    dataset: string;
    mode?: string;
    num_topics?: number;
  }): Promise<ExecuteScriptResponse> {
    return fetchApi('/api/scripts/pipeline', {
      method: 'POST',
      body: JSON.stringify({
        dataset: params.dataset,
        mode: params.mode || 'zero_shot',
        num_topics: params.num_topics || 20,
      }),
    });
  },
};

export default ETMAgentAPI;
