/**
 * THETA API Client — 适配 theta_1-main 后端
 *
 * 两个后端服务：
 *   主 API  (api/main.py)  → 认证、OSS 数据上传、DLC 训练任务管理
 *   Agent API (agent/api.py) → AI 分析、多轮对话、指标/主题解读、图表分析
 *
 * 前端通过 NEXT_PUBLIC_API_URL / NEXT_PUBLIC_AGENT_URL 指向它们。
 */

import { apiFetch, API_BASE, AGENT_BASE } from './config';

// ==================== 类型定义 ====================

export interface TaskResponse {
  task_id: string;
  status: 'pending_upload' | 'submitting_dlc' | 'training' | 'completed' | 'error'
    | 'pending' | 'running' | 'failed' | 'cancelled';
  current_step?: string;
  progress: number;
  message?: string;

  dataset?: string;
  mode?: string;
  num_topics?: number;

  metrics?: Record<string, number>;
  topic_words?: Record<string, string[]>;
  visualization_paths?: string[];

  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  duration_seconds?: number;

  dlc_job_id?: string;
  dlc_status?: string;
  error_message?: string;

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
  model_size?: string;
  models?: string;
}

export interface DatasetInfo {
  name: string;
  path: string;
  file_count?: number;
  total_size?: string;
  size?: number;
}

/** 数据库中的用户项目（需登录） */
export interface ProjectInfo {
  id: number;
  name: string;
  dataset_name?: string | null;
  mode: string;
  num_topics: number;
  status: string;
  pipeline_status?: string | null;
  task_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ResultInfo {
  dataset: string;
  mode: string;
  timestamp: string;
  path: string;
  num_topics?: number;
  vocab_size?: number;
  epochs_trained?: number;
  metrics?: Record<string, number>;
  has_model?: boolean;
  has_theta?: boolean;
  has_beta?: boolean;
  has_topic_words?: boolean;
  has_visualizations?: boolean;
}

export interface TopicWord { word: string; weight: number; }

export interface MetricsResponse {
  coherence?: number;
  diversity?: number;
  perplexity?: number;
  [key: string]: any;
}

export type PreprocessingJobStatus =
  | 'pending' | 'bow_generating' | 'bow_completed'
  | 'embedding_generating' | 'embedding_completed'
  | 'running' | 'completed' | 'failed';

export interface PreprocessingJob {
  job_id: string;
  dataset: string;
  model?: string;
  status: PreprocessingJobStatus;
  progress: number;
  message: string | null;
  current_stage?: string | null;
  error_message?: string | null;
  created_at?: string;
  updated_at?: string;
  bow_path?: string | null;
  embedding_path?: string | null;
  vocab_path?: string | null;
}

export interface PreprocessingStatus {
  dataset?: string;
  has_bow: boolean;
  has_embeddings: boolean;
  ready_for_training: boolean;
  bow_path?: string | null;
  embedding_path?: string | null;
  vocab_path?: string | null;
}

// ==================== OSS 直传辅助 ====================

interface PresignedUrlResponse {
  job_id: string;
  upload_url: string;
  oss_path: string;
  content_type: string;
  expires_in: number;
}

/**
 * OSS 三步直传流程：
 * 1. 后端签发上传 URL → 2. 前端 PUT 到 OSS → 3. 通知后端上传完成并触发训练
 */
async function ossUpload(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<PresignedUrlResponse> {
  onProgress?.(5);
  const presigned = await apiFetch<PresignedUrlResponse>(API_BASE, '/api/data/presigned-url', {
    method: 'POST',
    body: JSON.stringify({
      filename: file.name,
      content_type: file.type || null,
    }),
  });

  onProgress?.(10);
  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener('progress', (e) => {
      if (e.total > 0 && onProgress) {
        onProgress(10 + Math.round((e.loaded / e.total) * 80));
      }
    });
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        const detail = xhr.responseText?.match(/<Message>(.*?)<\/Message>/)?.[1] || '';
        reject(new Error(`OSS 上传失败 (HTTP ${xhr.status})${detail ? ': ' + detail : ''}`));
      }
    });
    xhr.addEventListener('error', () => reject(new Error('OSS 上传网络错误')));
    xhr.addEventListener('timeout', () => reject(new Error('OSS 上传超时')));
    xhr.open('PUT', presigned.upload_url);
    xhr.setRequestHeader('Content-Type', presigned.content_type);
    xhr.timeout = 300_000;
    xhr.send(file);
  });

  onProgress?.(95);
  return presigned;
}

// ==================== 主 API ====================

export const ETMAgentAPI = {
  // ========== 健康检查 ==========
  async healthCheck(): Promise<{ status: string; gpu_available?: boolean }> {
    return apiFetch(API_BASE, '/health');
  },

  // ========== 后端配置 ==========
  async getConfig(): Promise<{
    oss_bucket: string;
    supported_models: string[];
    supported_modes: string[];
    supported_model_sizes: string[];
    default_num_topics: number;
    default_epochs: number;
  }> {
    return apiFetch(API_BASE, '/config');
  },

  // ========== 项目管理（数据库，需登录） ==========

  async getProjects(): Promise<ProjectInfo[]> {
    try {
      return await apiFetch<ProjectInfo[]>(API_BASE, '/api/projects');
    } catch {
      return [];
    }
  },

  async createProject(data: { name: string; dataset_name?: string; mode?: string; num_topics?: number }): Promise<ProjectInfo> {
    return apiFetch<ProjectInfo>(API_BASE, '/api/projects', {
      method: 'POST',
      body: JSON.stringify({
        name: data.name,
        dataset_name: data.dataset_name,
        mode: data.mode ?? 'zero_shot',
        num_topics: data.num_topics ?? 20,
      }),
    });
  },

  async updateProject(
    id: number,
    data: Partial<{ name: string; dataset_name: string; mode: string; num_topics: number; status: string; pipeline_status: string; task_id: string }>,
  ): Promise<ProjectInfo> {
    return apiFetch<ProjectInfo>(API_BASE, `/api/projects/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  },

  /** theta_1 流程：在开始分析前将 job 中的文件落到 dataset 目录 */
  async prepareDataset(jobId: string, datasetName: string): Promise<{ status: string; dataset: string }> {
    return apiFetch(API_BASE, '/api/data/prepare-dataset', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, dataset_name: datasetName }),
    });
  },

  async deleteProject(id: number): Promise<void> {
    await apiFetch(API_BASE, `/api/projects/${id}`, { method: 'DELETE' });
  },

  // ========== 数据集管理 ==========

  async getDatasets(): Promise<DatasetInfo[]> {
    try {
      return await apiFetch<DatasetInfo[]>(API_BASE, '/api/datasets');
    } catch {
      try {
        const data = await apiFetch<{ jobs: any[] }>(API_BASE, '/api/data/jobs');
        const seen = new Set<string>();
        return (data.jobs || [])
          .filter((j: any) => {
            const name = j.dataset_name || j.filename?.replace('.csv', '') || j.job_id;
            if (seen.has(name)) return false;
            seen.add(name);
            return true;
          })
          .map((j: any) => ({
            name: j.dataset_name || j.filename?.replace('.csv', '') || j.job_id,
            path: j.oss_path || '',
            file_count: 1,
          }));
      } catch {
        return [];
      }
    }
  },

  async uploadDataset(
    files: File[],
    datasetName: string,
    onProgress?: (progress: number) => void,
  ): Promise<{
    success: boolean;
    message: string;
    dataset_name: string;
    file_count: number;
    total_size: number;
    files: string[];
    job_id: string;
  }> {
    const file = files[0];
    if (!file) throw new Error('请选择文件');

    // 优先 theta_1 流程：presigned-url → 直传 → job_id（兼容 OSS 与本地后端直传）
    try {
      const presigned = await ossUpload(file, onProgress);
      onProgress?.(100);
      return {
        success: true,
        message: '上传成功',
        dataset_name: datasetName,
        file_count: 1,
        total_size: file.size,
        files: [file.name],
        job_id: presigned.job_id,
      };
    } catch (err: any) {
      if (err.message?.includes('404') || err.message?.includes('Not Found') || err.message?.includes('401')) {
        // 回退到 FormData（/api/datasets/upload）
        const formResult = await this._uploadDatasetFormData(files, datasetName, onProgress);
        return {
          ...formResult,
          job_id: (formResult as any).job_id ?? formResult.dataset_name ?? '',
        };
      }
      throw err;
    }
  },

  async _uploadDatasetFormData(
    files: File[],
    datasetName: string,
    onProgress?: (progress: number) => void,
  ): Promise<any> {
    const formData = new FormData();
    formData.append('dataset_name', datasetName);
    files.forEach((f) => formData.append('files', f));

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.upload.addEventListener('progress', (e) => {
        if (e.total > 0 && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      });
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(new Error(`上传失败 (HTTP ${xhr.status})`));
        }
      });
      xhr.addEventListener('error', () => reject(new Error('上传网络错误')));
      xhr.open('POST', `${API_BASE}/api/datasets/upload`);
      const token = localStorage.getItem('access_token');
      if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      xhr.send(formData);
    });
  },

  async deleteDataset(name: string): Promise<{ success: boolean; message: string }> {
    try {
      return await apiFetch(API_BASE, `/api/datasets/${name}`, { method: 'DELETE' });
    } catch {
      return { success: false, message: '删除功能暂不支持' };
    }
  },

  // ========== 任务管理（job_id 体系） ==========

  async getTasks(params?: {
    status?: string; dataset?: string; limit?: number; offset?: number;
  }): Promise<TaskResponse[]> {
    try {
      const q = new URLSearchParams();
      if (params?.status) q.set('status', params.status);
      if (params?.dataset) q.set('dataset', params.dataset);
      if (params?.limit) q.set('limit', String(params.limit));
      if (params?.offset) q.set('offset', String(params.offset));
      const qs = q.toString();
      return await apiFetch(API_BASE, `/api/tasks${qs ? `?${qs}` : ''}`);
    } catch {
      const raw = await apiFetch<{ jobs: any[] }>(API_BASE, `/api/data/jobs?limit=${params?.limit ?? 20}`);
      return (raw.jobs || []).map(normalizeJob);
    }
  },

  async getTaskStats(): Promise<{
    total: number; pending: number; running: number; completed: number; failed: number; cancelled: number;
  }> {
    try {
      return await apiFetch(API_BASE, '/api/tasks/stats');
    } catch {
      const tasks = await this.getTasks({ limit: 100 });
      return {
        total: tasks.length,
        pending: tasks.filter((t) => t.status === 'pending_upload' || t.status === 'pending').length,
        running: tasks.filter((t) => t.status === 'training' || t.status === 'running' || t.status === 'submitting_dlc').length,
        completed: tasks.filter((t) => t.status === 'completed').length,
        failed: tasks.filter((t) => t.status === 'error' || t.status === 'failed').length,
        cancelled: tasks.filter((t) => t.status === 'cancelled').length,
      };
    }
  },

  async getTask(taskId: string): Promise<TaskResponse> {
    try {
      return await apiFetch(API_BASE, `/api/tasks/${taskId}`);
    } catch {
      const raw = await apiFetch<any>(API_BASE, `/api/data/jobs/${taskId}/status`);
      return normalizeJob(raw);
    }
  },

  async getTaskLogs(taskId: string, tail: number = 50): Promise<{
    task_id: string; status: string; logs: any[]; total_count: number;
  }> {
    try {
      return await apiFetch(API_BASE, `/api/tasks/${taskId}/logs?tail=${tail}`);
    } catch {
      return { task_id: taskId, status: 'unknown', logs: [], total_count: 0 };
    }
  },

  async createTask(request: CreateTaskRequest & { job_id?: string }): Promise<TaskResponse> {
    // job_id 仅用于 theta_1-main 的 OSS 流程；langgraph_agent 无 /api/data/upload-complete
    const jobIdFromOss = request.job_id && request.job_id !== request.dataset;
    if (jobIdFromOss) {
      try {
        const raw = await apiFetch<any>(API_BASE, '/api/data/upload-complete', {
          method: 'POST',
          body: JSON.stringify({
            job_id: request.job_id,
            dataset_name: request.dataset,
            num_topics: request.num_topics ?? 20,
            epochs: request.epochs ?? 100,
            mode: request.mode ?? 'zero_shot',
            model_size: request.model_size ?? '0.6B',
            models: request.models ?? 'theta',
          }),
        });
        return normalizeJob(raw);
      } catch {
        // 若 upload-complete 不存在，回退到 POST /api/tasks
      }
    }

    try {
      return await apiFetch(API_BASE, '/api/tasks', {
        method: 'POST',
        body: JSON.stringify(request),
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : '请先上传数据文件，再创建训练任务';
      throw new Error(msg);
    }
  },

  async cancelTask(taskId: string): Promise<{ message: string }> {
    try {
      return await apiFetch(API_BASE, `/api/tasks/${taskId}`, { method: 'DELETE' });
    } catch {
      return { message: '取消失败或不支持' };
    }
  },

  async pollTaskUntilDone(
    taskId: string,
    onProgress?: (task: TaskResponse) => void,
    interval = 5000,
    timeout = 3_600_000,
  ): Promise<TaskResponse> {
    const start = Date.now();
    while (true) {
      const task = await this.getTask(taskId);
      onProgress?.(task);
      if (['completed', 'failed', 'cancelled', 'error'].includes(task.status)) return task;
      if (Date.now() - start > timeout) throw new Error('轮询超时');
      await new Promise((r) => setTimeout(r, interval));
    }
  },

  // ========== 结果查询 ==========

  async getResults(): Promise<ResultInfo[]> {
    try { return await apiFetch(API_BASE, '/api/results'); } catch { return []; }
  },

  /**
   * 获取训练结果文件下载链接
   * theta_1-main: GET /api/data/jobs/{id}/results → {files: {filename: signed_url}}
   */
  async getJobResults(jobId: string): Promise<{ job_id: string; result_base: string; files: Record<string, string> }> {
    return apiFetch(API_BASE, `/api/data/jobs/${jobId}/results`);
  },

  async getTopicWords(dataset: string, mode: string, topK = 10): Promise<Record<string, string[]>> {
    try {
      return await apiFetch(API_BASE, `/api/results/${dataset}/${mode}/topic-words?top_k=${topK}`);
    } catch {
      return {};
    }
  },

  async getTopicProportions(dataset: string, mode: string): Promise<{ topics: string[]; proportions: number[] }> {
    try {
      return await apiFetch(API_BASE, `/api/results/${dataset}/${mode}/visualization-data?data_type=topic_distribution`);
    } catch {
      return { topics: [], proportions: [] };
    }
  },

  getTopicWordImportanceUrl(dataset: string, mode: string, topicIndex: number): string {
    return `${API_BASE}/api/results/${encodeURIComponent(dataset)}/${encodeURIComponent(mode)}/visualizations/topics/topic_${topicIndex}/word_importance.png`;
  },

  async getMetrics(dataset: string, mode: string): Promise<MetricsResponse> {
    try {
      return await apiFetch(API_BASE, `/api/results/${dataset}/${mode}/metrics`);
    } catch {
      return {};
    }
  },

  async getDatasetPreview(dataset: string, jobId?: string): Promise<{ columns: string[]; rows: string[][] }> {
    try {
      const qs = jobId ? `?job_id=${encodeURIComponent(jobId)}` : "";
      return await apiFetch(API_BASE, `/api/datasets/${encodeURIComponent(dataset)}/preview${qs}`);
    } catch {
      return { columns: [], rows: [] };
    }
  },

  async listVisualizations(dataset: string, mode: string): Promise<Array<{ name: string; path: string; type: string; size?: number }>> {
    try {
      return await apiFetch(API_BASE, `/api/results/${dataset}/${mode}/visualizations`);
    } catch {
      return [];
    }
  },

  async getModelComparison(dataset: string): Promise<{
    dataset: string;
    rows: Array<Record<string, unknown>>;
    columns: Array<{ key: string; label: string; direction: string }>;
  }> {
    try {
      return await apiFetch(API_BASE, `/api/results/${encodeURIComponent(dataset)}/model-comparison`);
    } catch {
      return { dataset, rows: [], columns: [] };
    }
  },

  async exportResults(
    dataset: string,
    mode: string,
    types: string[] = ['metrics', 'topic_words', 'visualizations'],
  ): Promise<void> {
    const params = new URLSearchParams({ types: types.join(',') });
    const url = `${API_BASE}/api/results/${encodeURIComponent(dataset)}/${encodeURIComponent(mode)}/export?${params}`;
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(url, { headers, credentials: 'include' });
    if (!res.ok) throw new Error(`导出失败: ${res.status}`);
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${dataset}_${mode}_results.zip`;
    a.click();
    URL.revokeObjectURL(a.href);
  },

  // ========== 预处理 ==========

  async startPreprocessing(params: { dataset: string; text_column?: string; config?: any }): Promise<PreprocessingJob> {
    try {
      return await apiFetch(API_BASE, '/api/preprocessing/start', {
        method: 'POST',
        body: JSON.stringify(params),
      });
    } catch {
      return {
        job_id: `prep_${Date.now()}`,
        dataset: params.dataset,
        status: 'completed',
        progress: 100,
        message: '预处理跳过（当前后端不支持独立预处理步骤）',
      };
    }
  },

  async getPreprocessingJob(jobId: string): Promise<PreprocessingJob> {
    try {
      return await apiFetch(API_BASE, `/api/preprocessing/${jobId}`);
    } catch {
      return {
        job_id: jobId,
        dataset: '',
        status: 'completed',
        progress: 100,
        message: null,
      };
    }
  },

  async checkPreprocessingStatus(dataset: string): Promise<PreprocessingStatus> {
    try {
      return await apiFetch(API_BASE, `/api/preprocessing/check/${dataset}`);
    } catch {
      return { has_bow: false, has_embeddings: false, ready_for_training: false };
    }
  },

  // ========== AI 对话 ==========

  async chat(
    message: string,
    context?: Record<string, unknown>,
  ): Promise<{ message: string; response?: string; action?: string; task_id?: string; data?: Record<string, unknown> }> {
    try {
      const raw = await apiFetch<any>(AGENT_BASE || API_BASE, '/api/agent/chat', {
        method: 'POST',
        body: JSON.stringify({
          message,
          job_id: context?.job_id ?? context?.dataset ?? '',
          session_id: context?.session_id ?? 'default',
        }),
      });
      return { message: raw.message, response: raw.message, action: undefined, data: undefined };
    } catch {
      try {
        return await apiFetch(API_BASE, '/api/chat', {
          method: 'POST',
          body: JSON.stringify({ message, context }),
        });
      } catch {
        return { message: '暂时无法连接 AI 服务，请稍后再试。' };
      }
    }
  },

  async *chatStream(
    message: string,
    sessionId = 'default',
  ): AsyncGenerator<{ type: string; content: string }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    const response = await fetch(`${AGENT_BASE}/api/agent/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (!response.ok) throw new Error(`SSE 请求失败 (HTTP ${response.status})`);

    const reader = response.body?.getReader();
    if (!reader) throw new Error('无法读取响应流');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          try {
            yield JSON.parse(data);
          } catch { /* skip malformed chunks */ }
        }
      }
    }
  },

  // ========== Agent 高级功能 ==========

  async getAgentTools(): Promise<{ tools: Array<{ name: string; description: string }> }> {
    return apiFetch(AGENT_BASE, '/api/agent/tools');
  },

  async clearAgentSession(sessionId = 'default'): Promise<void> {
    await apiFetch(AGENT_BASE, `/api/agent/sessions/${sessionId}`, { method: 'DELETE' });
  },

  async listAgentSessions(): Promise<{ sessions: string[] }> {
    return apiFetch(AGENT_BASE, '/api/agent/sessions');
  },

  // ========== 结果解读（Agent API） ==========

  async interpretMetrics(jobId: string, language = 'zh'): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/interpret/metrics', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, language }),
    });
  },

  async interpretTopics(jobId: string, language = 'zh', useLlm = true): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/interpret/topics', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, language, use_llm: useLlm }),
    });
  },

  async generateSummary(jobId: string, language = 'zh'): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/interpret/summary', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, language }),
    });
  },

  async analyzeChart(jobId: string, chartName: string, analysisType = 'general', language = 'zh'): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/vision/analyze-chart', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, chart_name: chartName, analysis_type: analysisType, language }),
    });
  },

  async getJobTopics(jobId: string): Promise<{ job_id: string; topics: any[] }> {
    return apiFetch(AGENT_BASE || API_BASE, `/api/jobs/${jobId}/topics`);
  },

  async getJobCharts(jobId: string): Promise<{ job_id: string; charts: any; wordclouds: string[]; downloads: any }> {
    return apiFetch(AGENT_BASE || API_BASE, `/api/jobs/${jobId}/charts`);
  },

  /**
   * 列出 OSS 上指定数据集的所有图表文件（png/jpg/pdf/html）。
   * 等同于：ossutil ls "oss://…/result/baseline/{dataset}/" -r | grep "\.png\|\.jpg\|\.pdf\|\.html"
   */
  async listOssChartFiles(dataset: string): Promise<{
    dataset: string;
    charts: Array<{ key: string; path: string; ext: string; size: number; url: string }>;
    total: number;
    note?: string;
  }> {
    try {
      return await apiFetch(API_BASE, `/api/data/oss-charts/${encodeURIComponent(dataset)}`);
    } catch {
      return { dataset, charts: [], total: 0, note: 'OSS 图表文件列举失败' };
    }
  },

  /**
   * 列出 OSS 上所有拥有可视化图表的数据集名称（用于选择器）。
   */
  async listOssDatasets(): Promise<{
    datasets: Array<{ name: string; chart_count: number }>;
    note?: string;
  }> {
    try {
      return await apiFetch(API_BASE, `/api/data/oss-datasets`);
    } catch {
      return { datasets: [], note: 'OSS 数据集列表获取失败' };
    }
  },

  getDownloadUrl(jobId: string, filename: string): string {
    return `${AGENT_BASE || API_BASE}/api/download/${jobId}/${filename}`;
  },

  // ========== 对话历史 ==========

  async saveConversationHistory(sessionId: string, messages: Array<{ role: string; content: string }>): Promise<any> {
    try {
      return await apiFetch(API_BASE, '/api/chat/history', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, messages }),
      });
    } catch { return { message: 'ok', session_id: sessionId, message_count: 0 }; }
  },

  async getConversationHistory(sessionId: string): Promise<any> {
    try {
      return await apiFetch(API_BASE, `/api/chat/history/${sessionId}`);
    } catch { return { session_id: sessionId, messages: [], count: 0 }; }
  },

  async clearConversationHistory(sessionId: string): Promise<any> {
    try {
      return await apiFetch(API_BASE, `/api/chat/history/${sessionId}`, { method: 'DELETE' });
    } catch { return { message: 'ok', session_id: sessionId }; }
  },

  // ========== 智能建议 ==========

  async getSuggestions(context?: Record<string, unknown>): Promise<any> {
    try {
      return await apiFetch(API_BASE, '/api/chat/suggestions', {
        method: 'POST',
        body: JSON.stringify(context || {}),
      });
    } catch {
      return {
        suggestions: [
          { text: '开始分析', action: 'start', description: '上传数据并运行分析流水线' },
          { text: '查看结果', action: 'results', description: '查看已有的分析结果' },
        ],
      };
    }
  },
};

// ==================== 辅助函数 ====================

// DLC 状态 → { message, progress } 映射（与阿里云控制台一致）
const DLC_STATUS_MAP: Record<string, { message: string; progress: number }> = {
  Creating:   { message: '任务创建中', progress: 5 },
  Created:    { message: '任务已创建，等待调度', progress: 8 },
  Queuing:    { message: '排队等待资源', progress: 10 },
  Waiting:    { message: '等待资源分配', progress: 12 },
  Scheduling: { message: '正在调度资源', progress: 15 },
  Preparing:  { message: '环境准备中', progress: 20 },
  Running:    { message: '训练运行中', progress: 50 },
  Stopping:   { message: '任务停止中', progress: 95 },
  Succeeded:  { message: '训练完成', progress: 100 },
  Failed:     { message: '训练失败', progress: 0 },
  Stopped:    { message: '训练已停止', progress: 0 },
};

function normalizeJob(raw: any): TaskResponse {
  const dlcStatus: string | undefined = raw.dlc_status;
  const dlcInfo = dlcStatus ? DLC_STATUS_MAP[dlcStatus] : undefined;

  let progress: number;
  if (raw.status === 'completed') progress = 100;
  else if (raw.status === 'error') progress = 0;
  else if (dlcInfo) progress = dlcInfo.progress;
  else progress = 50;

  // 计算已运行时长（仅 DLC 阶段）
  let elapsedSec = 0;
  if (raw.created_at && raw.status !== 'completed' && raw.status !== 'error') {
    elapsedSec = Math.floor((Date.now() - new Date(raw.created_at).getTime()) / 1000);
  }

  const dlcElapsed = elapsedSec > 0 ? `（已运行 ${elapsedSec} 秒）` : '';
  const message: string | undefined =
    raw.message ||
    (dlcInfo ? `${dlcInfo.message}${dlcElapsed}` : undefined) ||
    (dlcStatus ? `DLC: ${dlcStatus}${dlcElapsed}` : undefined) ||
    raw.error;

  return {
    task_id: raw.job_id ?? raw.task_id ?? '',
    status: raw.status ?? 'pending',
    progress,
    message,
    dataset: raw.dataset_name ?? raw.dataset ?? undefined,
    mode: raw.mode ?? undefined,
    num_topics: raw.num_topics ?? undefined,
    created_at: raw.created_at ?? undefined,
    completed_at: raw.completed_at ?? undefined,
    dlc_job_id: raw.dlc_job_id ?? undefined,
    dlc_status: dlcStatus,
    error_message: raw.error ?? undefined,
  };
}

export default ETMAgentAPI;
