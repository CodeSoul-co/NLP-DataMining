/**
 * API 配置 - 统一管理后端连接
 *
 * theta_1-main 后端由两个服务组成：
 *   1. 主 API (api/main.py) — 认证、OSS 上传、DLC 训练任务
 *   2. Agent API (agent/api.py) — AI 对话、分析解读、可视化
 *
 * 开发环境下两个服务通常运行在不同端口，生产环境通过 nginx 统一代理。
 */

import { toast } from 'sonner'

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL !== undefined
    ? process.env.NEXT_PUBLIC_API_URL
    : 'http://localhost:8000';

export const AGENT_BASE =
  process.env.NEXT_PUBLIC_AGENT_URL !== undefined
    ? process.env.NEXT_PUBLIC_AGENT_URL
    : API_BASE; // 默认与主 API 同域，通过 nginx 代理

function getAuthHeader(): Record<string, string> {
  if (typeof window === 'undefined') return {};
  const token = localStorage.getItem('access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** 将 FastAPI 的 detail（可能是字符串或对象数组）转为可读错误信息 */
function formatErrorDetail(detail: unknown, fallback: string): string {
  if (detail == null) return String(fallback);
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d: any) => {
      if (typeof d === 'string') return d;
      if (d?.msg) return d.msg;
      return JSON.stringify(d);
    });
    return parts.filter(Boolean).join('; ') || String(fallback);
  }
  if (typeof detail === 'object' && detail !== null && 'msg' in detail) {
    return (detail as { msg: string }).msg;
  }
  return String(fallback);
}

/**
 * 通用 fetch 封装，自动附加 auth header 并处理错误
 */
export async function apiFetch<T>(
  base: string,
  endpoint: string,
  options?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const { timeoutMs = 30_000, ...fetchOptions } = options ?? {};
  const url = `${base}${endpoint}`;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...getAuthHeader(),
    ...fetchOptions.headers,
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(url, { ...fetchOptions, headers, signal: controller.signal });
  } catch (err: any) {
    clearTimeout(timer);
    const msg = err?.name === 'AbortError'
      ? '请求超时，请检查后端服务是否已启动。'
      : '无法连接到后端服务，请检查网络和后端服务状态。';
    if (typeof window !== 'undefined') toast.error(msg);
    throw new Error(msg);
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    if (response.status === 401) {
      const hadToken = typeof window !== 'undefined' && !!localStorage.getItem('access_token');
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      if (hadToken && typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
        toast.error('登录已过期，请重新登录');
        setTimeout(() => { window.location.href = '/'; }, 0);
      }
    } else if (response.status >= 500 && typeof window !== 'undefined') {
      toast.error('服务异常，请稍后重试');
    }
    const body = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    const msg = formatErrorDetail(body.detail, `HTTP ${response.status}`);
    throw new Error(msg);
  }

  // 204 No Content 无响应体，直接返回
  if (response.status === 204) return null as T;
  return response.json();
}
