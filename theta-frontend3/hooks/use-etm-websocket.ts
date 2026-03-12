'use client';

import { useState, useCallback } from 'react';

/**
 * WebSocket hook — 当前后端不提供 WS 端点，此 hook 为 no-op 桩。
 * 任务进度跟踪请使用 ETMAgentAPI.pollTaskUntilDone() 替代。
 */

interface WebSocketMessage {
  type: string;
  task_id?: string;
  step?: string;
  status?: string;
  message?: string;
  progress?: number;
  [key: string]: unknown;
}

interface UseETMWebSocketReturn {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: Record<string, unknown>) => void;
  subscribe: (taskId: string) => void;
}

export function useETMWebSocket(_url?: string): UseETMWebSocketReturn {
  const [lastMessage] = useState<WebSocketMessage | null>(null);

  const sendMessage = useCallback((_message: Record<string, unknown>) => {
    // no-op: 后端无 WebSocket 端点
  }, []);

  const subscribe = useCallback((_taskId: string) => {
    // no-op
  }, []);

  return {
    isConnected: false,
    lastMessage,
    sendMessage,
    subscribe,
  };
}
