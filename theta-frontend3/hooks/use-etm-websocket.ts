'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

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

// 从环境变量获取 API URL，如果没有则使用默认值
const getApiBaseUrl = (): string => {
  if (typeof window === 'undefined') return 'http://localhost:8000';
  return process.env.NEXT_PUBLIC_ETM_AGENT_API_URL || 'http://localhost:8000';
};

export function useETMWebSocket(url: string = '/api/ws'): UseETMWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    // 获取 API 基础 URL
    const apiBaseUrl = getApiBaseUrl();
    
    // 将 HTTP/HTTPS URL 转换为 WebSocket URL
    let wsUrl: string;
    if (apiBaseUrl.startsWith('http://')) {
      wsUrl = apiBaseUrl.replace('http://', 'ws://') + url;
    } else if (apiBaseUrl.startsWith('https://')) {
      wsUrl = apiBaseUrl.replace('https://', 'wss://') + url;
    } else if (apiBaseUrl.startsWith('ws://') || apiBaseUrl.startsWith('wss://')) {
      wsUrl = apiBaseUrl + url;
    } else {
      // 默认使用 ws://localhost:8000
      wsUrl = `ws://localhost:8000${url}`;
    }
    
    const finalUrl = wsUrl;

    try {
      const ws = new WebSocket(finalUrl);

      ws.onopen = () => {
        console.log('ETM WebSocket connected');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        console.log('ETM WebSocket disconnected');
        setIsConnected(false);
        wsRef.current = null;

        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('ETM WebSocket error:', error);
        console.error('WebSocket URL:', finalUrl);
        // 提供更详细的错误信息
        if (error instanceof ErrorEvent) {
          console.error('Error details:', error.message);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create ETM WebSocket:', error);
    }
  }, [url]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('ETM WebSocket not connected');
    }
  }, []);

  const subscribe = useCallback(
    (taskId: string) => {
      sendMessage({ type: 'subscribe', task_id: taskId });
    },
    [sendMessage]
  );

  return {
    isConnected,
    lastMessage,
    sendMessage,
    subscribe,
  };
}
