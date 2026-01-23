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
  // 统一使用 NEXT_PUBLIC_API_URL，与其他 API 客户端保持一致
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
};

export function useETMWebSocket(url: string = '/api/ws'): UseETMWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    // 如果已经有连接，先关闭
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (e) {
        // 忽略关闭错误
      }
      wsRef.current = null;
    }

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

    // 如果重连次数过多，停止重连
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.warn(`ETM WebSocket: 已达到最大重连次数 (${maxReconnectAttempts})，停止重连`);
      return;
    }

    try {
      const ws = new WebSocket(finalUrl);

      ws.onopen = () => {
        console.log('ETM WebSocket connected:', finalUrl);
        setIsConnected(true);
        reconnectAttemptsRef.current = 0; // 重置重连计数
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = (event) => {
        console.log('ETM WebSocket disconnected:', {
          code: event.code,
          reason: event.reason || 'No reason provided',
          wasClean: event.wasClean
        });
        setIsConnected(false);
        wsRef.current = null;

        // 如果不是正常关闭，尝试重连
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(3000 * reconnectAttemptsRef.current, 30000); // 指数退避，最多30秒
          console.log(`ETM WebSocket: 将在 ${delay}ms 后尝试重连 (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, delay);
        }
      };

      ws.onerror = (event) => {
        // WebSocket onerror 接收的是 Event 对象，不是 Error
        // 检查连接状态以获取更多信息
        const readyState = ws.readyState;
        const stateNames = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'];
        
        // 只在连接失败时记录详细错误（避免在每次重连时都打印）
        const isConnectionFailed = readyState === WebSocket.CONNECTING || readyState === WebSocket.CLOSED;
        
        if (isConnectionFailed && reconnectAttemptsRef.current === 0) {
          // 只在第一次失败时显示详细错误
          console.warn('ETM WebSocket connection failed:', {
            url: finalUrl,
            readyState: stateNames[readyState] || readyState,
            hint: '请确保后端服务在本地运行（如 ./start.sh）。WebSocket 将自动重连。'
          });
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create ETM WebSocket:', error);
      reconnectAttemptsRef.current += 1;
      if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, 3000);
      }
    }
  }, [url]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        try {
          wsRef.current.close(1000, 'Component unmounting');
        } catch (e) {
          // 忽略关闭错误
        }
        wsRef.current = null;
      }
      reconnectAttemptsRef.current = 0; // 重置重连计数
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
