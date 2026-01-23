'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Zap, Play, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { WorkspaceLayout } from '@/components/workspace-layout';
import { useETMWebSocket } from '@/hooks/use-etm-websocket';
import { ETMAgentAPI, TaskResponse, CreateTaskRequest } from '@/lib/api/etm-agent';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  action?: string;
  data?: Record<string, unknown>;
}

function TrainingContent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [tasks, setTasks] = useState<TaskResponse[]>([]);
  const [selectedTask, setSelectedTask] = useState<TaskResponse | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const { lastMessage, sendMessage: wsSend, subscribe } = useETMWebSocket();

  useEffect(() => {
    loadTasks();
  }, []);

  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'step_update') {
        const systemMsg: Message = {
          id: `sys-${Date.now()}`,
          role: 'system',
          content: `**${lastMessage.step}**: ${lastMessage.message}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, systemMsg]);

        if (lastMessage.task_id) {
          updateTaskStatus(lastMessage.task_id);
        }
      } else if (lastMessage.type === 'task_update') {
        updateTaskStatus(lastMessage.task_id as string);
      }
    }
  }, [lastMessage]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadTasks = async () => {
    try {
      const data = await ETMAgentAPI.getTasks();
      setTasks(data);
    } catch (error) {
      console.error('Failed to load tasks:', error);
      const errorMessage = error instanceof Error ? error.message : '未知错误';
      if (errorMessage.includes('无法连接到 ETM Agent API')) {
        const systemMsg: Message = {
          id: `error-${Date.now()}`,
          role: 'system',
          content: `⚠️ ${errorMessage}\n\n请确保 ETM Agent API 服务正在运行在 ${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, systemMsg]);
      }
    }
  };

  const updateTaskStatus = async (taskId: string) => {
    try {
      const task = await ETMAgentAPI.getTask(taskId);
      setTasks((prev) => prev.map((t) => (t.task_id === taskId ? task : t)));
      if (selectedTask?.task_id === taskId) {
        setSelectedTask(task);
      }
    } catch (error) {
      console.error('Failed to update task status:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await ETMAgentAPI.chat(input);

      if (response.action === 'start_task' && response.data) {
        const taskRequest = response.data as CreateTaskRequest;
        const task = await ETMAgentAPI.createTask(taskRequest);

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: `任务已创建！任务 ID: ${task.task_id}\n状态: ${task.status}\n进度: ${task.progress}%`,
          timestamp: new Date(),
          action: 'task_created',
          data: { task_id: task.task_id },
        };

        setMessages((prev) => [...prev, assistantMessage]);
        setTasks((prev) => [task, ...prev]);
        setSelectedTask(task);
        subscribe(task.task_id);
      } else {
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.message,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
    } catch (error: any) {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `错误: ${error.message || '处理请求失败'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelTask = async (taskId: string) => {
    try {
      await ETMAgentAPI.cancelTask(taskId);
      await loadTasks();
      if (selectedTask?.task_id === taskId) {
        setSelectedTask(null);
      }
    } catch (error) {
      console.error('Failed to cancel task:', error);
    }
  };

  return (
    <div className="flex h-full">
      {/* 左侧主内容区 */}
      <div className="flex-1 flex flex-col p-8">
        <div className="mb-6">
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">模型训练</h2>
          <p className="text-slate-600">创建和管理 ETM 主题模型训练任务</p>
        </div>

        {/* 快捷操作 */}
        {messages.length === 0 && (
          <div className="mb-6">
            <Card className="p-6 bg-white border border-slate-200">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center flex-shrink-0">
                  <Zap className="w-6 h-6 text-blue-600" />
                </div>
                <div className="flex-1">
                  <h3 className="font-medium text-slate-900 mb-2">快速开始</h3>
                  <p className="text-sm text-slate-500 mb-4">选择一个示例命令开始训练</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      '训练 socialTwitter 数据集',
                      '使用 zero_shot 模式训练，20 个主题',
                      '训练 hatespeech，supervised 模式',
                    ].map((example) => (
                      <button
                        key={example}
                        onClick={() => setInput(example)}
                        className="px-4 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 transition-colors"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* 任务列表 */}
        <div className="flex-1 overflow-auto">
          <h3 className="font-medium text-slate-900 mb-4">任务列表</h3>
          {tasks.length === 0 ? (
            <Card className="p-8 bg-white border border-slate-200 text-center">
              <p className="text-slate-500">暂无任务，使用右侧 AI 助手创建新任务</p>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {tasks.map((task) => (
                <Card
                  key={task.task_id}
                  className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                    selectedTask?.task_id === task.task_id
                      ? 'bg-blue-50 border-blue-200'
                      : 'bg-white border-slate-200'
                  }`}
                  onClick={() => setSelectedTask(task)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-slate-900 truncate">
                        {task.dataset}
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        模式: {task.mode || 'zero_shot'} · 主题数: {task.num_topics || 20}
                      </p>
                    </div>
                    {task.status !== 'completed' && task.status !== 'failed' && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCancelTask(task.task_id);
                        }}
                        className="h-6 w-6 p-0 hover:bg-red-100 hover:text-red-600"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                  <Progress value={task.progress} className="h-2 mb-2" />
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-500">{task.progress}%</span>
                    <span className={`px-2 py-0.5 rounded ${
                      task.status === 'completed' ? 'bg-green-100 text-green-700' :
                      task.status === 'failed' ? 'bg-red-100 text-red-700' :
                      task.status === 'processing' ? 'bg-blue-100 text-blue-700' :
                      'bg-slate-100 text-slate-700'
                    }`}>
                      {task.status === 'completed' ? '已完成' :
                       task.status === 'failed' ? '失败' :
                       task.status === 'processing' ? '处理中' :
                       task.status === 'pending' ? '等待中' : task.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">
                    {new Date(task.created_at).toLocaleString()}
                  </p>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* 输入框 */}
        <div className="mt-6 pt-4 border-t border-slate-200">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="输入训练命令，例如：训练 socialTwitter 数据集，20 个主题..."
              className="flex-1"
              disabled={isLoading}
            />
            <Button type="submit" disabled={isLoading || !input.trim()}>
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default function TrainingPage() {
  return (
    <WorkspaceLayout currentStep="training">
      <TrainingContent />
    </WorkspaceLayout>
  );
}
