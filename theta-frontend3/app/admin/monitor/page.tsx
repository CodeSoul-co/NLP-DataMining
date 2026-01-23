'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { RefreshCw, Server, Database, Cpu, HardDrive, Activity, AlertCircle, CheckCircle2, XCircle, Power } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface ServiceStatus {
  name: string;
  type: 'backend' | 'database' | 'script' | 'model';
  status: 'running' | 'stopped' | 'error' | 'unknown';
  endpoint?: string;
  port?: number;
  lastCheck?: string;
  details?: Record<string, any>;
}

export default function MonitorPage() {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [mounted, setMounted] = useState(false);
  const [restarting, setRestarting] = useState(false);

  // 确保时间只在客户端渲染，避免 hydration 错误
  useEffect(() => {
    setMounted(true);
    setLastUpdate(new Date());
  }, []);

  const checkServices = async () => {
    setLoading(true);
    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    const serviceChecks: ServiceStatus[] = [];

    // 1. 检查主后端服务
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5秒超时
      
      const healthRes = await fetch(`${API_BASE_URL}/api/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (healthRes.ok) {
        const health = await healthRes.json();
        serviceChecks.push({
          name: 'THETA Backend API',
          type: 'backend',
          status: 'running',
          endpoint: API_BASE_URL,
          port: 8000,
          lastCheck: new Date().toISOString(),
          details: health,
        });
      } else {
        serviceChecks.push({
          name: 'THETA Backend API',
          type: 'backend',
          status: 'error',
          endpoint: API_BASE_URL,
          port: 8000,
          lastCheck: new Date().toISOString(),
          details: { http_status: healthRes.status },
        });
      }
    } catch (error: any) {
      const isTimeout = error.name === 'AbortError';
      serviceChecks.push({
        name: 'THETA Backend API',
        type: 'backend',
        status: 'stopped',
        endpoint: API_BASE_URL,
        port: 8000,
        lastCheck: new Date().toISOString(),
        details: { 
          error: isTimeout ? '连接超时' : error.message || '连接失败',
          hint: '请确保后端在本地运行（如 ./start.sh 或 uvicorn）'
        },
      });
    }

    // 2. 检查脚本服务
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const scriptsRes = await fetch(`${API_BASE_URL}/api/scripts`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (scriptsRes.ok) {
        const scripts = await scriptsRes.json();
        serviceChecks.push({
          name: 'Scripts Service',
          type: 'script',
          status: 'running',
          endpoint: `${API_BASE_URL}/api/scripts`,
          lastCheck: new Date().toISOString(),
          details: { available_scripts: Array.isArray(scripts) ? scripts.length : 0 },
        });
      } else {
        serviceChecks.push({
          name: 'Scripts Service',
          type: 'script',
          status: 'error',
          endpoint: `${API_BASE_URL}/api/scripts`,
          lastCheck: new Date().toISOString(),
          details: { http_status: scriptsRes.status },
        });
      }
    } catch (error: any) {
      const isTimeout = error.name === 'AbortError';
      serviceChecks.push({
        name: 'Scripts Service',
        type: 'script',
        status: 'stopped',
        endpoint: `${API_BASE_URL}/api/scripts`,
        lastCheck: new Date().toISOString(),
        details: { 
          error: isTimeout ? '连接超时' : error.message || '连接失败'
        },
      });
    }

    // 3. 检查数据库（用户数据库）
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const verifyRes = await fetch(`${API_BASE_URL}/api/auth/verify`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`,
        },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (verifyRes.ok || verifyRes.status === 401) {
        // 401 也说明数据库连接正常（只是未认证）
        serviceChecks.push({
          name: 'User Database',
          type: 'database',
          status: 'running',
          endpoint: `${API_BASE_URL}/api/auth`,
          lastCheck: new Date().toISOString(),
          details: { status: verifyRes.status === 401 ? '未认证' : '正常' },
        });
      } else {
        serviceChecks.push({
          name: 'User Database',
          type: 'database',
          status: 'error',
          endpoint: `${API_BASE_URL}/api/auth`,
          lastCheck: new Date().toISOString(),
          details: { http_status: verifyRes.status },
        });
      }
    } catch (error: any) {
      const isTimeout = error.name === 'AbortError';
      serviceChecks.push({
        name: 'User Database',
        type: 'database',
        status: 'stopped',
        endpoint: `${API_BASE_URL}/api/auth`,
        lastCheck: new Date().toISOString(),
        details: { 
          error: isTimeout ? '连接超时' : error.message || '连接失败'
        },
      });
    }

    // 4. 检查模型路径（通过健康检查的详细信息）
    try {
      const healthRes = await fetch(`${API_BASE_URL}/api/health`);
      if (healthRes.ok) {
        const health = await healthRes.json();
        const modelPath = health.etm_dir_exists ? 'Available' : 'Not Found';
        serviceChecks.push({
          name: 'ETM Models',
          type: 'model',
          status: health.etm_dir_exists ? 'running' : 'error',
          lastCheck: new Date().toISOString(),
          details: {
            etm_dir: health.etm_dir_exists,
            data_dir: health.data_dir_exists,
            result_dir: health.result_dir_exists,
            gpu_available: health.gpu_available,
            gpu_count: health.gpu_count,
          },
        });
      }
    } catch (error) {
      // 如果健康检查失败，上面已经记录了
    }

    setServices(serviceChecks);
    setLastUpdate(new Date());
    setLoading(false);
  };

  const handleRestart = async () => {
    if (!confirm('确定要重启所有后端服务吗？重启期间服务将暂时不可用。')) {
      return;
    }

    setRestarting(true);
    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${API_BASE_URL}/api/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        const result = await response.json();
        alert(`✅ ${result.message || '服务正在重启中，请稍候刷新页面...'}`);
        
        // 等待几秒后自动刷新服务状态
        setTimeout(() => {
          checkServices();
          setRestarting(false);
        }, 5000);
      } else {
        const error = await response.json().catch(() => ({ detail: '重启失败' }));
        alert(`❌ ${error.detail || '重启服务失败'}`);
        setRestarting(false);
      }
    } catch (error: any) {
      console.error('Restart error:', error);
      alert(`❌ 重启服务失败: ${error.message || '网络错误'}`);
      setRestarting(false);
    }
  };

  useEffect(() => {
    checkServices();
    // 每30秒自动刷新
    const interval = setInterval(checkServices, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'stopped':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Activity className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, string> = {
      running: 'default',
      error: 'destructive',
      stopped: 'secondary',
      unknown: 'outline',
    };
    
    const labels: Record<string, string> = {
      running: '运行中',
      error: '错误',
      stopped: '已停止',
      unknown: '未知',
    };

    return (
      <Badge variant={variants[status] as any} className="ml-2">
        {labels[status] || status}
      </Badge>
    );
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'backend':
        return <Server className="w-4 h-4" />;
      case 'database':
        return <Database className="w-4 h-4" />;
      case 'script':
        return <Activity className="w-4 h-4" />;
      case 'model':
        return <Cpu className="w-4 h-4" />;
      default:
        return <HardDrive className="w-4 h-4" />;
    }
  };

  const runningCount = services.filter(s => s.status === 'running').length;
  const errorCount = services.filter(s => s.status === 'error').length;
  const stoppedCount = services.filter(s => s.status === 'stopped').length;

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">服务监控面板</h1>
          <p className="text-muted-foreground mt-2">
            实时监控所有后端服务状态
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-sm text-muted-foreground" suppressHydrationWarning>
            最后更新: {mounted && lastUpdate ? lastUpdate.toLocaleTimeString() : '--:--:--'}
          </div>
          <Button 
            onClick={handleRestart} 
            disabled={restarting || loading} 
            variant="outline"
            className="bg-orange-50 hover:bg-orange-100 border-orange-200 text-orange-700"
          >
            <Power className={`w-4 h-4 mr-2 ${restarting ? 'animate-pulse' : ''}`} />
            {restarting ? '重启中...' : '重启服务'}
          </Button>
          <Button onClick={checkServices} disabled={loading || restarting} variant="outline">
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            刷新
          </Button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">总服务数</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{services.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">运行中</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{runningCount}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">错误</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-600">{errorCount}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">已停止</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{stoppedCount}</div>
          </CardContent>
        </Card>
      </div>

      {/* 服务列表 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {services.map((service, index) => (
          <Card key={index}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getTypeIcon(service.type)}
                  <CardTitle>{service.name}</CardTitle>
                </div>
                <div className="flex items-center">
                  {getStatusIcon(service.status)}
                  {getStatusBadge(service.status)}
                </div>
              </div>
              <CardDescription>
                {service.endpoint && (
                  <div className="mt-2 text-xs font-mono">{service.endpoint}</div>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {service.port && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">端口:</span>
                    <span className="font-mono">{service.port}</span>
                  </div>
                )}
                {service.lastCheck && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">检查时间:</span>
                    <span suppressHydrationWarning>
                      {mounted ? new Date(service.lastCheck).toLocaleTimeString() : '--:--:--'}
                    </span>
                  </div>
                )}
                {service.details && (
                  <div className="mt-3 pt-3 border-t">
                    <div className="text-xs font-semibold text-muted-foreground mb-2">详细信息:</div>
                    <div className="space-y-1">
                      {Object.entries(service.details).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-xs">
                          <span className="text-muted-foreground">{key}:</span>
                          <span className="font-mono">
                            {typeof value === 'boolean' ? (value ? '✓' : '✗') : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* 架构说明 */}
      <Card>
        <CardHeader>
          <CardTitle>系统架构</CardTitle>
          <CardDescription>当前系统的服务架构说明</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">前端服务</h3>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                <li>Next.js 前端 - localhost:3000</li>
                <li>通过 NEXT_PUBLIC_API_URL 请求本地后端</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">后端服务</h3>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                <li>FastAPI 后端 - localhost:8000（langgraph_agent/backend）</li>
                <li>DataClean - localhost:8001（ETM/dataclean，可选）</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">数据存储</h3>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                <li>本地模拟: SQLite（data/theta.db）</li>
                <li>数据目录: data/，结果目录: result/</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">脚本服务</h3>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                <li>通过 /api/scripts 执行（可选）</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 连接提示 */}
      {stoppedCount > 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>检测到服务未运行</strong>
            <br />
            <div className="mt-2 space-y-2">
              <p>请确保以下步骤已完成：</p>
              <ol className="list-decimal list-inside space-y-1 text-sm">
                <li>一键启动（在项目根目录）：
                  <code className="block mt-1 p-2 bg-muted rounded text-xs">
                    ./start.sh
                  </code>
                </li>
                <li>或分步启动后端（终端 1）：
                  <code className="block mt-1 p-2 bg-muted rounded text-xs">
                    cd langgraph_agent/backend<br />
                    SIMULATION_MODE=true python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
                  </code>
                </li>
                <li>检查端口：<code className="text-xs">lsof -i :8000</code></li>
              </ol>
              <p className="text-xs text-muted-foreground mt-2">
                如果所有步骤都已完成，请点击"刷新"按钮重新检查服务状态。
              </p>
            </div>
          </AlertDescription>
        </Alert>
      )}
      
      {/* 服务全部正常时的提示 */}
      {stoppedCount === 0 && errorCount === 0 && runningCount > 0 && (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle2 className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            <strong>所有服务运行正常</strong>
            <br />
            <span className="text-sm">所有后端服务已成功连接并运行正常。</span>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
