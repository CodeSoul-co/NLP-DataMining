'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/auth-context';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, AlertCircle, Eye, EyeOff, BrainCircuit } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import Link from 'next/link';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, isAuthenticated, loading: authLoading } = useAuth();
  const router = useRouter();

  // 从 localStorage 加载记住的用户名
  useEffect(() => {
    const rememberedUsername = localStorage.getItem('remembered_username');
    if (rememberedUsername) {
      setUsername(rememberedUsername);
      setRememberMe(true);
    }
  }, []);

  // 等待 auth 状态加载完成后再检查是否已登录
  // 登录成功后，isAuthenticated 会变为 true，自动跳转到首页
  // 只有在没有 loading 状态且确实已认证时才跳转
  useEffect(() => {
    if (!authLoading && !loading && isAuthenticated) {
      // 使用 replace 而不是 push，避免在历史记录中留下登录页
      router.replace('/');
    }
  }, [authLoading, loading, isAuthenticated, router]);
  
  // 当认证状态改变时，如果登录成功，停止加载状态
  useEffect(() => {
    if (isAuthenticated && loading) {
      setLoading(false);
    }
  }, [isAuthenticated, loading]);

  // 显示加载状态
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-slate-100">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-500">加载中...</p>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // 验证输入
    if (!username.trim() || !password.trim()) {
      setError('请输入用户名和密码');
      return;
    }
    
    setError('');
    setLoading(true);

    try {
      console.log('开始登录...', { username });
      
      // 调用登录 API
      await login(username, password, rememberMe);
      
      console.log('登录成功，等待状态更新...');
      
      // 检查 token 是否已保存
      const token = localStorage.getItem('access_token');
      if (!token) {
        console.error('登录成功但 token 未保存');
        setError('登录失败：token 未保存，请重试');
        setLoading(false);
        return;
      }
      
      console.log('Token 已保存，等待认证状态更新...');
      
      // 如果选择了"记住我"，保存用户名
      if (rememberMe) {
        localStorage.setItem('remembered_username', username);
      } else {
        localStorage.removeItem('remembered_username');
      }
      
      // 给一点时间让状态更新，然后检查是否已认证
      setTimeout(() => {
        const isAuth = localStorage.getItem('access_token');
        if (isAuth) {
          console.log('Token 存在，准备跳转...');
          // 手动触发跳转，确保状态已更新
          router.replace('/');
        } else {
          console.error('Token 在延迟检查时不存在');
          setError('登录状态未正确保存，请重试');
          setLoading(false);
        }
      }, 100);
      
    } catch (err: unknown) {
      console.error('登录失败:', err);
      
      let errorMessage = '登录失败，请检查用户名和密码';
      if (err instanceof Error) {
        errorMessage = err.message || errorMessage;
        // 如果是网络错误，提供更友好的提示
        if (errorMessage.includes('fetch') || errorMessage.includes('network') || errorMessage.includes('Failed to fetch')) {
          errorMessage = '无法连接到服务器。请检查后端是否在本地运行（如 ./start.sh）。';
        } else if (errorMessage.includes('404')) {
          errorMessage = '登录接口不存在，请检查后端服务';
        } else if (errorMessage.includes('401') || errorMessage.includes('Unauthorized') || errorMessage.includes('用户名或密码错误')) {
          errorMessage = '用户名或密码错误';
        } else if (errorMessage.includes('detail')) {
          // 尝试提取后端返回的详细错误信息
          try {
            const match = errorMessage.match(/detail[":\s]+([^"]+)/i);
            if (match) {
              errorMessage = match[1];
            }
          } catch {
            // 保持原错误信息
          }
        }
      }
      
      // 确保在设置错误后停止加载状态
      setError(errorMessage);
      setLoading(false);
      
      // 确保不会因为错误而跳转
      // 清除任何可能残留的 token
      if (!localStorage.getItem('access_token')) {
        // 没有 token，说明登录确实失败了，保持当前页面
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-slate-50 to-indigo-50 p-4">
      <Card className="w-full max-w-md shadow-xl border-slate-200/60">
        <CardHeader className="space-y-3 pb-6">
          <div className="flex items-center justify-center mb-2">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg">
              <BrainCircuit className="w-8 h-8 text-white" />
            </div>
          </div>
          <CardTitle className="text-2xl font-bold text-center bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            THETA 智能分析平台
          </CardTitle>
          <CardDescription className="text-center text-slate-500">
            登录您的账户以继续
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <Alert variant="destructive" className="animate-in slide-in-from-top-2">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="username" className="text-slate-700">用户名或邮箱</Label>
              <Input
                id="username"
                type="text"
                placeholder="请输入用户名或邮箱"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                disabled={loading}
                className="h-11 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors"
                autoComplete="username"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-slate-700">密码</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="请输入密码"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={loading}
                  className="h-11 bg-slate-50/50 border-slate-200 focus:bg-white transition-colors pr-10"
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                  tabIndex={-1}
                >
                  {showPassword ? (
                    <EyeOff className="w-5 h-5" />
                  ) : (
                    <Eye className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="remember-me"
                  checked={rememberMe}
                  onCheckedChange={(checked) => setRememberMe(checked === true)}
                  disabled={loading}
                />
                <Label
                  htmlFor="remember-me"
                  className="text-sm font-normal cursor-pointer text-slate-600"
                >
                  记住我
                </Label>
              </div>
              {/* <Link 
                href="/forgot-password" 
                className="text-sm text-blue-600 hover:text-blue-700 hover:underline"
              >
                忘记密码？
              </Link> */}
            </div>

            <Button 
              type="submit" 
              className="w-full h-11 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition-all"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  登录中...
                </>
              ) : (
                '登录'
              )}
            </Button>

            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-200"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-white px-2 text-slate-400">或</span>
              </div>
            </div>

            <div className="text-center text-sm text-slate-600">
              还没有账户？{' '}
              <Link href="/register" className="text-blue-600 hover:text-blue-700 font-medium hover:underline">
                立即注册
              </Link>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
