'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { AuthAPI, User, ProfileUpdateRequest, PasswordChangeRequest } from '@/lib/api/auth';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string, rememberMe?: boolean) => Promise<void>;
  register: (username: string, email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
  updateProfile: (data: ProfileUpdateRequest) => Promise<void>;
  changePassword: (data: PasswordChangeRequest) => Promise<void>;
  refreshUser: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const refreshUser = useCallback(async () => {
    try {
      const userData = await AuthAPI.getCurrentUser();
      setUser(userData);
      // Update stored user
      const token = AuthAPI.getToken();
      if (token) {
        AuthAPI.setAuth(token, userData);
      }
    } catch (error: any) {
      // 静默处理网络错误，避免在应用启动时显示错误
      // 只有在明确是认证错误时才记录
      if (error.message && !error.message.includes('无法连接到服务器')) {
        console.error('Failed to refresh user:', error);
      }
      // 如果是网络错误，保持当前用户状态（如果有缓存的用户信息）
    }
  }, []);

  useEffect(() => {
    // Check if user is already logged in
    const checkAuth = async () => {
      if (AuthAPI.isAuthenticated()) {
        try {
          const storedUser = AuthAPI.getStoredUser();
          if (storedUser) {
            setUser(storedUser);
            // Optionally refresh user data in background
            refreshUser().catch(() => {});
          } else {
            // Try to fetch user info from API
            const userData = await AuthAPI.getCurrentUser();
            setUser(userData);
            AuthAPI.setAuth(AuthAPI.getToken()!, userData);
          }
        } catch (error: any) {
          // 如果是网络错误，不清理认证状态（可能是临时网络问题）
          if (error.message && error.message.includes('无法连接到服务器')) {
            console.warn('Network error during auth check, keeping cached state');
          } else {
            // 其他错误（如认证失败），清理状态
            console.error('Auth check failed:', error);
            AuthAPI.logout();
          }
        }
      }
      setLoading(false);
    };

    checkAuth();
  }, [refreshUser]);

  const login = async (username: string, password: string, rememberMe: boolean = false) => {
    try {
      console.log('[Auth] 开始登录请求...');
      const tokenResponse = await AuthAPI.login({ username, password });
      
      console.log('[Auth] 登录响应:', { 
        hasToken: !!tokenResponse.access_token, 
        hasUser: !!tokenResponse.user 
      });
      
      if (!tokenResponse.access_token) {
        throw new Error('登录响应中没有 token');
      }
      
      // 使用登录响应中的用户信息（如果存在），否则获取
      let userData: User;
      if (tokenResponse.user) {
        userData = tokenResponse.user;
        console.log('[Auth] 使用响应中的用户信息');
      } else {
        console.log('[Auth] 从 API 获取用户信息...');
        userData = await AuthAPI.getCurrentUser();
      }
      
      console.log('[Auth] 用户信息:', { 
        id: userData.id, 
        username: userData.username 
      });
      
      // 如果选择了"记住我"，延长 token 有效期（实际应该在后端实现，这里只是前端标记）
      if (rememberMe) {
        localStorage.setItem('remember_me', 'true');
      } else {
        localStorage.removeItem('remember_me');
      }
      
      // 保存 token 和用户信息
      AuthAPI.setAuth(tokenResponse.access_token, userData);
      console.log('[Auth] Token 和用户信息已保存到 localStorage');
      
      // 更新状态
      setUser(userData);
      console.log('[Auth] 用户状态已更新，isAuthenticated:', !!userData);
      
    } catch (error) {
      // 用户名/密码错误是预期行为，不打印到控制台；仅记录意外错误
      const msg = error instanceof Error ? error.message : String(error);
      const isCredentialError = /用户名|密码|401|Incorrect|Unauthorized/i.test(msg);
      if (!isCredentialError) {
        console.error('[Auth] 登录失败:', error);
      }
      AuthAPI.logout();
      setUser(null);
      throw error;
    }
  };

  const register = async (_username: string, _email: string, _password: string, _fullName?: string) => {
    throw new Error('暂不支持注册，请联系管理员获取账号');
  };

  const logout = useCallback(() => {
    AuthAPI.logout();
    localStorage.removeItem('remember_me');
    setUser(null);
    router.push('/');
  }, [router]);

  const updateProfile = async (data: ProfileUpdateRequest) => {
    const updatedUser = await AuthAPI.updateProfile(data);
    setUser(updatedUser);
    // Update stored user
    const token = AuthAPI.getToken();
    if (token) {
      AuthAPI.setAuth(token, updatedUser);
    }
  };

  const changePassword = async (data: PasswordChangeRequest) => {
    await AuthAPI.changePassword(data);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        logout,
        updateProfile,
        changePassword,
        refreshUser,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
