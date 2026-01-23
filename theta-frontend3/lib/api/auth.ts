/**
 * Authentication API Client
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  created_at: string;
  is_active: boolean;
}

export interface Token {
  access_token: string;
  token_type: string;
  expires_in: number;
  user?: User;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface ProfileUpdateRequest {
  email?: string;
  full_name?: string;
}

export interface PasswordChangeRequest {
  current_password: string;
  new_password: string;
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
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
      throw new Error('无法连接到服务器。请检查后端是否在本地运行（如 ./start.sh）及网络连接。');
    }
    throw error;
  }

  if (!response.ok) {
    if (response.status === 401) {
      // Token expired or invalid
      // 只有在非登录/注册接口时才跳转到登录页
      // 登录接口返回 401 是正常的（用户名密码错误），不应该跳转
      const isAuthEndpoint = endpoint.includes('/auth/login') || endpoint.includes('/auth/register');
      if (!isAuthEndpoint) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('user');
        // 使用 setTimeout 避免在错误处理过程中跳转
        setTimeout(() => {
          if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
            window.location.href = '/login';
          }
        }, 0);
      }
      const error = await response.json().catch(() => ({ detail: 'Unauthorized' }));
      throw new Error(error.detail || 'Unauthorized');
    }
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const AuthAPI = {
  /**
   * Register a new user
   */
  async register(data: RegisterRequest): Promise<User> {
    return fetchApi<User>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  /**
   * Login and get access token
   */
  async login(data: LoginRequest): Promise<Token> {
    return fetchApi<Token>('/api/auth/login-json', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  /**
   * Get current user information
   */
  async getCurrentUser(): Promise<User> {
    return fetchApi<User>('/api/auth/me');
  },

  /**
   * Verify token
   */
  async verifyToken(): Promise<{ valid: boolean; username: string; user_id: number }> {
    return fetchApi('/api/auth/verify');
  },

  /**
   * Logout (clear local storage)
   */
  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
  },

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  },

  /**
   * Get stored token
   */
  getToken(): string | null {
    return localStorage.getItem('access_token');
  },

  /**
   * Store token and user info
   */
  setAuth(token: string, user: User): void {
    localStorage.setItem('access_token', token);
    localStorage.setItem('user', JSON.stringify(user));
  },

  /**
   * Get stored user info
   */
  getStoredUser(): User | null {
    const userStr = localStorage.getItem('user');
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    }
    return null;
  },

  /**
   * Update user profile
   */
  async updateProfile(data: ProfileUpdateRequest): Promise<User> {
    return fetchApi<User>('/api/auth/profile', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  /**
   * Change password
   */
  async changePassword(data: PasswordChangeRequest): Promise<{ message: string }> {
    return fetchApi<{ message: string }>('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
};

export default AuthAPI;
