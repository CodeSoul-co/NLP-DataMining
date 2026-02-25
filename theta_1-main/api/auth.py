"""
THETA API 认证模块
提供简单的用户认证功能

测试账号：
- 用户名: admin
- 密码: admin123
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer
from pydantic import BaseModel


# ============ 配置 ============
# 测试账号配置（生产环境应使用数据库）
TEST_USERS = {
    "admin": {
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "created_at": "2026-02-18"
    }
}

# Token 存储（生产环境应使用 Redis）
active_tokens: Dict[str, dict] = {}

# Token 有效期（小时）
TOKEN_EXPIRE_HOURS = int(os.environ.get("TOKEN_EXPIRE_HOURS", "24"))

# ==============================

security = HTTPBasic()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


class UserInfo(BaseModel):
    """用户信息"""
    username: str
    role: str
    created_at: str


class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfo


class TokenInfo(BaseModel):
    """Token 信息"""
    username: str
    role: str
    expires_at: str


def hash_password(password: str) -> str:
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return hash_password(plain_password) == hashed_password


def generate_token() -> str:
    """生成随机 Token"""
    return secrets.token_urlsafe(32)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    验证用户凭证
    
    Args:
        username: 用户名
        password: 密码
    
    Returns:
        用户信息字典，验证失败返回 None
    """
    user = TEST_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def create_access_token(username: str, role: str) -> tuple[str, datetime]:
    """
    创建访问 Token
    
    Args:
        username: 用户名
        role: 用户角色
    
    Returns:
        (token, expires_at)
    """
    token = generate_token()
    expires_at = datetime.now() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    
    active_tokens[token] = {
        "username": username,
        "role": role,
        "expires_at": expires_at.isoformat()
    }
    
    return token, expires_at


def verify_token(token: str) -> Optional[dict]:
    """
    验证 Token
    
    Args:
        token: 访问 Token
    
    Returns:
        Token 信息，无效返回 None
    """
    token_info = active_tokens.get(token)
    if not token_info:
        return None
    
    # 检查是否过期
    expires_at = datetime.fromisoformat(token_info["expires_at"])
    if datetime.now() > expires_at:
        del active_tokens[token]
        return None
    
    return token_info


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    获取当前用户（依赖注入）
    
    用于需要认证的 API 端点
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证 Token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token_info = verify_token(token)
    if not token_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return token_info


def get_current_user_optional(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """
    获取当前用户（可选，不强制认证）
    """
    if not token:
        return None
    return verify_token(token)


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """
    要求管理员权限
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user


# ============ HTTP Basic 认证（备用） ============

def get_current_user_basic(credentials: HTTPBasicCredentials = Depends(security)) -> dict:
    """
    HTTP Basic 认证方式
    """
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Basic"}
        )
    return user
