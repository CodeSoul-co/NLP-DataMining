"""
THETA API 认证路由
提供登录、登出、Token 验证等端点
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime

from .auth import (
    LoginRequest, LoginResponse, UserInfo, TokenInfo,
    authenticate_user, create_access_token, verify_token,
    get_current_user, active_tokens
)


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    用户登录
    
    测试账号：
    - 用户名: admin
    - 密码: admin123
    
    返回 access_token，后续请求需要在 Header 中携带：
    Authorization: Bearer {access_token}
    """
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误"
        )
    
    token, expires_at = create_access_token(user["username"], user["role"])
    expires_in = int((expires_at - datetime.now()).total_seconds())
    
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserInfo(
            username=user["username"],
            role=user["role"],
            created_at=user["created_at"]
        )
    )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    用户登出
    
    使当前 Token 失效
    """
    # 找到并删除当前用户的 Token
    tokens_to_remove = [
        token for token, info in active_tokens.items()
        if info["username"] == current_user["username"]
    ]
    for token in tokens_to_remove:
        del active_tokens[token]
    
    return {"message": "登出成功"}


@router.get("/me", response_model=TokenInfo)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    获取当前用户信息
    
    用于验证 Token 是否有效
    """
    return TokenInfo(
        username=current_user["username"],
        role=current_user["role"],
        expires_at=current_user["expires_at"]
    )


@router.post("/verify")
async def verify_token_endpoint(token: str):
    """
    验证 Token 是否有效
    
    前端可用此接口检查 Token 状态
    """
    token_info = verify_token(token)
    if not token_info:
        return {"valid": False, "message": "Token 无效或已过期"}
    
    return {
        "valid": True,
        "username": token_info["username"],
        "role": token_info["role"],
        "expires_at": token_info["expires_at"]
    }
