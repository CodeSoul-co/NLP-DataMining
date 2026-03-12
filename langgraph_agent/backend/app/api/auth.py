"""
Authentication API Routes
User registration, login, and profile management
"""

from datetime import timedelta
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm

from ..schemas.auth import (
    UserRegister, UserLogin, UserResponse, UserUpdate,
    PasswordChange, Token
)
from ..models.user import user_db, User
from ..services.auth_service import (
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["auth"])
async def register(user_data: UserRegister):
    """Register a new user"""
    try:
        # Check if username or email already exists
        existing_user = await user_db.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        existing_email = await user_db.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = await user_db.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        logger.info(f"New user registered: {user.username} ({user.email})")

        # 初始化 OSS 用户目录（非阻断，失败只记 warning）
        try:
            from ..services.oss_service import init_user_oss_space
            init_user_oss_space(user.id)
        except Exception as oss_err:
            logger.warning(f"OSS user space init skipped for user {user.id}: {oss_err}")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
            is_active=user.is_active
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token, tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token"""
    user = await user_db.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    )


@router.post("/login-json", response_model=Token, tags=["auth"])
async def login_json(login_data: UserLogin):
    """Login with JSON body (alternative to form data)"""
    user = await user_db.authenticate_user(login_data.username, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username}")
    
    user_response = UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        created_at=user.created_at,
        is_active=user.is_active
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(ACCESS_TOKEN_EXPIRE_MINUTES * 60),
        user=user_response
    )


@router.get("/me", response_model=UserResponse, tags=["auth"])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
        is_active=current_user.is_active
    )


@router.get("/verify", tags=["auth"])
async def verify_token_endpoint(current_user: User = Depends(get_current_active_user)):
    """Verify if token is valid"""
    return {
        "valid": True,
        "username": current_user.username,
        "user_id": current_user.id
    }


@router.put("/profile", response_model=UserResponse, tags=["auth"])
async def update_profile(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update user profile"""
    try:
        updated_user = await user_db.update_user(
            user_id=current_user.id,
            email=update_data.email,
            full_name=update_data.full_name
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        logger.info(f"User profile updated: {updated_user.username}")
        
        return UserResponse(
            id=updated_user.id,
            username=updated_user.username,
            email=updated_user.email,
            full_name=updated_user.full_name,
            created_at=updated_user.created_at,
            is_active=updated_user.is_active
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/logout", tags=["auth"])
async def logout():
    """Logout - client should clear token (JWT 无状态，服务端无需处理)"""
    return {"message": "已登出"}


@router.post("/change-password", tags=["auth"])
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user)
):
    """Change user password"""
    success = await user_db.change_password(
        user_id=current_user.id,
        current_password=password_data.current_password,
        new_password=password_data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前密码错误"
        )
    
    logger.info(f"Password changed for user: {current_user.username}")
    
    return {"message": "密码修改成功"}
