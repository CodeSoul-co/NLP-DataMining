#!/usr/bin/env python3
"""
创建测试用户脚本
用于快速创建测试账号
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "langgraph_agent" / "backend"))

from app.models.user import user_db, User
from app.core.logging import get_logger

logger = get_logger(__name__)


async def create_test_user():
    """创建测试用户"""
    # 初始化数据库
    await user_db.initialize()
    
    # 测试用户信息
    test_users = [
        {
            "username": "admin",
            "email": "admin@theta.test",
            "password": "admin123",
            "full_name": "管理员"
        },
        {
            "username": "test",
            "email": "test@theta.test",
            "password": "test123",
            "full_name": "测试用户"
        },
        {
            "username": "demo",
            "email": "demo@theta.test",
            "password": "demo123",
            "full_name": "演示用户"
        }
    ]
    
    created_count = 0
    skipped_count = 0
    
    for user_info in test_users:
        try:
            # 检查用户是否已存在
            existing_user = await user_db.get_user_by_username(user_info["username"])
            if existing_user:
                logger.warning(f"用户 '{user_info['username']}' 已存在，跳过创建")
                skipped_count += 1
                continue
            
            # 创建用户
            user = await user_db.create_user(
                username=user_info["username"],
                email=user_info["email"],
                password=user_info["password"],
                full_name=user_info["full_name"]
            )
            
            logger.info(f"✅ 成功创建测试用户: {user.username} ({user.email})")
            created_count += 1
            
        except Exception as e:
            logger.error(f"❌ 创建用户 '{user_info['username']}' 失败: {e}")
    
    print("\n" + "="*50)
    print("测试用户创建完成")
    print("="*50)
    print(f"✅ 成功创建: {created_count} 个用户")
    if skipped_count > 0:
        print(f"⚠️  跳过: {skipped_count} 个已存在的用户")
    print("\n测试账号信息:")
    print("-"*50)
    for user_info in test_users:
        print(f"用户名: {user_info['username']}")
        print(f"密码:   {user_info['password']}")
        print(f"邮箱:   {user_info['email']}")
        print("-"*50)
    print("\n⚠️  注意: 这些是测试账号，请在生产环境中修改或删除！")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(create_test_user())
