#!/usr/bin/env python3
"""
PostgreSQL 初始化脚本：创建表结构 + 首个管理员用户
使用 .env 中的 DATABASE_URL，需设置 SIMULATION_MODE=false
"""

import asyncio
import os
import sys
from pathlib import Path

# 确保使用 PostgreSQL（不启用模拟模式）
os.environ.setdefault("SIMULATION_MODE", "false")

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ETM"))
sys.path.insert(0, str(project_root / "langgraph_agent" / "backend"))

# 必须在导入 app 之前设置好环境变量
os.chdir(project_root / "langgraph_agent" / "backend")


async def main():
    from app.core.database import init_db
    from app.models.user import user_db

    print("正在连接 PostgreSQL 并初始化表结构...")
    await init_db()
    print("✅ 表结构创建成功")

    # 默认管理员（可修改）
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@code-soul.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "Admin123!@#")

    existing = await user_db.get_user_by_username(admin_username)
    if existing:
        print(f"⚠️  用户 '{admin_username}' 已存在，跳过创建")
        return

    from sqlalchemy import update
    from app.models.user import User
    from app.core.database import async_session_maker

    user = await user_db.create_user(
        username=admin_username,
        email=admin_email,
        password=admin_password,
        full_name="管理员",
    )
    # 设置为超级管理员
    async with async_session_maker() as session:
        await session.execute(update(User).where(User.id == user.id).values(is_superuser=True))
        await session.commit()

    print(f"✅ 管理员用户创建成功: {admin_username}")
    print(f"   邮箱: {admin_email}")
    print(f"   密码: {admin_password}")
    print("\n⚠️  请登录后立即修改密码！")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
