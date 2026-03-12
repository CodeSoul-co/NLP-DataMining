#!/usr/bin/env python3
"""
重置管理员密码
"""

import asyncio
import os
import sys
from pathlib import Path

# 确保使用正确的模式
os.environ.setdefault("SIMULATION_MODE", "false")

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ETM"))
sys.path.insert(0, str(project_root / "langgraph_agent" / "backend"))

os.chdir(project_root / "langgraph_agent" / "backend")


async def main():
    from app.models.user import User
    from app.models import task, dataset, project, user_dataset  # Import all models
    from app.core.database import async_session_maker
    from sqlalchemy import select, update

    # 新密码
    new_password = os.environ.get("ADMIN_PASSWORD", "Admin123!@#")
    
    async with async_session_maker() as session:
        # 获取admin用户
        result = await session.execute(
            select(User).where(User.username == "admin")
        )
        user = result.scalar_one_or_none()
        
        if not user:
            print("❌ 未找到admin用户")
            return
        
        # 重新生成密码哈希
        new_hash = User.get_password_hash(new_password)
        
        # 更新密码
        await session.execute(
            update(User)
            .where(User.username == "admin")
            .values(hashed_password=new_hash)
        )
        await session.commit()
        
        print(f"✅ 管理员密码已重置")
        print(f"   用户名: admin")
        print(f"   新密码: {new_password}")
        print("\n现在可以使用这些凭据登录了！")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
