#!/usr/bin/env python3
"""
独立创建测试用户脚本（不需要后端运行）
直接操作数据库创建用户，适用于后端未启动的情况
"""

import sqlite3
import hashlib
import base64
from datetime import datetime
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "langgraph_agent" / "backend"))

from passlib.context import CryptContext

# 密码加密上下文（与 User 模型保持一致）
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prehash_password(password: str) -> str:
    """Pre-hash password with SHA-256 before bcrypt"""
    password_bytes = password.encode('utf-8')
    sha256_hash = hashlib.sha256(password_bytes).digest()
    return base64.b64encode(sha256_hash).decode('ascii')


def get_password_hash(password: str) -> str:
    """Hash a password"""
    prehashed = _prehash_password(password)
    return pwd_context.hash(prehashed)


def create_test_users():
    """创建测试用户（直接操作数据库）"""
    # 数据库路径（与 UserDB 保持一致）
    # 默认在 langgraph_agent/backend/data/users.db
    backend_dir = Path(__file__).parent.parent
    data_dir = backend_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "users.db"
    
    print(f"数据库路径: {db_path}")
    
    # 连接数据库
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 创建表（如果不存在）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT NOT NULL,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_username ON users(username)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_email ON users(email)
    """)
    
    conn.commit()
    
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
            cursor.execute("SELECT id FROM users WHERE username = ?", (user_info["username"],))
            if cursor.fetchone():
                print(f"⚠️  用户 '{user_info['username']}' 已存在，跳过创建")
                skipped_count += 1
                continue
            
            # 检查邮箱是否已存在
            cursor.execute("SELECT id FROM users WHERE email = ?", (user_info["email"],))
            if cursor.fetchone():
                print(f"⚠️  邮箱 '{user_info['email']}' 已存在，跳过创建")
                skipped_count += 1
                continue
            
            # 生成密码哈希
            hashed_password = get_password_hash(user_info["password"])
            created_at = datetime.now().isoformat()
            
            # 插入用户
            cursor.execute("""
                INSERT INTO users (username, email, hashed_password, full_name, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_info["username"],
                user_info["email"],
                hashed_password,
                user_info["full_name"],
                created_at,
                1
            ))
            
            print(f"✅ 成功创建测试用户: {user_info['username']} ({user_info['email']})")
            created_count += 1
            
        except sqlite3.IntegrityError as e:
            print(f"❌ 创建用户 '{user_info['username']}' 失败: 用户名或邮箱已存在")
            skipped_count += 1
        except Exception as e:
            print(f"❌ 创建用户 '{user_info['username']}' 失败: {e}")
    
    conn.commit()
    conn.close()
    
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
    print(f"\n数据库位置: {db_path}")
    print("现在可以启动后端服务，使用上述账号登录了！")


if __name__ == "__main__":
    try:
        create_test_users()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("\n请先安装依赖:")
        print("  pip install passlib[bcrypt]")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
