#!/usr/bin/env python3
"""
独立创建测试用户（不启动后端，直接写 data/theta.db）
与后端 SIMULATION_MODE 使用同一库：data/theta.db，表结构与 SQLAlchemy User 一致。
"""

import hashlib
import base64
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# 项目根 = THETA
project_root = Path(__file__).resolve().parent.parent.parent.parent
db_path = project_root / "data" / "theta.db"

# 与 User 模型一致的密码哈希（需: pip install 'passlib[bcrypt]'）
try:
    from passlib.context import CryptContext
except ImportError:
    print("缺少 passlib，请先执行: pip install 'passlib[bcrypt]'")
    sys.exit(1)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prehash(p: str) -> str:
    return base64.b64encode(hashlib.sha256(p.encode()).digest()).decode("ascii")


def hash_password(p: str) -> str:
    return pwd_context.hash(_prehash(p))


def main():
    (project_root / "data").mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 与 SQLAlchemy User 表结构一致（SQLite: Boolean→0/1, DateTime→文本）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            full_name VARCHAR(100),
            is_active BOOLEAN DEFAULT 1,
            is_superuser BOOLEAN DEFAULT 0,
            created_at DATETIME,
            updated_at DATETIME,
            last_login DATETIME
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_users_username ON users(username)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_users_email ON users(email)")
    conn.commit()

    users = [
        ("admin", "admin@theta.test", "admin123", "管理员"),
        ("test", "test@theta.test", "test123", "测试用户"),
        ("demo", "demo@theta.test", "demo123", "演示用户"),
    ]

    now = datetime.utcnow().isoformat()
    created = 0
    for username, email, password, full_name in users:
        try:
            cur.execute("SELECT 1 FROM users WHERE username=? OR email=?", (username, email))
            if cur.fetchone():
                print(f"⊘ 已存在: {username} / {email}，跳过")
                continue
            cur.execute(
                """INSERT INTO users (username, email, hashed_password, full_name, is_active, is_superuser, created_at, updated_at)
                   VALUES (?,?,?,?,1,0,?,?)""",
                (username, email, hash_password(password), full_name, now, now),
            )
            print(f"✓ 已创建: {username} 密码={password}")
            created += 1
        except Exception as e:
            print(f"✗ 创建 {username} 失败: {e}")
    conn.commit()
    conn.close()

    print("\n--- 测试账号（data/theta.db）---")
    for u, _, p, _ in users:
        print(f"  用户名: {u}  密码: {p}")
    print("--- 用以上账号登录后，请在生产环境中修改或删除 ---\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
