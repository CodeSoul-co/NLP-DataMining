# 测试账号说明

## 📝 默认测试账号

项目**没有预设的测试账号**，需要先注册账号或运行初始化脚本创建测试用户。

---

## 🚀 方法一：运行初始化脚本（推荐）

### 在本地开发环境

```bash
# 进入后端目录
cd langgraph_agent/backend

# 运行初始化脚本
python scripts/create_test_user.py
```

### 在 Docker 容器中

```bash
# 进入后端容器
docker exec -it etm-agent-api bash

# 运行脚本
python scripts/create_test_user.py
```

### 脚本会创建以下测试账号：

| 用户名 | 密码 | 邮箱 | 说明 |
|--------|------|------|------|
| `admin` | `admin123` | admin@theta.test | 管理员账号 |
| `test` | `test123` | test@theta.test | 测试用户 |
| `demo` | `demo123` | demo@theta.test | 演示用户 |

---

## 🎯 方法二：通过前端注册

1. 访问前端页面：`http://localhost:3000`（或你的域名）
2. 点击 **注册** 按钮
3. 填写注册信息：
   - 用户名：至少 3 个字符
   - 邮箱：有效的邮箱地址
   - 密码：至少 6 个字符
   - 全名（可选）

---

## 🔧 方法三：通过 API 注册

### 使用 curl

```bash
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "test123",
    "full_name": "测试用户"
  }'
```

### 使用 Python

```python
import requests

url = "http://localhost:8000/api/auth/register"
data = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "test123",
    "full_name": "测试用户"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 🔐 登录

创建账号后，使用以下方式登录：

### 前端登录

1. 访问登录页面：`http://localhost:3000/login`
2. 输入用户名和密码
3. 点击 **登录**

### API 登录

```bash
curl -X POST "http://localhost:8000/api/auth/login-json" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "test123"
  }'
```

返回的 `access_token` 用于后续 API 请求的认证。

---

## ⚠️ 安全提示

1. **测试账号仅用于开发/测试环境**
2. **生产环境必须删除或修改默认测试账号**
3. **使用强密码策略**
4. **定期更换密码**

---

## 🗑️ 删除测试账号

如果需要删除测试账号，可以通过数据库操作：

```bash
# 进入后端容器
docker exec -it etm-agent-api bash

# 使用 Python 删除用户
python -c "
import asyncio
from app.models.user import user_db

async def delete_user():
    await user_db.initialize()
    # 这里需要实现删除方法，或直接操作数据库
    pass

asyncio.run(delete_user())
"
```

或直接操作 SQLite 数据库：

```bash
# 数据库位置
# langgraph_agent/backend/data/users.db

sqlite3 langgraph_agent/backend/data/users.db
DELETE FROM users WHERE username = 'test';
```

---

## 📚 相关文档

- 认证 API 文档：`langgraph_agent/backend/BACKEND_STRUCTURE.md`
- 用户模型：`langgraph_agent/backend/app/models/user.py`
- 认证服务：`langgraph_agent/backend/app/services/auth_service.py`
