-- THETA 数据库初始化脚本
-- PostgreSQL

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 注意：表结构由 SQLAlchemy 自动创建
-- 此脚本用于初始化额外配置

-- 设置时区
SET timezone = 'UTC';

-- 创建索引优化查询 (SQLAlchemy 可能不会自动创建的复合索引)
-- CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user_id, status);
-- CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC);
-- CREATE INDEX IF NOT EXISTS idx_datasets_user_name ON datasets(user_id, name);

-- 添加默认管理员用户 (可选)
-- 密码: admin123 (需要用 Python 生成正确的 bcrypt hash)
-- INSERT INTO users (username, email, hashed_password, full_name, is_active, is_superuser, created_at)
-- VALUES ('admin', 'admin@theta.ai', '$2b$12$...', 'Administrator', true, true, NOW())
-- ON CONFLICT (username) DO NOTHING;

SELECT 'THETA database initialized' AS status;
