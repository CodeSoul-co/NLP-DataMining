"""
Database Configuration
PostgreSQL (生产) / SQLite (本地模拟) 异步连接
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Base class for models
Base = declarative_base()

# 模拟模式用 SQLite，避免连不上 PostgreSQL 时卡住
if getattr(settings, "SIMULATION_MODE", False):
    _db_path = str(settings.DATA_DIR / "theta.db")
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _url = f"sqlite+aiosqlite:///{_db_path}"
    engine = create_async_engine(_url, echo=settings.DEBUG)
    logger.info(f"Database: SQLite @ {_db_path}")
else:
    _url = settings.DATABASE_URL
    engine = create_async_engine(
        _url,
        echo=settings.DEBUG,
        poolclass=NullPool,
    )

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        # Import all models to register them with Base
        from ..models import user, task, dataset  # noqa
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")


async def close_db():
    """Close database connection"""
    await engine.dispose()
    logger.info("Database connection closed")
