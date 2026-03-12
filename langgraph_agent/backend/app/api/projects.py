"""
Projects API - 用户项目 CRUD，持久化到数据库
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..models.project import Project
from ..models.user import User
from ..services.auth_service import get_current_active_user
from ..core.database import async_session_maker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_name: Optional[str] = None
    mode: str = "zero_shot"
    num_topics: int = 20


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    dataset_name: Optional[str] = None
    mode: Optional[str] = None
    num_topics: Optional[int] = None
    status: Optional[str] = None
    pipeline_status: Optional[str] = None
    task_id: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    dataset_name: Optional[str]
    mode: str
    num_topics: int
    status: str
    pipeline_status: Optional[str]
    task_id: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]

    class Config:
        from_attributes = True


@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_active_user),
):
    """获取当前用户的所有项目"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(Project)
            .where(Project.user_id == current_user.id)
            .order_by(Project.updated_at.desc())
        )
        projects = result.scalars().all()
        return [
            ProjectResponse(
                id=p.id,
                name=p.name,
                dataset_name=p.dataset_name,
                mode=p.mode or "zero_shot",
                num_topics=p.num_topics or 20,
                status=p.status or "draft",
                pipeline_status=p.pipeline_status,
                task_id=p.task_id,
                created_at=p.created_at.isoformat() if p.created_at else None,
                updated_at=p.updated_at.isoformat() if p.updated_at else None,
            )
            for p in projects
        ]


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
):
    """创建新项目"""
    async with async_session_maker() as session:
        project = Project(
            user_id=current_user.id,
            name=data.name,
            dataset_name=data.dataset_name,
            mode=data.mode,
            num_topics=data.num_topics,
            status="draft",
        )
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return ProjectResponse(
            id=project.id,
            name=project.name,
            dataset_name=project.dataset_name,
            mode=project.mode or "zero_shot",
            num_topics=project.num_topics or 20,
            status=project.status or "draft",
            pipeline_status=project.pipeline_status,
            task_id=project.task_id,
            created_at=project.created_at.isoformat() if project.created_at else None,
            updated_at=project.updated_at.isoformat() if project.updated_at else None,
        )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
):
    """获取单个项目"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == current_user.id,
            )
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="项目不存在")
        return ProjectResponse(
            id=project.id,
            name=project.name,
            dataset_name=project.dataset_name,
            mode=project.mode or "zero_shot",
            num_topics=project.num_topics or 20,
            status=project.status or "draft",
            pipeline_status=project.pipeline_status,
            task_id=project.task_id,
            created_at=project.created_at.isoformat() if project.created_at else None,
            updated_at=project.updated_at.isoformat() if project.updated_at else None,
        )


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    data: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
):
    """更新项目"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == current_user.id,
            )
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="项目不存在")

        updates = data.model_dump(exclude_unset=True)
        for k, v in updates.items():
            setattr(project, k, v)

        await session.commit()
        await session.refresh(project)
        return ProjectResponse(
            id=project.id,
            name=project.name,
            dataset_name=project.dataset_name,
            mode=project.mode or "zero_shot",
            num_topics=project.num_topics or 20,
            status=project.status or "draft",
            pipeline_status=project.pipeline_status,
            task_id=project.task_id,
            created_at=project.created_at.isoformat() if project.created_at else None,
            updated_at=project.updated_at.isoformat() if project.updated_at else None,
        )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
):
    """删除项目（仅删除数据库记录，不删除数据集文件）"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == current_user.id,
            )
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="项目不存在")
        await session.delete(project)
        await session.commit()
