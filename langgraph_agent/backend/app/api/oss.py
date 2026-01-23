"""
OSS 相关 API（由 theta-backend 合并）
- POST /api/upload/presign：获取 OSS 直传预签名 URL（需登录）
- GET  /api/oss/datasets：列举当前用户在 OSS 下的数据集（需登录）
仅当配置 OSS_ACCESS_KEY_ID、OSS_ACCESS_KEY_SECRET、OSS_BUCKET_NAME、OSS_ENDPOINT 时可用。
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..services.oss_service import is_oss_enabled, presign_upload, list_oss_datasets
from ..services.auth_service import get_current_active_user
from ..models.user import User
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["oss"])


class PresignRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"


class PresignResponse(BaseModel):
    upload_url: str
    object_key: str
    dataset_id: str


class OssDatasetItem(BaseModel):
    dataset_id: str
    path: str


class OssDatasetsResponse(BaseModel):
    datasets: list[OssDatasetItem]


@router.post("/upload/presign", response_model=PresignResponse)
async def get_presign_url(
    request: PresignRequest,
    current_user: User = Depends(get_current_active_user),
):
    """获取 OSS 直传预签名 URL。需配置 OSS 且已登录。"""
    if not is_oss_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OSS is not configured. Set OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET_NAME, OSS_ENDPOINT.",
        )
    result = presign_upload(
        user_id=current_user.id,
        filename=request.filename,
        content_type=request.content_type,
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate presign URL.",
        )
    return PresignResponse(**result)


@router.get("/oss/datasets", response_model=OssDatasetsResponse)
async def list_oss_datasets_for_user(
    current_user: User = Depends(get_current_active_user),
):
    """列举当前用户在 OSS 下的数据集。需配置 OSS 且已登录。"""
    if not is_oss_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OSS is not configured. Set OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET_NAME, OSS_ENDPOINT.",
        )
    items = list_oss_datasets(user_id=current_user.id)
    return OssDatasetsResponse(
        datasets=[OssDatasetItem(dataset_id=x["dataset_id"], path=x["path"]) for x in items]
    )
