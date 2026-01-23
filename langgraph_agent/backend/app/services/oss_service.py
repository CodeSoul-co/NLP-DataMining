"""
OSS 服务（由 theta-backend 合并）
阿里云 OSS：预签名上传、按用户列举数据集。仅在配置 OSS_* 时可用。
"""

import json
import uuid
from typing import Optional, List, Dict, Any

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

_bucket = None


def _get_bucket():
    """延迟初始化 OSS Bucket，仅在 OSS 已配置时返回。"""
    global _bucket
    if not settings.OSS_ENABLED:
        return None
    if _bucket is None:
        try:
            import oss2
            auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
            _bucket = oss2.Bucket(auth, settings.OSS_ENDPOINT, settings.OSS_BUCKET_NAME)
        except Exception as e:
            logger.warning(f"OSS bucket init failed: {e}")
            return None
    return _bucket


def is_oss_enabled() -> bool:
    return settings.OSS_ENABLED and _get_bucket() is not None


def presign_upload(user_id: int, filename: str, content_type: str = "application/octet-stream") -> Optional[Dict[str, Any]]:
    """
    生成 OSS 直传预签名 URL。
    路径约定：data/{user_id}/{dataset_id}/raw/{filename}
    """
    bucket = _get_bucket()
    if not bucket:
        return None
    dataset_id = str(uuid.uuid4())
    object_key = f"data/{user_id}/{dataset_id}/raw/{filename}"
    url = bucket.sign_url("PUT", object_key, 3600)
    return {"upload_url": url, "object_key": object_key, "dataset_id": dataset_id}


def list_oss_datasets(user_id: int) -> List[Dict[str, str]]:
    """
    列举用户在 OSS 下的“数据集”前缀。
    约定：data/{user_id}/{dataset_id}/ 视为一个数据集，dataset_id 取路径倒数第二段。
    """
    bucket = _get_bucket()
    if not bucket:
        return []
    try:
        import oss2
        prefix = f"data/{user_id}/"
        datasets = []
        for obj in oss2.ObjectIterator(bucket, prefix=prefix, delimiter="/"):
            if obj.is_prefix():
                # key 形如 "data/1/xxx-uuid/"
                parts = obj.key.rstrip("/").split("/")
                if len(parts) >= 2:
                    dataset_id = parts[-1]
                    datasets.append({"dataset_id": dataset_id, "path": obj.key})
        return datasets
    except Exception as e:
        logger.warning(f"list_oss_datasets error: {e}")
        return []


# ---------- 以下为 theta-backend 的 OSS 用户存储，供可选“仅 OSS 认证”模式使用 ----------


def get_user_from_oss(email: str) -> Optional[Dict[str, Any]]:
    """从 OSS 读取用户：users/{email_safe}.json。未配置 OSS 或不存在则返回 None。"""
    import oss2
    bucket = _get_bucket()
    if not bucket:
        return None
    try:
        key = "users/" + email.replace("@", "_at_") + ".json"
        result = bucket.get_object(key)
        return json.loads(result.read().decode())
    except oss2.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning(f"get_user_from_oss error: {e}")
        return None


def save_user_to_oss(user_data: dict) -> bool:
    """写入用户到 OSS。"""
    bucket = _get_bucket()
    if not bucket:
        return False
    try:
        email = user_data.get("email")
        if not email:
            return False
        key = "users/" + email.replace("@", "_at_") + ".json"
        bucket.put_object(key, json.dumps(user_data))
        return True
    except Exception as e:
        logger.warning(f"save_user_to_oss error: {e}")
        return False
