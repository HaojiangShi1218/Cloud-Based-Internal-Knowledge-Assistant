import secrets
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def require_admin_token(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> None:
    token = (settings.ADMIN_TOKEN or "").strip()
    if not token:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled")

    provided = (x_admin_token or "").strip()
    if not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="Forbidden")
