from typing import Optional
from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session
from database import SystemUsers, get_db
from auth import auth_service
from service import ModerationService
from schemas import ModerationUpdate
from config import settings

router = APIRouter(prefix="/api/moderation", tags=["moderation"])

@router.get("/pending")
async def get_pending_moderation(
    decision:Optional[str] = Query(None),
    limit:int = 50,
    offset:int = 0,
    needs_review:bool = True,
    db:Session = Depends(get_db),
):
    try:
        query = ModerationService.get_pending_moderation(
            db=db,
            decision=decision,
            limit=limit,
            offset=offset,
            need_review=needs_review
        )
        return query
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{log_id}")
@settings.limiter.limit("45/minute")
async def update_moderation_decision(
    request:Request,
    log_id:int,
    update: ModerationUpdate,
    db:Session = Depends(get_db),
    current_moderator:SystemUsers = Depends(auth_service.get_current_moderator),
):
    try:
        response = ModerationService.update_moderation_decision(
            log_id=log_id,
            update=update,
            db=db,
            moderator_id=current_moderator.id)
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))