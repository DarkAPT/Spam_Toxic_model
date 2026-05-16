from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schemas import UserStatsResponse
from service import UserStatsService, ModerationActionsService

router = APIRouter(prefix="/api/stats", tags=["statistics"])

@router.get("/user/{user_id}", response_model=UserStatsResponse)
async def get_user_statistics(
    user_id:str,
    db:Session = Depends(get_db)
):
    try:
        response = UserStatsService.get_or_create_user_stats(db=db,user_id=user_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/users/top-violators")
async def get_top_violators(
    db:Session = Depends(get_db),
    limit: int = Query(10, ge=1,le=100)
):
    try:
        items = UserStatsService.get_top_violators(db=db,limit=limit)
        return {
            "count": len(items.items),
            "violators": items.items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/users/banned")
async def get_banned_users(
    db:Session = Depends(get_db),
    limit:int = Query(10, ge=1, le=100)
):
    try:
        items = UserStatsService.get_banned_users(db=db, limit=limit)
        return items.items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/moderation/actions/log/{log_id}")
async def get_actions_for_log(
    log_id:int,
    db:Session = Depends(get_db)
):
    try:
        actions = ModerationActionsService.get_actions_for_log(db=db, log_id=log_id)
        return actions.items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/moderation/actions/moderator/{moderator_id}")
async def get_moderator_actions(
    moderator_id:int,
    db:Session = Depends(get_db),
    limit:int =  Query(50, ge=1, le=200),
    offset:int = Query(0, ge=0)
):
    try:
        actions = ModerationActionsService.get_moderator_actions(db=db, moderator_id=moderator_id
                                                                 ,limit=limit, offset=offset)
        return actions.items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))