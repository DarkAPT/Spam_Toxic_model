from sqlalchemy import func
from model_manager import model_manager
from config import settings
from database import PredictionLogs, ModerationActions, UserStats
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from schemas import ModerationUpdate, PredictionRequest, PredictionResponse, ModerationUpdateResponse, UserStatItem, UserStatItemList, ModerationActionItem, ModerationActionItemList


class PredictionService:
    @staticmethod
    def predict_content(request: PredictionRequest):
        toxic_prob = model_manager.toxic_predict(text=request.text)
        spam_prob = model_manager.spam_predict(text=request.text)
        
        if (toxic_prob > settings.toxic_block_threshold or
            spam_prob > settings.spam_block_threshold):
            decision = "block"
        elif(toxic_prob > settings.toxic_review_threshold or
             spam_prob > settings.spam_review_threshold):
            decision = "review"
        else:
            decision = "allow"
        
        return PredictionResponse(
            decision=decision,
            spam_prob=spam_prob,
            toxic_prob=toxic_prob
        )
    
    @staticmethod
    def save_prediction_to_db(
        db: Session,
        request: PredictionRequest,
        response: PredictionResponse,
    ):
        log_entry = PredictionLogs(
            user_id = request.user_id,
            text = request.text,
            toxic_model_res = response.toxic_prob > settings.toxic_block_threshold,
            spam_model_res = response.spam_prob > settings.spam_block_threshold,
            final_decision = response.decision,
            confidence = max(response.spam_prob, response.toxic_prob)
        )
        
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        if request.user_id:
            UserStatsService.update_user_stats(db, request.user_id, response.decision)
        
        db.commit()
        return log_entry

class ModerationService:
    @staticmethod
    def get_pending_moderation(
        db:Session,
        decision: Optional[str] = None,
        limit:int = 50,
        offset:int = 0,
        need_review:bool = True
    ):
        query = db.query(PredictionLogs)
        if decision:
            query = query.filter(PredictionLogs.final_decision == decision)
        if need_review:
            query = query.filter(PredictionLogs.moderator_conf == False)
            
        query = query.order_by(PredictionLogs.timestamp.desc())
        return query.offset(offset).limit(limit).all()
        
    @staticmethod
    def update_moderation_decision(
        log_id:int,
        update: ModerationUpdate,
        moderator_id:int,
        db:Session
    ):
        log_entry = db.query(PredictionLogs).filter(PredictionLogs.id == log_id).first()
        if not log_entry:
            raise ValueError("Запись не найдена")
        
        if update.decision not in ["block", "allow", "confirm"]:
            raise ValueError("Решение должно быть одно из следующих: block, allow, confirm")
            
        previous_decision = log_entry.final_decision
        log_entry.moderator_conf = True
        
        if update.decision == "confirm":
            log_entry.correct_label = True
        elif update.decision != previous_decision:
            log_entry.correct_label = False
            log_entry.final_decision = update.decision
        
        
        if log_entry.user_id:
            UserStatsService.update_user_stats(db, log_entry.user_id, log_entry.final_decision)
            
        db.commit()
        
        ModerationActionsService.log_moderation_action(
            db=db,
            log_id=log_id,
            decision=update.decision,
            previous_decision=previous_decision,
            final_decision=log_entry.final_decision,
            moderator_id=moderator_id
        )
        
        
        return ModerationUpdateResponse(
            status = "success",
            message = f"Решение обновлено на '{update.decision}'",
            id = log_id,
            moderator_decision = update.decision,
            final_decision = log_entry.final_decision,
            system_was_correct = log_entry.correct_label
        )

class ModerationActionsService:
    @staticmethod
    def log_moderation_action(
        db: Session,
        log_id:int,
        decision:str,
        previous_decision:str,
        moderator_id:int,
        final_decision:str
    ):  
        moderation_action = ModerationActions(
            log_id = log_id,
            moderator_id = moderator_id,
            action = decision,
            previous_decision = previous_decision,
            new_decision = final_decision
        )
        
        db.add(moderation_action)
        db.commit()
        db.refresh(moderation_action)
        
        return moderation_action
    
    @staticmethod
    def get_actions_for_log(
        db:Session,
        log_id:int
    ):
        query = db.query(ModerationActions).filter(ModerationActions.log_id == log_id).order_by(ModerationActions.timestamp).all()
        schemas = [ModerationActionItem.model_validate(item) for item in query]
        return ModerationActionItemList(items=schemas)
    
    @staticmethod
    def get_moderator_actions(
        db:Session,
        moderator_id:int,
        limit:int=20,
        offset:int = 0
    ):
        query = db.query(ModerationActions).filter(ModerationActions.moderator_id == moderator_id
                                                    ).order_by(ModerationActions.timestamp.desc()
                                                    ).offset(offset).limit(limit).all()
        schemas = [ModerationActionItem.model_validate(item) for item in query]
        return ModerationActionItemList(items=schemas)


class UserStatsService:
    @staticmethod
    def get_or_create_user_stats(
        db:Session,
        user_id:str
    ):
        user_stat = db.query(UserStats).filter(UserStats.user_id==user_id).first()
        
        if not user_stat:
            user_stat = UserStats(user_id=user_id)
            db.add(user_stat)

        return user_stat
    
    @staticmethod
    def update_user_stats(
        db:Session,
        user_id:str,
        decision:str
    ):
        user_stat = UserStatsService.get_or_create_user_stats(db=db,user_id=user_id)
        
        if decision == "block":
            user_stat.violations_count += 1
            user_stat.last_violation = datetime.utcnow()
            
        user_stat.total_content += 1
        
        violations_last_24h = db.query(func.count(PredictionLogs.id)).filter(
            PredictionLogs.user_id == user_id,
            PredictionLogs.final_decision == "block",
            PredictionLogs.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).scalar()
        
        if violations_last_24h >= 3:
            user_stat.status = "banned"
        elif violations_last_24h >= 2:
            user_stat.status = "warned"

        return user_stat
        
    @staticmethod
    def get_top_violators(
        db:Session,
        limit:int = 10
    ):
        violators = db.query(UserStats).order_by(UserStats.violations_count.desc()).limit(limit).all()
            
        items = [UserStatItem.model_validate(item) for item in violators]
        return UserStatItemList(items=items)
        
    @staticmethod
    def get_banned_users(
        db:Session,
        limit:int = 10
    ):
        banned_users = db.query(UserStats
                         ).filter(UserStats.status == "banned"
                                  ).order_by(UserStats.last_violation.desc()
                                             ).limit(limit).all()
        
        items = [UserStatItem.model_validate(item) for item in banned_users]
        return UserStatItemList(items=items)
    
    