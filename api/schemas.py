from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

#Items/Lists

class ModerationItem(BaseModel):
    id: int
    text: str
    user_id: Optional[str]
    timestamp: datetime
    toxic_model_res: bool
    spam_model_res: bool
    final_decision: str
    confidence: float
    moderator_conf: bool
    correct_label: bool
    
    model_config = ConfigDict(from_attributes=True)
 
class ModerationActionItem(BaseModel):
    log_id: int
    moderator_id:Optional[int]
    action:str
    previous_decision:str
    new_decision:str
    timestamp:datetime
    
    model_config = ConfigDict(from_attributes=True)
        
class ModerationActionItemList(BaseModel):
    items: List[ModerationActionItem]

class UserStatItem(BaseModel):
    user_id:str
    violations_count:int
    total_content:int
    status:str
    
    model_config = ConfigDict(from_attributes=True)
    
class UserStatItemList(BaseModel):
    items: List[UserStatItem]
    
class ModerationUpdate(BaseModel):
    decision: str  # "block", "allow", "confirm"
    
# Requests

class PredictionRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    
class UserRequest(BaseModel):
    username:str
    password:str
    
# Respones

class PredictionResponse(BaseModel):
    decision: str # block, allow, confirm
    toxic_prob: float
    spam_prob: float

class ModerationUpdateResponse(BaseModel):
    status: str
    message: str
    id: int
    moderator_decision: str
    final_decision: str
    system_was_correct: bool
    
class UserStatsResponse(BaseModel):
    user_id:str
    violations_count:int
    total_content:int
    status:str
    last_violation: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
    
class ModeratorActionsResponse(BaseModel):
    moderator_id: int
    count: int
    actions: List[ModerationActionItem]
