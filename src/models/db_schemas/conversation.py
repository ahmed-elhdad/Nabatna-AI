from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Message(BaseModel):
    sender: str
    text: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    chat_id: str
    user_name: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Pydantic's dict() will convert datetime to isoformat which is fine for DB storage
        return self.dict()
