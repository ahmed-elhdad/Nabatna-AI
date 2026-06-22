from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List

from src.models import ChatModel
from src.models.db_schemas.conversation import Conversation, Message


chat_router = APIRouter(prefix="/api/v1/chat", tags=["api_v1", "chat"])


class CreateConversationRequest(BaseModel):
    chat_id: str
    user_name: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    user_name: Optional[str]


class SendMessageRequest(BaseModel):
    sender: str
    text: str


@chat_router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_conversation(request: Request, payload: CreateConversationRequest):
    chat_model = await ChatModel.create_instance(db_client=request.app.state.db_client)
    conv = Conversation(chat_id=payload.chat_id, user_name=payload.user_name)
    created = await chat_model.create_conversation(conv)
    if not created:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    return created


@chat_router.get("/{chat_id}")
async def get_conversation(request: Request, chat_id: str):
    chat_model = await ChatModel.create_instance(db_client=request.app.state.db_client)
    conv = await chat_model.get_conversation_or_create_one(chat_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@chat_router.get("/")
async def list_conversations(request: Request, page: int = 1, page_size: int = 10):
    chat_model = await ChatModel.create_instance(db_client=request.app.state.db_client)
    convs, total_pages = await chat_model.get_all_conversations(page=page, page_size=page_size)
    return {"conversations": convs, "total_pages": total_pages}


@chat_router.put("/{chat_id}")
async def update_conversation(request: Request, chat_id: str, payload: UpdateConversationRequest):
    chat_model = await ChatModel.create_instance(db_client=request.app.state.db_client)
    update_fields = {k: v for k, v in payload.dict().items() if v is not None}
    updated = await chat_model.update_conversation(chat_id=chat_id, update_fields=update_fields)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found or not updated")
    return updated


@chat_router.delete("/{chat_id}")
async def delete_conversation(request: Request, chat_id: str):
    chat_model = await ChatModel.create_instance(db_client=request.app.state.db_client)
    deleted = await chat_model.delete_conversation(chat_id=chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@chat_router.post("/{chat_id}/message")
async def send_message(request: Request, chat_id: str, payload: SendMessageRequest):
    chat_model = await ChatModel.create_instance(db_client=request.app.state.db_client)
    message = Message(sender=payload.sender, text=payload.text)
    updated = await chat_model.add_message(chat_id=chat_id, message=message)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to add message")
    return updated