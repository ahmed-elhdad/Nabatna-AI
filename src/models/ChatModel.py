from .BaseDataModel import BaseDataModel
from typing import Optional, Dict, Any, List
from bson import ObjectId
import math

from .db_schemas.conversation import Conversation, Message


class ChatModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        # Normalize db and collection: accept either MotorClient, MotorDatabase or similar
        try:
            # If a client was passed (like AsyncIOMotorClient), try to pick a DB name
            if hasattr(db_client, "get_database"):
                self.db = db_client.get_database("nabatna")
            elif hasattr(db_client, "__getitem__"):
                # could be a client or a database
                try:
                    self.db = db_client["nabatna"]
                except Exception:
                    self.db = db_client
            else:
                self.db = db_client
        except Exception:
            self.db = db_client

        self.collection = getattr(self.db, "conversations", None) or self.db["conversations"]

    @classmethod
    async def create_instance(cls, db_client: object):
        return cls(db_client)

    async def create_conversation(self, chat: Conversation) -> Dict[str, Any]:
        doc = chat.to_dict()
        result = await self.collection.insert_one(doc)
        inserted = await self.collection.find_one({"_id": result.inserted_id})
        if inserted and "_id" in inserted:
            inserted["_id"] = str(inserted["_id"])
        return inserted

    async def get_conversation_or_create_one(self, chat_id: str, user_name: Optional[str] = None) -> Dict[str, Any]:
        chat = await self.collection.find_one({"chat_id": chat_id})
        if chat:
            chat["_id"] = str(chat["_id"])
            return chat

        # create new
        conv = Conversation(chat_id=chat_id, user_name=user_name)
        return await self.create_conversation(conv)

    async def get_all_conversations(self, page: int = 1, page_size: int = 10):
        total_documents = await self.collection.count_documents({})
        total_pages = math.ceil(total_documents / page_size) if page_size else 1
        skip_value = (page - 1) * page_size
        cursor = self.collection.find({}).skip(skip_value).limit(page_size)
        conversations = await cursor.to_list(length=page_size)
        for conversation in conversations:
            if "_id" in conversation:
                conversation["_id"] = str(conversation["_id"])

        return conversations, total_pages

    async def update_conversation(self, chat_id: str, update_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await self.collection.update_one({"chat_id": chat_id}, {"$set": update_fields})
        updated = await self.collection.find_one({"chat_id": chat_id})
        if updated and "_id" in updated:
            updated["_id"] = str(updated["_id"])
        return updated

    async def add_message(self, chat_id: str, message: Message) -> Optional[Dict[str, Any]]:
        # push message into messages array
        await self.collection.update_one({"chat_id": chat_id}, {"$push": {"messages": message.dict()}}, upsert=True)
        updated = await self.collection.find_one({"chat_id": chat_id})
        if updated and "_id" in updated:
            updated["_id"] = str(updated["_id"])
        return updated

    async def delete_conversation(self, chat_id: str) -> bool:
        result = await self.collection.delete_one({"chat_id": chat_id})
        return result.deleted_count > 0