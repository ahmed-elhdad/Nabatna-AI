from fastapi import FastAPI
from src.config import connect_mongodb
from src.routes.chat import chat_router
app=FastAPI()

app.state.db_client=connect_mongodb()

app.include_router(chat_router)