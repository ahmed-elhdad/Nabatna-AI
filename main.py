from fastapi import FastAPI
from src.config import connect_mongodb
from src.routes import auth_router,chat_router
app = FastAPI()

app.state.db_client=connect_mongodb()
app.include_router(auth_router)

app.include_router(chat_router)
