from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from dto.Chat import Chat

from chat.service import chat_fn


server = FastAPI()

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@server.get("/")
def root():
    return "Legality's AI Agent API"


@server.post("/chat")
def chat(chat: Chat):
    return chat_fn(chat)
