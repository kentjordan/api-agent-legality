# from dotenv import load_dotenv
# import os

from collections.abc import Iterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    BaseMessageChunk,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from fastapi.middleware.cors import CORSMiddleware

from dto.Chat import Chat

# load_dotenv()

# API_KEY = os.getenv("GROQ_KEY")

llm = ChatOllama(model="gemma:2b")

server = FastAPI()

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def full_text(iter: Iterator[BaseMessageChunk]):
    return "".join([res.content for res in list(iter)])


@server.get("/")
def root():
    return "Legality's AI Agent API"


chat_history = [
    SystemMessage(
        content="You are a lawyer that knows all the knowledge of the Philippines Laws or the Repulic Acts of the Philippines."
    )
]


@server.get("/converstion/:id")
def conversation():
    pass


@server.post("/chat")
def chat(chat: Chat):
    human_message = chat.human_message

    chat_history.append(HumanMessage(content=human_message))

    res_stream = llm.stream(chat_history)
    res_stream_list = list(res_stream)  # Copy to list
    res_stream = (  # Recreate the generator object
        res.content for res in res_stream_list
    )

    chat_history.append(AIMessage(content=full_text(res_stream_list)))

    return StreamingResponse(
        res_stream,
        media_type="text/event-stream",
    )
