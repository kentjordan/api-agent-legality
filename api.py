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

llm = ChatOllama(model="gemma:2b")

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


chat_history = [
    SystemMessage(
        content="You are an AI but be a lawyer that provides legal assitance, advice, and knows all the knowledge of the Philippines' Statues including the Acts, Commonwealth Acts, Mga Batas Pambansa, and Republic Acts"
    ),
    SystemMessage(
        content="You also know the Constituion of the Philippines including 1986 Constitution, 1987 Constitution, 1973 Constitution, 1943 Constitution, 1934 Constitution, and the Malolos Constitution."
    ),
    SystemMessage(
        content="Answer the problem or concern of the human, person or user as being a lawyer. Give legal assitance and advice."
    ),
    SystemMessage(
        content="Be kind and objective to your answer. If you do not know the answer, just say you do not know."
    ),
    SystemMessage(content="Do not respond to your own chats or messages"),
]


@server.get("/converstion/:id")
def conversation():
    pass


def llm_stream(llm_respose: Iterator[BaseMessageChunk]):
    ai_message = ""
    for res in llm_respose:
        ai_message += res.content
        yield res.content
    chat_history.append(AIMessage(content=ai_message))


@server.post("/chat")
def chat(chat: Chat):
    human_message = chat.human_message

    chat_history.append(HumanMessage(content=human_message))

    res_stream = llm_stream(llm.stream(chat_history))

    return StreamingResponse(
        res_stream,
        media_type="text/event-stream",
    )
