from collections.abc import Iterator

from fastapi.responses import StreamingResponse

from rag import contexualize_history, retrieve_documents, chat_prompt
from store import load_vector_store

from langchain_core.messages import HumanMessage, BaseMessageChunk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from dto import Chat

llm = ChatOllama(model="gemma2:9b")

chat_history = []


def llm_stream(llm_respose: Iterator[BaseMessageChunk]):
    ai_message = ""
    for res in llm_respose:
        ai_message += res.content
        yield res.content
    chat_history.append(("ai", ai_message))


vector_store = load_vector_store(
    "faiss-store-v2",
    embedding_fn=HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1"),
)

prompt = chat_prompt(
    contexualize_history=contexualize_history(llm=llm),
    retrieve_documents=retrieve_documents(vectore_store=vector_store),
)


def chat_fn(chat: Chat):
    human_message = chat.human_message
    chat_history.append(HumanMessage(content=human_message))

    llm_prompt = prompt.invoke(
        {"chat_history": chat_history, "client_prompt": human_message}
    )

    llm_res = llm.stream(llm_prompt)
    res_stream = llm_stream(llm_res)

    return StreamingResponse(
        res_stream,
        media_type="text/event-stream",
    )
