import os
from langchain_community.vectorstores import FAISS


def load_vector_store(name, embedding_fn, documents=None):
    path = os.path.join("./", name)
    if os.path.exists(path) and not documents:
        return FAISS.load_local(
            path, embeddings=embedding_fn, allow_dangerous_deserialization=True
        )

    FAISS.from_documents(documents=documents, embedding=embedding_fn).save_local(path)
    return FAISS.load_local(
        path, embeddings=embedding_fn, allow_dangerous_deserialization=True
    )
