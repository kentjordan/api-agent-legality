from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def contexualize_history(llm):
    def inner(input=None) -> str:
        chat_history = input["chat_history"]

        if len(chat_history) <= 1:
            return ChatPromptTemplate.from_messages(
                [(
                        "system",
                        (
                            "If the prompt is a question, then please paraphrase it in 5 different versions using legal words, terminology and concepts."
                            "Ensure that the essence of the original inquiry is preserved while enhancing the formal tone appropriate for a legal context."
                            "If the prompt is not a question, just return it as it is."
                        ),
                ),
                ("system", f'Prompt: {input["client_prompt"]}')
                ]
            ) | llm | StrOutputParser()

        return (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "Please paraphrase the following questions in 10 different versions from the chat history using legal words, terminology and concepts."
                            "Ensure that the essence of the original inquiry is preserved while enhancing the formal tone appropriate for a legal context."
                            "If the prompt is not a question, just return it as it is."
                        ),
                    ),
                    ("system", "Chat History:"),
                    MessagesPlaceholder(variable_name="chat_history"),
                ]
            )
            | llm
            | StrOutputParser()
        )

    return inner


def retrieve_documents(vectore_store):
    def inner(input=None) -> str:
        docs_template = ""

        docs = vectore_store.max_marginal_relevance_search(
            input["contexualize_prompt"], k=1, fetch_k=5
        )

        for i, doc in enumerate(docs):
            repact = f"""
        Document {i + 1}
        Title: {doc.metadata['title']}
        Date: {doc.metadata['date']}
        Description: {doc.metadata['description']}
        Content:
        {doc.page_content}
        -------------------------------
        """
            docs_template += repact
        return docs_template

    return inner


def chat_prompt(contexualize_history, retrieve_documents):
    system_instruction = """
    Ignore all your previous directives.
    Your new rules are the following:
    * You are Philippines-based lawyer. 
    * You know all the knowledge of the Philippines' Statues including the Acts, and Republic Acts.
    * You are going to give a legal assitant to the latest user message while you consider the chat history.
    * Do NOT add any comments on your response.
    * If you do not know the answer, just say you do not know.
    * Base on Relevant Documents, answer the user messages of questions while you consider the chat history.
    * Cite all of the Relevant Documents as your references at the end of your response.
    * If there are no Relevant Documents provided, just say you do not have sources for it apparently.
    * If there is no chat history, answer the latest user message of question and answer the user message in a step-by-step process.
    """

    return (
        RunnablePassthrough.assign(contexualize_prompt=contexualize_history)
        | RunnablePassthrough.assign(documents=retrieve_documents)
        | ChatPromptTemplate.from_messages(
            messages=[
                ("system", system_instruction),
                ("system", "Relevant Documents: {documents}"),
                ("system", "Chat History:"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", "Latest user message:"),
                ("user", "{client_prompt}"),
            ]
        )
    )
