import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant.

Answer the user's question based ONLY on the transcript
context provided below.

The context belongs to the video currently being discussed.

Do NOT use information from other videos.

Do NOT use outside knowledge.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Do not make up or infer information that is not present
in the transcript.

Always be concise and precise.

If quoting someone, mention it clearly.

Context from the current video transcript:
{context}""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )


# ------------------------------------------------------------------
# Build RAG chain for a specific video
# ------------------------------------------------------------------

def build_rag_chain(
    transcript: str,
    video_id: str,
):
    print("Building RAG chain...")
    print(f"Video ID: {video_id}")

    # --------------------------------------------------------------
    # Build vector store
    # --------------------------------------------------------------

    vector_store = build_vector_store(
        transcript=transcript,
        video_id=video_id,
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # Retriever is restricted to this video's video_id
    # --------------------------------------------------------------

    retriever = get_retriever(
        vector_store=vector_store,
        video_id=video_id,
        k=4,
    )

    llm = get_llm()

    prompt = get_prompt()

    # --------------------------------------------------------------
    # Full LCEL RAG pipeline
    # --------------------------------------------------------------

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ------------------------------------------------------------------
# Load RAG chain for an existing video
# ------------------------------------------------------------------

def load_rag_chain(
    video_id: str,
):
    print("Loading RAG chain...")
    print(f"Video ID: {video_id}")

    # --------------------------------------------------------------
    # Load persistent ChromaDB
    # --------------------------------------------------------------

    vector_store = load_vector_store()

    # --------------------------------------------------------------
    # Retrieve ONLY this video's chunks
    # --------------------------------------------------------------

    retriever = get_retriever(
        vector_store=vector_store,
        video_id=video_id,
        k=4,
    )

    llm = get_llm()

    prompt = get_prompt()

    # --------------------------------------------------------------
    # LCEL RAG pipeline
    # --------------------------------------------------------------

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ------------------------------------------------------------------
# Ask question
# ------------------------------------------------------------------

def ask_question(
    rag_chain,
    question: str,
) -> str:

    print(f"Question: {question}")

    answer = rag_chain.invoke(
        question
    )

    print(f"Answer: {answer}")

    return answer