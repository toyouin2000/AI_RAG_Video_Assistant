from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def build_vector_store(
    transcript: str,
    video_id: str
) -> Chroma:

    print("Building vector store...")
    print(f"Video ID: {video_id}")

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    if not video_id:
        raise ValueError("video_id is required.")

    # ---------------------------------------------------------
    # Split transcript into chunks
    # ---------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    print(f"Created {len(chunks)} transcript chunks.")

    # ---------------------------------------------------------
    # Add video_id to every document
    # ---------------------------------------------------------

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "chunk_index": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    # ---------------------------------------------------------
    # Create embeddings
    # ---------------------------------------------------------

    embeddings = get_embeddings()

    # ---------------------------------------------------------
    # Connect to ChromaDB
    # ---------------------------------------------------------

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    # ---------------------------------------------------------
    # Add documents
    # ---------------------------------------------------------

    vector_store.add_documents(
        documents=docs
    )

    print("Vector store created successfully.")

    return vector_store


def load_vector_store() -> Chroma:

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def get_retriever(
    vector_store: Chroma,
    video_id: str,
    k: int = 4
):

    if not video_id:
        raise ValueError("video_id is required.")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {
                "video_id": video_id
            }
        }
    )