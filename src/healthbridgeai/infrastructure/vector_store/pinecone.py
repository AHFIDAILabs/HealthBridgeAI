# Vector Store Management - vector_store_manager.py: vector store management utilities for Lighthouse HealthConnect
import logging
from typing import List
from langchain_core.documents import Document
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import time

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    PINECONE_CLOUD,
    EMBEDDING_MODEL,
    TEXT_KEY,
)

logger = logging.getLogger(__name__)


def _init_pinecone_index():
    """
    Initializes Pinecone client and ensures the index exists.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in environment variables")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(
            f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one in {PINECONE_CLOUD}-{PINECONE_REGION}..."
        )
        
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # matches paraphrase-multilingual-MiniLM-L12-v2 embedding size
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            while True:
                index_stats = pc.describe_index(PINECONE_INDEX_NAME)
                if index_stats.status.ready:
                    break
                time.sleep(1)
            logger.info("Index is ready!")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    else:
        logger.info(f"Using existing index '{PINECONE_INDEX_NAME}'.")

    return pc


def get_vector_store():
    """
    Returns an initialized LangchainPinecone vector store instance.
    This function is used for querying the Pinecone vector store.
    """
    try:
        _init_pinecone_index()
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            text_key=TEXT_KEY,
        )
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        return vector_store
    except Exception as e:
        logger.error(f"Could not get LangchainPinecone vector store: {e}")
        raise


def save_vector_store(documents: List[Document]):
    """
    Saves already chunked documents to the Pinecone vector store in batches.
    """
    if not documents:
        logger.info("No documents provided to save to Pinecone. Skipping.")
        return

    try:
        pc = _init_pinecone_index()
        
        # Get the index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Clear existing data (optional - remove if you want to append)
        logger.info("Clearing existing vectors from index...")
        try:
            # Check if index has any vectors first
            index_stats = index.describe_index_stats()
            if index_stats.total_vector_count > 0:
                logger.info(f"Found {index_stats.total_vector_count} existing vectors, deleting...")
                index.delete(delete_all=True)
                # Wait a bit for deletion to complete
                time.sleep(2)
            else:
                logger.info("Index is already empty, skipping deletion")
        except Exception as e:
            logger.warning(f"Could not clear existing vectors (this is OK for new indexes): {e}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store and add documents
        vector_store = LangchainPinecone(
            index=index,
            embedding=embeddings,
            text_key=TEXT_KEY,
        )

        BATCH_SIZE = 50  # Reduced batch size for stability
        total_chunks_saved = 0
        
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            try:
                vector_store.add_documents(batch)
                total_chunks_saved += len(batch)
                logger.info(
                    f"Saved batch {i // BATCH_SIZE + 1} "
                    f"({total_chunks_saved}/{len(documents)} chunks saved)"
                )
                # Small delay between batches
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to save batch {i // BATCH_SIZE + 1}: {e}")
                # Continue with next batch

        logger.info(
            f"Saved {total_chunks_saved} chunks to index '{PINECONE_INDEX_NAME}'."
        )
    except Exception as e:
        logger.error(
            f"Failed to save documents to Pinecone index '{PINECONE_INDEX_NAME}': {e}"
        )
        raise