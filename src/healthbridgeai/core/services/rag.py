# Knowledge base management utilities for lightHouse Connect: modules/knowledge_base_manager.py
import os
import tempfile
import zipfile
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

logger = logging.getLogger(__name__)

def load_documents_from_huggingface(
    repo_id: str = "AHFIDAILabs/tb-knowledge-base",
    filename: str = "TB_knowledge_base.zip",
    token: Optional[str] = None,
    max_retries: int = 3
) -> List[Document]:
    """
    Downloads and loads documents from a HuggingFace repository.
    Extracts and processes files from a ZIP archive.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: Name of the ZIP file in the repository
        token: HuggingFace API token (if needed for private repos)
        max_retries: Maximum number of download attempts
    
    Returns:
        List of loaded documents
    """
    documents = []
    
    if hf_hub_download is None:
        logger.error("huggingface-hub package not found. Install with: pip install huggingface-hub")
        raise ImportError("huggingface-hub is required but not installed")
    
    import time
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {filename} from HuggingFace: {repo_id} (attempt {attempt + 1}/{max_retries})")
            
            zip_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                resume_download=True
            )
            
            logger.info(f"Successfully downloaded to: {zip_file_path}")
            
            # Process the downloaded ZIP file
            documents = load_documents_from_zip(zip_file_path)
            
            return documents
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} download attempts failed.")
                logger.error("Possible causes:")
                logger.error("1. Network connectivity issue")
                logger.error("2. Repository access denied")
                logger.error("3. File not found in repository")
                logger.error(f"Repository: https://huggingface.co/datasets/{repo_id}")
                raise

def load_documents_from_zip(zip_file_path: str) -> List[Document]:
    """
    Extracts and loads documents from a ZIP archive using a temp dir.
    Supports .txt, .pdf, and .md files.
    """
    documents = []
    
    if not os.path.exists(zip_file_path):
        logger.error(f"ZIP file not found: {zip_file_path}")
        return documents
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            with tempfile.TemporaryDirectory() as temp_dir:
                for filename in zf.namelist():
                    # Skip directories and non-supported files
                    if filename.endswith('/') or not filename.lower().endswith((".pdf", ".txt", ".md")):
                        continue
                    
                    try:
                        # Extract file to temp directory
                        temp_file_path = os.path.join(temp_dir, os.path.basename(filename))
                        with open(temp_file_path, "wb") as f:
                            f.write(zf.read(filename))
                        
                        # Load document based on file type
                        if filename.lower().endswith(".pdf"):
                            loader = PyPDFLoader(temp_file_path)
                        elif filename.lower().endswith(".txt"):
                            loader = TextLoader(temp_file_path, encoding='utf-8')
                        elif filename.lower().endswith(".md"):
                            loader = UnstructuredMarkdownLoader(temp_file_path)
                        else:
                            continue
                        
                        loaded_docs = loader.load()
                        # Add source metadata
                        for doc in loaded_docs:
                            doc.metadata['source'] = filename
                        
                        documents.extend(loaded_docs)
                        logger.info(f"Loaded {len(loaded_docs)} documents from {filename}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process file {filename}: {e}")
                        continue
                        
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_file_path}")
    except Exception as e:
        logger.error(f"Error reading ZIP file {zip_file_path}: {e}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunked_docs = splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise