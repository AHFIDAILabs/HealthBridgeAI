# populate_kb.py: populate knowledge base from ZIP file
import sys
import logging
import os
from pathlib import Path

# Add the current directory to Python path to ensure modules can be imported
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from modules.knowledge_base_manager import load_documents_from_huggingface, load_documents_from_zip, chunk_documents
from modules.vector_store_manager import save_vector_store
from modules.utils import get_file_hash, save_last_kb_hash, load_last_kb_hash

# Fix Windows console encoding issues BEFORE setting up logging
if sys.platform.startswith('win'):
    import codecs
    # Use a safer approach: set encoding on the handler instead of detaching stdout
    sys.stdout.reconfigure(encoding='utf-8')

# Set up logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main(zip_path=None):
    """Main function to populate the knowledge base from HuggingFace or local ZIP."""
    try:
        documents = None
        current_hash = None
        
        # Try to load from HuggingFace first
        if not zip_path:
            logger.info("Step 0: Checking if knowledge base needs updating...")
            logger.info("Attempting to download from HuggingFace...")
            
            try:
                from huggingface_hub import hf_hub_download
                
                temp_zip_path = hf_hub_download(
                    repo_id="AHFIDAILabs/tb-knowledge-base",
                    filename="TB_knowledge_base.zip",
                    repo_type="dataset"
                )
                
                current_hash = get_file_hash(temp_zip_path)
                last_hash = load_last_kb_hash()
                
                if current_hash == last_hash:
                    logger.info("Knowledge base is up-to-date. No changes detected.")
                    logger.info("Skipping vector store update.")
                    return
                
                logger.info("New data detected. Updating knowledge base...")
                logger.info("Step 1: Loading documents from HuggingFace repository...")
                documents = load_documents_from_huggingface()
                
            except Exception as e:
                logger.warning(f"Failed to download from HuggingFace: {e}")
                logger.info("Falling back to local ZIP file if available...")
                
                # Try to find local ZIP file
                if os.path.exists("data/TB_knowledge_base.zip"):
                    zip_path = "data/TB_knowledge_base.zip"
                    logger.info(f"Found local ZIP file: {zip_path}")
                elif os.path.exists("TB_knowledge_base.zip"):
                    zip_path = "TB_knowledge_base.zip"
                    logger.info(f"Found local ZIP file: {zip_path}")
                else:
                    logger.error("Could not reach HuggingFace and no local ZIP file found.")
                    logger.error("Please ensure internet connection or provide a local ZIP file.")
                    sys.exit(1)
        
        # Load from local ZIP file if specified or if HuggingFace failed
        if zip_path and not documents:
            if not os.path.exists(zip_path):
                logger.error(f"ZIP file not found: {zip_path}")
                sys.exit(1)
            
            if not zip_path.lower().endswith('.zip'):
                logger.error(f"File must be a ZIP file: {zip_path}")
                sys.exit(1)
            
            logger.info(f"Loading from local ZIP file: {zip_path}")
            
            # Check hash for local file
            logger.info("Step 0: Checking if knowledge base needs updating...")
            current_hash = get_file_hash(zip_path)
            last_hash = load_last_kb_hash()
            
            if current_hash == last_hash:
                logger.info("Knowledge base is up-to-date. No changes detected.")
                logger.info("Skipping vector store update.")
                return
            
            logger.info("New data detected. Updating knowledge base...")
            logger.info("Step 1: Loading documents from ZIP file...")
            documents = load_documents_from_zip(zip_path)

        if not documents:
            logger.error("No documents found or all documents failed to load.")
            logger.info("Supported file types: .pdf, .txt, .md")
            sys.exit(1)

        logger.info(f"Successfully loaded {len(documents)} documents")

        # Step 2: Chunk documents
        logger.info("Step 2: Splitting documents into chunks...")
        chunked_docs = chunk_documents(documents)
        
        if not chunked_docs:
            logger.error("Document chunking failed or resulted in no chunks.")
            sys.exit(1)
            
        logger.info(f"Created {len(chunked_docs)} chunks from documents")

        # Step 3: Save to vector store
        logger.info("Step 3: Saving chunks to vector store...")
        save_vector_store(chunked_docs)
        logger.info("Vector store update completed successfully")

        # Step 4: Update hash record
        if current_hash:
            logger.info("Step 4: Updating hash record...")
            save_last_kb_hash(current_hash)
            logger.info(f"Hash record updated: {current_hash}")

        logger.info("Knowledge base population completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error populating knowledge base: {e}")
        sys.exit(1)

if __name__ == "__main__":
    zip_path = None
    
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
        # Convert to absolute path
        zip_path = os.path.abspath(zip_path)
    
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {__file__}")
    
    if zip_path:
        logger.info(f"Using provided ZIP file: {zip_path}")
    else:
        logger.info("No ZIP file provided. Will attempt HuggingFace download.")
    
    main(zip_path)