# modules/utils.py
import hashlib
import os
import logging

HASH_RECORD_PATH = "last_kb_hash.txt"
logger = logging.getLogger(__name__)

def get_file_hash(file_path):
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        logger.info(f"Computed hash for {file_path}: {file_hash}")
        return file_hash
    except Exception as e:
        logger.error(f"Failed to compute file hash for {file_path}: {e}")
        raise

def save_last_kb_hash(hash_value, hash_file=HASH_RECORD_PATH):
    try:
        with open(hash_file, "w") as f:
            f.write(hash_value)
        logger.info("Saved latest KB hash.")
    except Exception as e:
        logger.error(f"Failed to save last KB hash: {e}")
        raise

def load_last_kb_hash(hash_file=HASH_RECORD_PATH):
    try:
        if os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                return f.read().strip()
        return None
    except Exception as e:
        logger.error(f"Failed to read last KB hash: {e}")
        return None