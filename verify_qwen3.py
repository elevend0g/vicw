
import asyncio
import logging
import os
import sys
import numpy as np

# Add app directory to path
sys.path.append(os.path.join(os.getcwd(), "app"))

from config import EMBEDDING_MODEL_PATH
from llama_cpp import Llama

def verify_qwen3():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Qwen3Verification")
    
    logger.info(f"Loading model from {EMBEDDING_MODEL_PATH}...")
    
    try:
        model = Llama(
            model_path=EMBEDDING_MODEL_PATH,
            embedding=True,
            verbose=False
        )
        logger.info("Model loaded successfully!")
        
        text = "This is a test sentence for Qwen3 embedding."
        logger.info(f"Generating embedding for: '{text}'")
        
        response = model.create_embedding(text)
        embedding = response['data'][0]['embedding']
        
        logger.info(f"Embedding generated. Dimension: {len(embedding)}")
        logger.info(f"First 5 values: {embedding[:5]}")
        
        if len(embedding) == 1024:
            logger.info("✅ Dimension check passed (1024)")
        else:
            logger.warning(f"⚠️ Dimension check failed. Expected 1024, got {len(embedding)}")
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")

if __name__ == "__main__":
    verify_qwen3()
