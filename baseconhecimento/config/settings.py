from dotenv import load_dotenv
import os
class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
    VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "gemma3:4b")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "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")
    CHROMA_DIRECTORY = os.getenv("CHROMA_DIRECTORY", "./chroma_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 50))

    # Configurações de paralelização
    MAX_PARALLEL_PAGES = 4
    MAX_PARALLEL_IMAGES = 2

    # Configurações de OCR
    OCR_LANGUAGES = ["por", "pt"]
    OCR_PSM = 6  # Assume uniform block of text
    OCR_OEM = 3  # Default OCR Engine Mode
    
    # Cache settings
    ENABLE_IMAGE_CACHE = True
    ENABLE_TEXT_CACHE = True
    MAX_CACHE_SIZE = 100