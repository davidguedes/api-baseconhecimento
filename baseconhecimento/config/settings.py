from dotenv import load_dotenv
import os
class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
    VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "gemma3:4b")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    CHROMA_DIRECTORY = os.getenv("CHROMA_DIRECTORY", "./chroma_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 10))