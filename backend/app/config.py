from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    APP_NAME: str = "Cloud-Based Internal Knowledge Assistant"
    ENV: str = os.getenv("ENV", "dev")

    DOCS_DIR: str = os.getenv("DOCS_DIR", "docs")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
    META_PATH: str = os.getenv("META_PATH", "data/meta.json")
    MAX_ANSWER_TOKENS: int = int(os.getenv("MAX_ANSWER_TOKENS", "250"))
settings = Settings()
