# app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    APP_NAME: str = "Cloud-Based Internal Knowledge Assistant"
    ENV: str = os.getenv("ENV", "dev")

settings = Settings()
