import os
# python-dotenv reads .env and copies key=value pairs into os.environ.
# Keeps secrets like GROQ_API_KEY out of source control.
from dotenv import load_dotenv
# LangChain wrapper around Groq's OpenAI-compatible API.
# Swappable later for ChatOpenAI / ChatAnthropic with no other code changes.
from langchain_groq import ChatGroq

# Load env vars once at import time.
load_dotenv()

def get_llm():
    """Return a configured ChatGroq instance. Fail loudly if the key is missing."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY in .env")
    # Default model is a small fast OSS model; override via GROQ_MODEL=...
    # temperature=0 -> deterministic answers (essential for SQL/analytics).
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                    temperature=0)