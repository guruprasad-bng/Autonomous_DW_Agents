"""Run with: python -m app.build_index"""
from .rag_store import rebuild_index
if __name__ == "__main__":
    print(rebuild_index())
