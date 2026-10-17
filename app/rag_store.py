"""Persistent ChromaDB metadata/glossary RAG using ONNX (no PyTorch)."""
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "enterprise_metadata"
EMBEDDING_MODEL = "ChromaDB DefaultEmbeddingFunction (ONNX all-MiniLM-L6-v2)"
_embedding_function = None


def embedding_function():
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = DefaultEmbeddingFunction()
    return _embedding_function


def documents():
    catalog = json.loads((ROOT / "data/catalog.json").read_text(encoding="utf-8"))
    glossary = json.loads((ROOT / "data/business_glossary.json").read_text(encoding="utf-8"))
    docs = []
    for a in catalog["assets"]:
        content = (f"Dataset {a['name']}. Description: {a['description']}. "
                   f"Domain: {a['domain']}. Tags: {', '.join(a['tags'])}. "
                   f"Columns: {', '.join(a['schema'])}. Owner: {a['owner']}.")
        docs.append((f"asset:{a['id']}", content,
                     {"kind": "asset", "record_id": a["id"], "name": a["name"]}))
    for p in catalog["products"]:
        content = (f"Data product {p['name']}. Description: {p['description']}. "
                   f"Tags: {', '.join(p['tags'])}. Source asset IDs: {', '.join(p['assets'])}. "
                   f"Owner: {p['owner']}.")
        docs.append((f"product:{p['id']}", content,
                     {"kind": "product", "record_id": p["id"], "name": p["name"]}))
    for g in glossary["terms"]:
        content = (f"Business glossary term {g['term']}. Definition: {g['definition']}. "
                   f"Synonyms: {', '.join(g['synonyms'])}. Domain: {g['domain']}. "
                   f"Related assets: {', '.join(g['related_assets'])}. "
                   f"Related products: {', '.join(g['related_products'])}.")
        docs.append((f"glossary:{g['id']}", content,
                     {"kind": "glossary", "record_id": g["id"], "name": g["term"]}))
    return docs


def _client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def collection():
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function(),
    )


def rebuild_index():
    docs = documents()
    client = _client()
    # Recreate the collection to avoid mixing embeddings from different models.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function(),
    )
    if docs:
        col.upsert(
            ids=[d[0] for d in docs],
            documents=[d[1] for d in docs],
            metadatas=[d[2] for d in docs],
        )
    return {"indexed_documents": len(docs),
            "embedding_model": EMBEDDING_MODEL, "storage": "data/chroma_db"}


def search(query: str, kind: str = "all", top_k: int = 5):
    if not isinstance(query, str) or not query.strip():
        return {"error": "Query must not be empty"}
    if kind not in {"all", "asset", "product", "glossary"}:
        return {"error": "Invalid kind"}
    col = collection()
    if col.count() == 0:
        return {"error": "Vector index is empty. Run python -m app.build_index first."}
    kwargs = {
        "query_texts": [query],
        "n_results": min(max(1, int(top_k)), col.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if kind != "all":
        kwargs["where"] = {"kind": kind}
    result = col.query(**kwargs)
    hits = []
    for source_id, doc, meta, distance in zip(
        result["ids"][0], result["documents"][0],
        result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "source_id": source_id, "kind": meta["kind"],
            "record_id": meta["record_id"], "name": meta["name"],
            "text": doc, "cosine_distance": round(float(distance), 4),
        })
    return {"query": query, "retrieval": "embedding cosine similarity",
            "hits": hits,
            "note": "Similarity is relevance evidence, not an authorization or quality guarantee."}
