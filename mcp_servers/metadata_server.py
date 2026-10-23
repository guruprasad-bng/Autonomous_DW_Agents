"""Read-only metadata catalog MCP server."""

import contextlib
import json
import sys
import time
import copy
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb

from mcp.server.fastmcp import FastMCP

CATALOG = ROOT / "data" / "catalog.json"
mcp = FastMCP("data-agents-catalog")


def load():
    return json.loads(CATALOG.read_text(encoding="utf-8"))

def tokens(s):
    return set(str(s).lower().replace('_', ' ').split())

@mcp.tool()
def search_catalog(query: str) -> dict:
    """Find metadata assets matching a business concept, domain, tag, or description."""
    terms = tokens(query)
    scored = []
    for asset in load()['assets']:
        fields = ' '.join(str(asset.get(k, '')) for k in ('name','description','domain','tags'))
        score = len(terms & tokens(fields))
        if score:
            scored.append({'match_score': score, **asset})
    return {'results': sorted(scored, key=lambda a: (-a['match_score'], -a['quality_score']))[:10]}

@mcp.tool()
def recommend_data_products(requirement: str) -> dict:
    """Rank data products by matching terms in their metadata and associated assets."""
    data = load(); terms = tokens(requirement); scored = []
    for product in data['products']:
        assets = [a for a in data['assets'] if a['id'] in product['assets']]
        text = ' '.join([product['name'], product['description'], str(product['tags'])] + [a['name']+' '+a['description'] for a in assets])
        score = len(terms & tokens(text))
        if score:
            scored.append({'match_score':score,'product':product,'asset_names':[a['name'] for a in assets], 'average_quality':round(sum(a['quality_score'] for a in assets)/len(assets),1) if assets else None})
    return {'recommendations':sorted(scored,key=lambda p:-p['match_score']), 'method':'metadata keyword overlap; not semantic search'}

@mcp.tool()
def get_asset(asset_id: str) -> dict:
    """Retrieve an asset's metadata and current registered schema by exact ID."""
    return next((a for a in load()['assets'] if a['id']==asset_id), {'error':'Asset not found'})

@mcp.tool()
def analyze_impact(asset_id: str, column: str = '') -> dict:
    """Traverse downstream dataset lineage, optionally following known column mappings."""
    data=load(); assets={a['id']:a for a in data['assets']}
    if asset_id not in assets: return {'error':'Asset not found'}
    frontier=[(asset_id,column,0)]; seen={(asset_id,column)}; impacts=[]
    while frontier:
        current,col,depth=frontier.pop(0)
        for edge in data['lineage']:
            if edge['upstream']!=current: continue
            mapped=edge.get('columns',{})
            if col and col not in mapped: continue
            nextcols=mapped.get(col,[]) if col else ['']
            for nextcol in nextcols:
                key=(edge['downstream'],nextcol)
                if key in seen: continue
                seen.add(key)
                target=assets.get(edge['downstream'],{})
                impacts.append({'asset_id':edge['downstream'],'asset_name':target.get('name'),'column':nextcol or None,'depth':depth+1,'owner':target.get('owner')})
                frontier.append((edge['downstream'],nextcol,depth+1))
    return {'source':asset_id,'source_column':column or None,'downstream':impacts,'coverage_note':'Only registered lineage edges are assessed; absence of a path does not prove no impact'}

@mcp.tool()
def detect_schema_drift(asset_id: str, incoming_schema_json: str) -> dict:
    """Compare incoming column:type JSON object to registered schema; report breaking changes."""
    asset=get_asset(asset_id)
    if 'error' in asset: return asset
    try: incoming=json.loads(incoming_schema_json)
    except json.JSONDecodeError: return {'error':'Incoming schema must be valid JSON'}
    if not isinstance(incoming,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in incoming.items()):
        return {'error':'Expected JSON object mapping column names to type strings'}
    old=asset['schema']; added=sorted(set(incoming)-set(old)); removed=sorted(set(old)-set(incoming))
    changed=[{'column':k,'previous_type':old[k],'incoming_type':incoming[k]} for k in sorted(set(old)&set(incoming)) if old[k].lower()!=incoming[k].lower()]
    breaking=bool(removed or changed)
    return {'asset_id':asset_id,'drift_detected':bool(added or removed or changed),'severity':'HIGH' if breaking else 'LOW' if added else 'NONE','added':added,'removed':removed,'type_changes':changed,'policy':'Conservative: any removal or type change is potentially breaking; validate against consumers','registered_schema':old,'incoming_schema':incoming}

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_catalog(data):
    """Atomically replace the catalog JSON file."""
    fd, tmp = tempfile.mkstemp(
        dir=str(CATALOG.parent),
        suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, CATALOG)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def schema_diff(old, new):
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = [
        {
            "column": col,
            "previous_type": old[col],
            "incoming_type": new[col]
        }
        for col in sorted(set(old) & set(new))
        if old[col].lower() != new[col].lower()
    ]
    return added, removed, changed


@mcp.tool()
def propose_schema_change(
    asset_id: str,
    incoming_schema_json: str
) -> dict:
    """Create a pending schema change. Does not update the schema."""
    data = load()
    asset = next(
        (a for a in data["assets"] if a["id"] == asset_id),
        None
    )
    if asset is None:
        return {"error": "Asset not found"}

    try:
        incoming = json.loads(incoming_schema_json)
    except json.JSONDecodeError:
        return {"error": "Invalid schema JSON"}

    if (
        not isinstance(incoming, dict)
        or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in incoming.items()
        )
    ):
        return {"error": "Expected {column: type} JSON"}

    old = asset["schema"]
    added, removed, changed = schema_diff(old, incoming)

    if not (added or removed or changed):
        return {"status": "NO_DRIFT", "asset_id": asset_id}

    severity = "HIGH" if removed or changed else "LOW"

    notes = []
    notes += [f"Added column: {c} ({incoming[c]})"
              for c in added]
    notes += [f"Removed column: {c}" for c in removed]
    notes += [
        f"Changed {c['column']}: "
        f"{c['previous_type']} -> {c['incoming_type']}"
        for c in changed
    ]

    request = {
        "request_id": str(uuid.uuid4()),
        "asset_id": asset_id,
        "status": "PENDING",
        "created_at": utc_now(),
        "base_version": asset.get("schema_version", 1),
        "registered_schema": copy.deepcopy(old),
        "incoming_schema": incoming,
        "severity": severity,
        "documentation": {
            "summary": "Schema drift detected; human review required",
            "change_notes": notes,
            "compatibility": (
                "Potentially breaking"
                if severity == "HIGH"
                else "Additive change"
            )
        }
    }

    data.setdefault("schema_change_requests", []).append(request)
    save_catalog(data)

    return request


@mcp.tool()
def resolve_schema_change(
    request_id: str,
    decision: str,
    reviewer: str
) -> dict:
    """Apply or reject a pending change after trusted approval.

    IMPORTANT: This tool must NOT be exposed to the LLM.
    Only the authenticated approval UI/backend may invoke it.
    """
    if decision not in ("APPROVE", "REJECT"):
        return {"error": "Invalid decision"}

    if not reviewer.strip():
        return {"error": "Reviewer identity is required"}

    data = load()
    request = next(
        (
            r for r in data.get("schema_change_requests", [])
            if r["request_id"] == request_id
        ),
        None
    )

    if request is None:
        return {"error": "Change request not found"}

    if request["status"] != "PENDING":
        return {"error": "Request already resolved"}

    asset = next(
        (
            a for a in data["assets"]
            if a["id"] == request["asset_id"]
        ),
        None
    )

    if asset is None:
        return {"error": "Asset not found"}

    # A stale proposal may be rejected/closed, but must never be approved.
    stale = (
        asset["schema"] != request["registered_schema"]
        or asset.get("schema_version", 1) != request["base_version"]
    )
    if stale and decision == "APPROVE":
        return {
            "error": "Catalog changed since proposal; approval blocked. Reject the stale request and submit a new proposal if needed.",
            "code": "STALE_PROPOSAL",
            "current_version": asset.get("schema_version", 1),
            "base_version": request["base_version"],
        }

    request["reviewer"] = reviewer
    request["reviewed_at"] = utc_now()

    if decision == "REJECT":
        request["status"] = "REJECTED"
        save_catalog(data)
        return {
            "status": "REJECTED",
            "asset_id": asset["id"],
            "schema_updated": False
        }

    old_version = asset.get("schema_version", 1)
    new_version = old_version + 1

    asset.setdefault("schema_history", []).append({
        "version": old_version,
        "schema": copy.deepcopy(asset["schema"]),
        "archived_at": utc_now(),
        "superseded_by_request": request_id
    })

    asset["schema"] = copy.deepcopy(request["incoming_schema"])
    asset["schema_version"] = new_version
    asset.setdefault("version_notes", []).append({
        "version": new_version,
        "request_id": request_id,
        "approved_by": reviewer,
        "approved_at": utc_now(),
        "severity": request["severity"],
        "documentation": copy.deepcopy(
            request["documentation"]
        )
    })

    request["status"] = "APPROVED"
    request["applied_version"] = new_version

    save_catalog(data)

    return {
        "status": "APPROVED",
        "asset_id": asset["id"],
        "previous_version": old_version,
        "new_version": new_version,
        "schema_updated": True
    }


@mcp.tool()
def semantic_search(
    query: str,
    kind: str = "all",
    top_k: int = 5
) -> dict:
    start = time.perf_counter()

    def log(message):
        log_path = (
            Path(__file__).resolve().parents[1]
            / "mcp_debug.log"
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[MCP DEBUG] {time.perf_counter() - start:.2f}s "
                f"{message}\n"
            )

    log("semantic_search entered")

    with contextlib.redirect_stdout(sys.stderr):

        # Step 1: Measure ChromaDB import
        log("Importing chromadb")
        import chromadb
        log("chromadb imported")

        # Step 2: Measure embedding function import
        log("Importing DefaultEmbeddingFunction")
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        log("DefaultEmbeddingFunction imported")

        # Step 3: Measure RAG module import
        log("Importing rag_store")
        from app.rag_store import search
        log("rag_store imported")

        # Step 4: Measure semantic search execution
        log("Starting ChromaDB search")
        result = search(query, kind, top_k)
        log("ChromaDB search completed")

    log("Returning semantic search result")
    return result


@mcp.tool()
def search_glossary(
    query: str,
    top_k: int = 5
) -> dict:
    """Semantic search for business glossary terms."""

    with contextlib.redirect_stdout(sys.stderr):
        from app.rag_store import search
        return search(query, "glossary", top_k)

if __name__=='__main__':
    mcp.run(transport='stdio')
