"""Read-only metadata catalog MCP server (demonstration adapter)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.server.fastmcp import FastMCP

CATALOG = Path(__file__).resolve().parents[1] / "data" / "catalog.json"
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

@mcp.tool()
def semantic_search(query: str, kind: str = "all", top_k: int = 5) -> dict:
    """RAG retrieval of semantically related catalog assets, data products and glossary terms; kind=all|asset|product|glossary."""
    from app.rag_store import search
    return search(query, kind, top_k)

@mcp.tool()
def search_glossary(query: str, top_k: int = 5) -> dict:
    """Find business glossary definitions and synonyms using embedding similarity."""
    from app.rag_store import search
    return search(query, "glossary", top_k)

if __name__=='__main__':
    mcp.run(transport='stdio')
