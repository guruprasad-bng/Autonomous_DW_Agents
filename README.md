# Autonomous Data Warehouse Agents

## What it does
- Discovery agent: catalog search and keyword-based data product recommendations.
- Impact agent: downstream dataset/column lineage traversal.
- Schema drift agent: compare incoming column/type JSON to registered schema.

All metadata is demo data in `data/catalog.json`. This is **not** connected to Dataplex, Knowledge Catalog, or Informatica EDC. Catalog data is read-only and impact completeness depends on recorded lineage.

## Setup (Windows PowerShell)
From the extracted project root:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```
Set a real `GROQ_API_KEY` in `.env` (do not commit this file). Run:
```powershell
streamlit run streamlit_app.py
```

## Test questions
1. Discovery: `Find customer data assets and recommend a customer profile data product.`
2. Impact: `Analyze downstream impact if loan_accounts.balance changes.`
3. Drift: `Check drift for loan_accounts with {"loan_id":"string","customer_id":"string","balance":"string","status":"string","currency":"string"}`

## Extending to real catalogs
Replace `mcp_servers/metadata_server.py`'s `load()` adapter with authenticated catalog API retrieval and lineage mapping. Add access controls, audit logging, column-level policy enforcement, lineage provenance, and automated schema snapshots before production use.

## Notes
- This is a demonstration prototype, not a production-ready solution.
- It does not perform scheduled schema monitoring; drift checks are on-demand.
- The original SQL firewall is intentionally not used because this demo exposes no SQL execution tool.
- MCP stdio tool connections are recreated by the adapter as needed; ensure your local dependencies are compatible.
