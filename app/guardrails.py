"""Metadata-only guardrails; original SQL guardrail is not needed for this catalog demo."""
from .mcp_clients import ROOT
SAFE_TOOLS={'search_catalog','recommend_data_products','get_asset','analyze_impact','detect_schema_drift','semantic_search','search_glossary','propose_schema_change'}
MAX_QUESTION=4000

BLOCKED_TOOLS = {
    "resolve_schema_change",
}

def validate_question(question):
    if not isinstance(question,str) or not question.strip(): raise ValueError('Question is empty')
    if len(question)>MAX_QUESTION: raise ValueError('Question exceeds 4000 characters')
    return question.strip()
  
def filter_tools(tools):
    return [
        tool for tool in tools
        if tool.name in SAFE_TOOLS
        and tool.name not in BLOCKED_TOOLS
    ]
