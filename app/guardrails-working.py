"""Metadata-only guardrails; original SQL guardrail is not needed for this catalog demo."""
from .mcp_clients import ROOT
SAFE_TOOLS={'search_catalog','recommend_data_products','get_asset','analyze_impact','detect_schema_drift','semantic_search','search_glossary'}
MAX_QUESTION=4000

def validate_question(question):
    if not isinstance(question,str) or not question.strip(): raise ValueError('Question is empty')
    if len(question)>MAX_QUESTION: raise ValueError('Question exceeds 4000 characters')
    return question.strip()

def filter_tools(tools):
    return [t for t in tools if t.name in SAFE_TOOLS]
