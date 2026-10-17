"""MCP client loader, adapted from the uploaded mcp_clients.py."""
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).parent / 'config' / 'mcp.json'

def load_config():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))['mcpServers']
    for spec in cfg.values():
        if spec['command']=='python': spec['command']=sys.executable
        spec['args']=[re.sub(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}',lambda m: os.environ.get(m.group(1),m.group(0)),arg) for arg in spec.get('args',[])]
        if any('${' in arg for arg in spec['args']): raise ValueError('Unresolved MCP environment variable')
        spec['cwd']=str(ROOT)
    return cfg

async def aload_mcp_tools():
    cfg=load_config()
    client=MultiServerMCPClient(cfg)
    return await client.get_tools(),cfg

def load_mcp_tools():
    return asyncio.run(aload_mcp_tools())
