"""Data Agents capstone Streamlit interface."""
import streamlit as st
import json
from app.chat_agent import ChatAgent
from app.mcp_clients import load_config, load_mcp_tools
from pathlib import Path
import asyncio


st.set_page_config(page_title='Data Agents Capstone',page_icon='🗂️',layout='wide')
st.title('Data Agents — Enterprise Metadata Assistant')
st.caption('Local metadata catalog + ChromaDB semantic RAG via MCP; no production catalog connection')

@st.cache_resource(show_spinner='Connecting to metadata MCP server...')
def get_agent(): return ChatAgent()

with st.sidebar:
    mode=st.radio('Choose an agent',['Discovery','Impact','Schema Drift'])
    st.write('MCP servers:',', '.join(load_config().keys()))
    if st.button('Clear chat'):
        st.session_state.history_by_mode[mode] = []
        st.cache_resource.clear()
        st.rerun()

try: agent=get_agent()
except Exception as exc:
    st.error(f'Initialization failed: {exc}');st.stop()

examples={
 'Discovery':'I need a unified borrower view for loan servicing. Find relevant data products and explain the business glossary terms.',
 'Impact':'Analyze downstream impact if loan_accounts.balance changes.',
 'Schema Drift':'Check schema drift for loan_accounts using this incoming schema: {"loan_id":"string","customer_id":"string","balance":"string","status":"string","currency":"string"}'
}
st.info('Example: '+examples[mode])

if "history_by_mode" not in st.session_state:
    st.session_state.history_by_mode = {
        "Discovery": [],
        "Impact": [],
        "Schema Drift": []
    }

history = st.session_state.history_by_mode[mode]
for h in history:
    with st.chat_message(h['role']):st.markdown(h['content'])
if question:=st.chat_input('Ask the selected metadata agent...'):
    history.append({'role':'user','content':question})
    with st.chat_message('user'):st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Analyzing metadata...'):result=agent.ask(question,mode)
        response=result.error or result.answer
        st.markdown(response)
        retrievals=[c for c in result.tool_calls if c['tool'] in ('semantic_search','search_glossary')]
        if retrievals:
            with st.expander('RAG retrieval evidence (ChromaDB)', expanded=True):
                for call in retrievals:
                    st.write('Tool:',call['tool'])
                    try:
                        payload=json.loads(call['output']) if isinstance(call['output'],str) else call['output']
                        st.json(payload)
                    except (ValueError,TypeError): st.code(str(call['output'])[:5000])
        with st.expander(f'Tool trace: {len(result.tool_calls)} calls · {result.latency_ms} ms'):
            for call in result.tool_calls:
                st.write(call['tool']);st.json(call['input']);st.code(str(call['output'])[:5000])
    history.append({'role':'assistant','content':response})    

# =====================================================
# Schema Drift - Human-in-the-Loop Approval Queue
# =====================================================


CATALOG_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "catalog.json"
)


def get_pending_requests():
    if not CATALOG_PATH.exists():
        return []

    catalog = json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )

    return [
        request
        for request in catalog.get("schema_change_requests", [])
        if request.get("status") == "PENDING"
    ]


def process_approval(request_id, decision, reviewer):
    # Access the unfiltered MCP tools from the trusted UI.
    tools, _ = load_mcp_tools()

    approval_tool = next(
        t for t in tools
        if t.name == "resolve_schema_change"
    )

    return asyncio.run(
        approval_tool.ainvoke({
            "request_id": request_id,
            "decision": decision,
            "reviewer": reviewer
        })
    )


# =====================================================
# Schema Drift - Human-in-the-Loop Approval Queue
# Display ONLY when Schema Drift mode is selected
# =====================================================

if mode == "Schema Drift":

    st.divider()
    st.subheader("Schema Drift — Human Approval Queue")

    pending_requests = get_pending_requests()

    if not pending_requests:
        st.info("No schema changes awaiting approval.")

    else:
        reviewer = st.text_input(
            "Reviewer Name",
            placeholder="Enter your name"
        )

        for request in pending_requests:

            with st.container(border=True):

                st.write("**Asset ID:**", request["asset_id"])
                st.write("**Request ID:**", request["request_id"])
                st.write("**Severity:**", request["severity"])

                st.write("### Proposed Documentation")
                st.json(request["documentation"])

                with st.expander("Compare Schemas"):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Current Schema")
                        st.json(request["registered_schema"])

                    with col2:
                        st.write("Proposed Schema")
                        st.json(request["incoming_schema"])

                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "Approve",
                        key=f"approve_{request['request_id']}",
                        disabled=not reviewer.strip()
                    ):
                        try:
                            result = process_approval(
                                request["request_id"],
                                "APPROVE",
                                reviewer.strip()
                            )

                            st.write(result)
                            st.rerun()

                        except Exception as e:
                            st.error(f"Approval failed: {e}")

                with col2:
                    if st.button(
                        "Reject",
                        key=f"reject_{request['request_id']}",
                        disabled=not reviewer.strip()
                    ):
                        try:
                            result = process_approval(
                                request["request_id"],
                                "REJECT",
                                reviewer.strip()
                            )

                            st.write(result)
                            st.rerun()

                        except Exception as e:
                            st.error(f"Rejection failed: {e}")
