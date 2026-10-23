"""Data Agents capstone Streamlit interface."""

import asyncio
import json
from pathlib import Path

import streamlit as st

from app.chat_agent import ChatAgent
from app.coordinator_agent import CoordinatorAgent
from app.mcp_clients import load_config, load_mcp_tools
from app.guardrails import validate_question


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
UI_MODES = ("Automatic", "Discovery", "Impact Analysis", "Schema Drift")
SPECIALIST_MODES = {"Discovery": "Discovery", "Impact Analysis": "Impact", "Schema Drift": "Schema Drift"}

st.set_page_config(
    page_title="Data Agents Capstone",
    page_icon="🗂️",
    layout="wide",
)
st.title("Autonomous Data Warehouse agents")
st.caption(
    "Local metadata catalog + ChromaDB semantic RAG via MCP; "
    "no production catalog connection"
)


@st.cache_resource(show_spinner="Connecting to metadata MCP server...")
def get_agent():
    return ChatAgent()


@st.cache_resource(show_spinner="Initializing coordinator...")
def get_coordinator():
    return CoordinatorAgent(get_agent())


def get_pending_requests():
    if not CATALOG_PATH.exists():
        return []
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [
        request
        for request in catalog.get("schema_change_requests", [])
        if request.get("status") == "PENDING"
    ]


def process_approval(request_id, decision, reviewer):
    # Demo-only trusted UI path. Production requires authenticated authorization.
    tools, _ = load_mcp_tools()
    approval_tool = next(
        (tool for tool in tools if tool.name == "resolve_schema_change"),
        None,
    )
    if approval_tool is None:
        raise RuntimeError("MCP tool resolve_schema_change is unavailable")

    result = asyncio.run(
        approval_tool.ainvoke(
            {
                "request_id": request_id,
                "decision": decision,
                "reviewer": reviewer,
            }
        )
    )
    # MCP tools may return a string, dictionary, or a structured tool message.
    if isinstance(result, dict) and (result.get("error") or result.get("isError")):
        raise RuntimeError(str(result))
    if getattr(result, "isError", False):
        raise RuntimeError(str(result))
    return result


def render_handoff_trace(handoffs, plan):
    st.subheader("Agent-to-Agent Handoff Trace")
    st.write("**Execution plan:**", " → ".join(plan))
    for index, handoff in enumerate(handoffs, start=1):
        with st.expander(
            f"{index}. {handoff.get('from', '?')} → {handoff.get('to', '?')}",
            expanded=True,
        ):
            st.write("**Status:**", handoff.get("status", "unknown"))
            st.write("**Context transferred:**")
            st.text(str(handoff.get("message", "")))
            for call in handoff.get("tool_calls", []) or []:
                st.write("**MCP tool:**", call.get("tool", "unknown"))
                st.json(call.get("input", {}))
                st.code(str(call.get("output", ""))[:3000])
            if handoff.get("latency_ms") is not None:
                st.caption(f"Execution time: {handoff['latency_ms']} ms")


def render_specialist_trace(tool_calls, latency_ms):
    retrievals = [
        call for call in tool_calls
        if call.get("tool") in ("semantic_search", "search_glossary")
    ]
    if retrievals:
        with st.expander("RAG retrieval evidence (ChromaDB)", expanded=True):
            for call in retrievals:
                st.write("Tool:", call.get("tool"))
                output = call.get("output", "")
                try:
                    payload = json.loads(output) if isinstance(output, str) else output
                    st.json(payload)
                except (ValueError, TypeError):
                    st.code(str(output)[:5000])

    with st.expander(f"Tool trace: {len(tool_calls)} calls · {latency_ms} ms"):
        for call in tool_calls:
            st.write(call.get("tool", "unknown"))
            st.json(call.get("input", {}))
            st.code(str(call.get("output", ""))[:5000])


# Keep separate UI histories for automatic and direct specialist conversations.
if "history_by_selection" not in st.session_state:
    st.session_state.history_by_selection = {mode: [] for mode in UI_MODES}
    # Preserve the prior automatic-mode conversation after upgrading the UI.
    st.session_state.history_by_selection["Automatic"] = list(
        st.session_state.get("auto_history", [])
    )
for selection in UI_MODES:
    st.session_state.history_by_selection.setdefault(selection, [])

with st.sidebar:
    selected_mode = st.radio("Agent selection", UI_MODES, index=0)
    st.caption(
        "Automatic routes queries to one specialist or invokes the "
        "internal Coordinator when multiple agents are required."
        if selected_mode == "Automatic"
        else f"Directly invokes the {selected_mode} agent."
    )
    st.write("MCP servers:", ", ".join(load_config().keys()))
    if st.button("Clear chat"):
        st.session_state.history_by_selection[selected_mode] = []
        st.cache_resource.clear()
        st.rerun()

try:
    agent = get_agent()
    coordinator = get_coordinator()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

with st.expander("Example questions", expanded=False):
    st.markdown("""- **Discovery:** Find relevant data products for a unified borrower view and explain glossary terms.
- **Impact:** Analyze downstream impact if `loan_accounts.balance` changes.
- **Schema Drift:** Check schema drift for `loan_accounts` using an incoming schema.
- **Multi-agent:** A new field `risk_category` is being introduced in `loan_accounts`. Detect schema drift, analyze downstream impact, and recommend related data products.
""")

history = st.session_state.history_by_selection[selected_mode]

# Re-display the complete response, including traces, after Streamlit reruns.
for entry in history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant":
            if entry.get("handoffs") is not None:
                render_handoff_trace(entry["handoffs"], entry.get("plan", []))
            elif entry.get("tool_calls") is not None:
                render_specialist_trace(
                    entry["tool_calls"], entry.get("latency_ms", 0)
                )

if question := st.chat_input("Ask about enterprise metadata..."):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # Validate the original request before routing or calling any agent.
            clean_question = validate_question(question)
            history.append({"role": "user", "content": clean_question})
            if selected_mode == "Automatic":
                plan = coordinator.plan_query(clean_question)
                use_coordinator = len(plan) > 1
                specialist_mode = plan[0] if not use_coordinator else None
            else:
                use_coordinator = False
                specialist_mode = SPECIALIST_MODES[selected_mode]
            with st.spinner("Analyzing metadata..."):
                if use_coordinator:
                    coordinator_result = coordinator.ask(clean_question)
                    response = coordinator_result["answer"]
                else:
                    result = agent.ask(clean_question, mode=specialist_mode)
                    response = result.error or result.answer or "No answer returned."

            st.markdown(response)
            assistant_entry = {"role": "assistant", "content": response}
            if use_coordinator:
                assistant_entry["plan"] = coordinator_result["plan"]
                assistant_entry["handoffs"] = coordinator_result["handoffs"]
                render_handoff_trace(
                    coordinator_result["handoffs"], coordinator_result["plan"]
                )
            else:
                assistant_entry["tool_calls"] = result.tool_calls
                assistant_entry["latency_ms"] = result.latency_ms
                render_specialist_trace(result.tool_calls, result.latency_ms)
            history.append(assistant_entry)
        except Exception as exc:
            st.error(f"Agent execution failed: {exc}")
            history.append({"role": "assistant", "content": f"Agent execution failed: {exc}"})

# Approval queue is accessible regardless of which specialist was invoked.
st.divider()
st.subheader("Schema Drift — Human Approval Queue")
try:
    pending_requests = get_pending_requests()
except (OSError, ValueError, TypeError) as exc:
    st.error(f"Unable to read approval queue: {exc}")
    pending_requests = []

if not pending_requests:
    st.info("No schema changes awaiting approval.")
else:
    reviewer = st.text_input(
        "Reviewer Name", placeholder="Enter your name"
    )
    for request in pending_requests:
        request_id = request.get("request_id", "")
        with st.container(border=True):
            st.write("**Asset ID:**", request.get("asset_id", ""))
            st.write("**Request ID:**", request_id)
            st.write("**Severity:**", request.get("severity", ""))
            st.write("### Proposed Documentation")
            st.json(request.get("documentation", {}))

            with st.expander("Compare Schemas"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Current Schema")
                    st.json(request.get("registered_schema", {}))
                with col2:
                    st.write("Proposed Schema")
                    st.json(request.get("incoming_schema", {}))

            col1, col2 = st.columns(2)
            with col1:
                approve = st.button(
                    "Approve",
                    key=f"approve_{request_id}",
                    disabled=not reviewer.strip(),
                )
            with col2:
                reject = st.button(
                    "Reject",
                    key=f"reject_{request_id}",
                    disabled=not reviewer.strip(),
                )

            if approve or reject:
                decision = "APPROVE" if approve else "REJECT"
                try:
                    outcome = process_approval(
                        request_id, decision, reviewer.strip()
                    )
                    st.session_state["last_approval_result"] = (
                        f"{decision} request {request_id}: {outcome}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"{decision} failed: {exc}")

if st.session_state.get("last_approval_result"):
    st.success(st.session_state.pop("last_approval_result"))
