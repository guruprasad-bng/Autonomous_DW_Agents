
"""LangGraph coordinator for the Data Agents capstone.

Coordinates existing ReAct agents and records explicit handoffs.
Routing is deterministic; specialist agents remain LLM-powered.
"""

from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END


class CoordinatorState(TypedDict):
    question: str
    plan: list[str]
    step_index: int
    results: dict[str, str]
    handoffs: list[dict[str, Any]]
    final_answer: str


class CoordinatorAgent:
    def __init__(self, specialist_agent):
        # Reuse the existing ChatAgent instance and its MCP tools.
        self.specialist_agent = specialist_agent

        builder = StateGraph(CoordinatorState)

        builder.add_node("plan", self._plan)
        builder.add_node("Schema Drift", self._schema_drift)
        builder.add_node("Impact", self._impact)
        builder.add_node("Discovery", self._discovery)
        builder.add_node("summarize", self._summarize)

        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan", self._next_step
        )

        for name in ("Schema Drift", "Impact", "Discovery"):
            builder.add_conditional_edges(
                name, self._next_step
            )

        builder.add_edge("summarize", END)
        self.graph = builder.compile()

    def _build_plan(self, question: str) -> list[str]:
        q = question.lower()

        drift_terms = (
            "schema", "drift", "new column", "new field",
            "datatype", "data type", "schema change"
        )
        impact_terms = (
            "impact", "downstream", "dependency",
            "dependencies", "lineage", "affected"
        )
        discovery_terms = (
            "discover", "recommend", "data product",
            "glossary", "find relevant", "search catalog"
        )

        plan = []

        if any(term in q for term in drift_terms):
            plan.append("Schema Drift")

        if any(term in q for term in impact_terms):
            plan.append("Impact")

        if any(term in q for term in discovery_terms):
            plan.append("Discovery")

        # Default to Discovery for an unclassified request.
        return plan or ["Discovery"]

    def plan_query(self, question: str) -> list[str]:
        """Shared routing decision for UI and coordinator; one source of truth."""
        return self._build_plan(question)

    def _plan(self, state: CoordinatorState) -> dict:
        plan = self.plan_query(state["question"])

        return {
            "plan": plan,
            "step_index": 0,
            "results": {},
            "handoffs": [{
                "from": "User",
                "to": "Coordinator",
                "message": state["question"],
                "status": "received",
            }, {
                "from": "Coordinator",
                "to": plan[0],
                "message": (
                    f"Delegating first task. Execution plan: "
                    f"{' -> '.join(plan)}"
                ),
                "status": "delegated",
            }],
        }

    def _next_step(self, state: CoordinatorState) -> str:
        index = state["step_index"]
        plan = state["plan"]

        if index < len(plan):
            return plan[index]

        return "summarize"

    def _run_specialist(
        self, state: CoordinatorState, name: str
    ) -> dict:
        previous_results = state["results"]

        # Pass the original request and prior agents' findings.
        task = (
            f"Original user request:\n{state['question']}\n\n"
            f"You are the {name} specialist.\n"
            "Address only your specialist responsibilities.\n"
        )

        if previous_results:
            task += "\nFindings handed off by previous agents:\n"
            for agent_name, result in previous_results.items():
                task += f"\n[{agent_name}]\n{result[:6000]}\n"

        if name == "Schema Drift":
            task += (
                "\nDetect and explain schema differences. "
                "Only create a schema change proposal if the "
                "user explicitly requests one. Never approve "
                "or reject a proposal."
            )
        elif name == "Impact":
            task += (
                "\nIdentify downstream dependencies and impact. "
                "Use the previous findings as context, but "
                "verify dependencies through available tools."
            )
        elif name == "Discovery":
            task += (
                "\nFind relevant catalog assets, data products "
                "and glossary terms using available tools."
            )

        result = self.specialist_agent.ask(task, mode=name)

        answer = result.error or result.answer or "No answer returned."
        updated_results = dict(previous_results)
        updated_results[name] = answer

        handoffs = list(state["handoffs"])
        handoffs.append({
            "from": name,
            "to": "Coordinator",
            "message": answer,
            "status": "error" if result.error else "completed",
            "tool_calls": result.tool_calls,
            "latency_ms": result.latency_ms,
        })

        next_index = state["step_index"] + 1
        plan = state["plan"]

        if next_index < len(plan):
            next_name = plan[next_index]
            handoffs.append({
                "from": "Coordinator",
                "to": next_name,
                "message": (
                    f"Handing off the original request and "
                    f"findings from: {', '.join(updated_results)}"
                ),
                "status": "delegated",
            })

        return {
            "results": updated_results,
            "handoffs": handoffs,
            "step_index": next_index,
        }

    def _schema_drift(self, state: CoordinatorState) -> dict:
        return self._run_specialist(state, "Schema Drift")

    def _impact(self, state: CoordinatorState) -> dict:
        return self._run_specialist(state, "Impact")

    def _discovery(self, state: CoordinatorState) -> dict:
        return self._run_specialist(state, "Discovery")

    def _summarize(self, state: CoordinatorState) -> dict:
        sections = [
            "## Coordinator Execution Summary",
            f"**Agent sequence:** {' → '.join(state['plan'])}",
        ]

        for name in state["plan"]:
            sections.extend([
                f"### {name} Agent Findings",
                state["results"].get(name, "No findings."),
            ])

        sections.append(
            "\n**Workflow:** The coordinator passed findings "
            "between specialist agents. Any schema change "
            "still requires separate human approval."
        )

        return {"final_answer": "\n\n".join(sections)}

    def ask(self, question: str) -> dict:
        initial_state: CoordinatorState = {
            "question": question,
            "plan": [],
            "step_index": 0,
            "results": {},
            "handoffs": [],
            "final_answer": "",
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "answer": final_state["final_answer"],
            "plan": final_state["plan"],
            "results": final_state["results"],
            "handoffs": final_state["handoffs"],
        }