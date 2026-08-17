"""
LangGraph example: two-level dynamic reasoning (agent selection + tool selection)

This mirrors ServiceNow's Agentic Workflow architecture:

Level 1 - Supervisor (the "agentic workflow"):
    An LLM given a role + instructions + descriptions of the available
    agents. It reasons, per request, about WHICH AGENT to invoke.
    This is dynamic - no hardcoded keyword/if-else routing.

Level 2 - Each specialized agent (create_react_agent):
    An LLM ReAct loop that reasons about WHICH TOOL to invoke, given
    its own tool set. Also dynamic.

Both levels use the same underlying mechanism: give the model a
description of its options (agents, or tools) and let it choose.
LangChain's `create_react_agent` + LangGraph's `create_supervisor`
helper (from the `langgraph-supervisor` package) implement exactly
this pattern; the manual version below shows what's happening under
the hood without that extra dependency.

Agents and tools are not hardcoded here - they're loaded dynamically:
  - tools/*.py       modules of @tool-decorated functions
  - agents/*.yaml     one file per agent (role: worker or supervisor)
Adding a new agent or tool means adding a file, not editing this one.
"""

import warnings

import importlib
import pkgutil
from pathlib import Path

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool, StructuredTool

# langgraph's own jsonplus.py creates a langchain_core `Reviver()` at import
# time (triggered below by `from langgraph.prebuilt import ...`) without an
# explicit `allowed_objects`, which raises this pending deprecation warning
# from inside the library, not from anything in this file. Importing
# langchain_core above installs its own "always show" filter for this
# warning, so ours must be added afterward to take priority.
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

from langgraph.prebuilt import create_react_agent

BASE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = BASE_DIR / "agents"
TOOLS_PACKAGE = "tools"

llm = ChatAnthropic(model="claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# Dynamic loading: tools
#    Import every module under tools/ and collect the @tool-decorated
#    functions it defines into a name -> BaseTool registry, so agent
#    configs can reference tools by name without this file hardcoding them.
# ---------------------------------------------------------------------------
def load_tool_registry() -> dict[str, BaseTool]:
    registry: dict[str, BaseTool] = {}
    package = importlib.import_module(TOOLS_PACKAGE)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, prefix=f"{TOOLS_PACKAGE}."):
        module = importlib.import_module(module_name)
        for obj in vars(module).values():
            if isinstance(obj, BaseTool):
                registry[obj.name] = obj
    return registry


# ---------------------------------------------------------------------------
# Dynamic loading: agent configs
#    Read every agents/*.yaml file into a name -> config dict.
# ---------------------------------------------------------------------------
def load_agent_configs() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for path in sorted(AGENTS_DIR.glob("*.yaml")):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        configs[cfg["name"]] = cfg
    return configs


# ---------------------------------------------------------------------------
# Level 2 - Specialized (worker) agents: dynamic TOOL selection.
#    Each agent decides, turn by turn, which of its own tools to call.
# ---------------------------------------------------------------------------
def build_worker_agent(cfg: dict, tool_registry: dict[str, BaseTool]):
    tools = [tool_registry[name] for name in cfg["tools"]]
    return create_react_agent(llm, tools=tools, prompt=cfg["prompt"])


# ---------------------------------------------------------------------------
# Expose each worker agent AS A TOOL to the supervisor.
#    This is the key structural piece: from the supervisor's point of
#    view, "call the support agent" looks just like calling any other
#    tool. The supervisor doesn't know or care that the tool it's
#    picking is itself a whole ReAct loop with its own tools.
# ---------------------------------------------------------------------------
def make_delegate_tool(agent_name: str, agent, description: str) -> BaseTool:
    def _delegate(request: str) -> str:
        result = agent.invoke({"messages": [("user", request)]})
        return result["messages"][-1].content

    return StructuredTool.from_function(
        func=_delegate,
        name=f"delegate_to_{agent_name}",
        description=description,
    )


# ---------------------------------------------------------------------------
# Level 1 - Supervisor: dynamic AGENT selection.
#    Built from agents/supervisor.yaml's `delegates` list - the LLM still
#    decides at runtime which delegate tool (i.e. which agent) to call;
#    only the wiring of "which agents exist" is now data, not code.
# ---------------------------------------------------------------------------
def build_graph():
    tool_registry = load_tool_registry()
    agent_configs = load_agent_configs()

    worker_cfgs = {name: cfg for name, cfg in agent_configs.items() if cfg["role"] == "worker"}
    supervisor_cfgs = [cfg for cfg in agent_configs.values() if cfg["role"] == "supervisor"]
    if len(supervisor_cfgs) != 1:
        raise ValueError(
            f"Expected exactly one agents/*.yaml with role: supervisor, found {len(supervisor_cfgs)}"
        )
    supervisor_cfg = supervisor_cfgs[0]

    worker_agents = {name: build_worker_agent(cfg, tool_registry) for name, cfg in worker_cfgs.items()}

    delegate_tools = [
        make_delegate_tool(name, worker_agents[name], worker_cfgs[name]["delegate_description"])
        for name in supervisor_cfg["delegates"]
    ]

    # The supervisor's internal graph already IS the full system - no extra
    # StateGraph wiring needed, since delegation happens through tool calls.
    return create_react_agent(llm, tools=delegate_tools, prompt=supervisor_cfg["prompt"])


graph = build_graph()


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
def print_transcript(result: dict) -> None:
    """Print the human question, which agent(s) got delegated to, and the
    final synthesized answer - skipping each delegate's raw tool output,
    since the supervisor's final answer already restates it."""
    for m in result["messages"]:
        if m.type == "tool":
            continue
        if m.type == "ai" and getattr(m, "tool_calls", None):
            names = ", ".join(tc["name"] for tc in m.tool_calls)
            print(f"ai : (delegating to {names})")
            continue
        print(m.type, ":", m.content)


if __name__ == "__main__":
    print("LangGraph dynamic supervisor demo. Type a request (or 'quit' to exit).")
    while True:
        try:
            user_input = input("human: ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        result = graph.invoke({"messages": [("user", user_input)]})
        print_transcript(result)
        print()
