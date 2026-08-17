# LangGraph Dynamic Supervisor Example

A minimal example of a two-level dynamic agent architecture built with
[LangGraph](https://github.com/langchain-ai/langgraph):

- **Level 1 — Supervisor**: a ReAct agent whose "tools" are actually other
  agents. Given a user request, it reasons about *which agent* to delegate
  to, with no hardcoded keyword/if-else routing.
- **Level 2 — Specialized agents**: each one is its own `create_react_agent`
  loop that reasons about *which tool* to call from its own tool set.

Both levels use the same mechanism: give the model descriptions of its
options and let it choose at runtime. This mirrors patterns like
ServiceNow's Agentic Workflow, and shows what LangChain's
`create_react_agent` / LangGraph's `create_supervisor` helper do under the
hood.

## Dynamic agent/tool loading

Nothing in [langgraph_dynamic_supervisor.py](langgraph_dynamic_supervisor.py)
hardcodes which agents or tools exist — it discovers them at runtime:

```
tools/
  support_tools.py       @tool-decorated functions for the support agent
  data_tools.py            @tool-decorated functions for the data agent
  web_search_tools.py      @tool-decorated functions for the web search agent
agents/
  support_agent.yaml      role: worker, prompt, tools list, delegate description
  data_agent.yaml          role: worker, prompt, tools list, delegate description
  web_search_agent.yaml    role: worker, prompt, tools list, delegate description
  supervisor.yaml           role: supervisor, prompt, list of agents it delegates to
```

- `load_tool_registry()` imports every module under `tools/` and collects
  every `@tool`-decorated function into a `name -> tool` map.
- `load_agent_configs()` reads every `agents/*.yaml` into a `name -> config`
  map.
- `build_graph()` builds a worker agent for each `role: worker` config
  (looking up its `tools:` list in the registry), wraps each worker as a
  `delegate_to_<name>` tool (using that worker's own `delegate_description`),
  and builds the single `role: supervisor` config from its `delegates:` list.

To add a new agent: drop a new `tools/*.py` file with `@tool` functions,
a new `agents/*.yaml` (`role: worker`), and add its name to
`supervisor.yaml`'s `delegates:` list. No edits to the main script needed.

## Tools: stubs vs. real APIs

- `support_tools.py` (`search_knowledge_base`, `get_ticket_status`) and
  `data_tools.py` (`run_sql_query`, `generate_chart`) are **stubs** that
  return canned strings — the point is to demonstrate the delegation
  architecture, not real integrations.
- `web_search_tools.py` makes **real HTTP calls** to free, keyless services:
  - `search_web` — Wikipedia's MediaWiki Action API (`list=search`) to find
    the best-matching page, then Wikipedia's REST summary API to fetch its
    opening extract. Covers any topic with a Wikipedia article; not a full
    web index, so very recent events or niche topics may come back empty.
  - `geocode_address` — OpenStreetMap's Nominatim API, a free stand-in for
    the Google Maps Geocoding API (which requires a billed Google Cloud
    key).
  - `find_nearby_places` — OpenStreetMap's Overpass API, a free stand-in
    for the Google Maps Places "nearby search" API. It maps common
    categories (coffee, restaurant, pharmacy, hotel, etc. — see
    `CATEGORY_TAGS` in the file) to OpenStreetMap tags, or falls back to
    matching the query against place names.

These free public endpoints are shared, rate-limited, and occasionally
return timeouts (504) or throttling errors under load — that's the
tradeoff of not needing an API key. Each tool catches request failures and
returns an `[error] ...` string rather than crashing, so the agent can
explain the failure (and typically falls back on its own knowledge) rather
than the whole run failing.

If you want authoritative, reliable results instead, swap these for the
real Google Maps Platform APIs (Geocoding/Places) and a paid search API
(e.g. Tavily, Google Custom Search) — that requires API keys and, for
Google Maps, a billed Google Cloud project.

## Requirements

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com) (API Keys section)
- Internet access (the web search agent makes live HTTP calls)

## Setup

```bash
cd /Users/rongmingsun/Documents/code_local/LangGraph_code
python3 -m venv .venv
source .venv/bin/activate
pip install langgraph langchain_anthropic pyyaml requests
```

Create a `.env` file in this folder with your API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Run

```bash
cd /Users/rongmingsun/Documents/code_local/LangGraph_code
source .venv/bin/activate
set -a && source .env && set +a
python langgraph_dynamic_supervisor.py
```

The script prompts for a request at a `human:` prompt, runs it through the
supervisor graph, and prints which agent(s) it delegated to plus the final
answer. Type `quit` (or press Ctrl-D) to exit.

### Example inputs to try

- `My ticket #4521 hasn't been updated, can you check?` — routed to the
  support agent.
- `Please query the database by SQL to get last month's tickets` — routed
  to the data agent.
- `Check ticket #4521, and also chart last month's ticket volume by
  category.` — routed to *both* agents, with results combined into one
  answer.
- `Find the latitude/longitude of 2779 Huff Dr, Pleasanton, CA 94588` —
  routed to the web search agent's `geocode_address` tool.
- `Find coffee shops near the Space Needle in Seattle.` — routed to the
  web search agent's `find_nearby_places` tool.
- `What is LangGraph?` — routed to the web search agent's `search_web`
  tool (Wikipedia).
