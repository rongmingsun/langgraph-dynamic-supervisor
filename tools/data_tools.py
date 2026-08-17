"""Tools available to the data analyst agent (agents/data_agent.yaml)."""

from langchain_core.tools import tool


@tool
def run_sql_query(query: str) -> str:
    """Run a read-only SQL query against the reporting database."""
    return f"[stub] Query result for: {query}"


@tool
def generate_chart(data_description: str) -> str:
    """Generate a chart from a described dataset."""
    return f"[stub] Chart generated for: {data_description}"
