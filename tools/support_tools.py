"""Tools available to the support agent (agents/support_agent.yaml)."""

from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """Search internal docs/knowledge base for an answer."""
    return f"[stub] KB result for: {query}"


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up the status of a support ticket by ID."""
    return f"[stub] Ticket {ticket_id} status: In Progress"
