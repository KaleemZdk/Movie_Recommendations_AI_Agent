"""
agent.py — Cinephile LangGraph agent.

Loads tools from the local MCP server (mcp/server.py) and exposes:
    build_agent(model_name=None)                  → compiled LangGraph agent (cached per model)
    run_agent(message, thread_id, model_name)     → assistant reply text
    shutdown()                                    → close the checkpoint DB (call on app exit)

Run standalone for CLI testing:
    python agent.py
"""

import asyncio
import os
import sys
from pathlib import Path

import aiosqlite
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from groq import RateLimitError                                      # noqa: E402
from langchain_core.messages import SystemMessage                    # noqa: E402
from langchain_groq import ChatGroq                                  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient       # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver         # noqa: E402
from langgraph.graph import START, MessagesState, StateGraph         # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition             # noqa: E402


# Anchor every path to this file, so it works no matter where you launch it from.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

ALLOWED_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
}

MCP_SERVER_PATH = BASE_DIR / "mcp" / "server.py"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"


SYSTEM_PROMPT = """
You are Cinephile, the user's personal movie and TV show recommender.

PERSONALITY:
Speak like a knowledgeable, passionate cinephile and a friend who understands cinema.
Be conversational, insightful, and never pretentious.

RECOMMENDATIONS:
Recommend titles based on the user's taste, considering genre, atmosphere, themes,
pacing, storytelling, and emotional impact. Explain briefly why each recommendation
fits. Favor quality over quantity and introduce hidden gems when appropriate.

PERSONALIZATION:
Use available conversation history, favorites, watching lists, and reviews to personalize
recommendations. Ask follow-up questions only when necessary.

TOOLS:
- search_movies(query): Search TMDB for titles.
- get_movie_details(tmdb_id): Full details for one movie.
- get_similar_movies(tmdb_id), get_recommended_movies(tmdb_id): "More like this".
- get_trending_movies(window), discover_movies(...), list_genres(): Discovery.
- add_favorite(tmdb_id, title), list_favorites(), remove_favorite(tmdb_id).
- set_watching(tmdb_id, title), list_watching(), remove_watching(tmdb_id).
- save_review(tmdb_id, rating, comment), get_review(tmdb_id), list_reviews().
- get_user_profile(), recommend_for_user(limit): Personalized context.

Use tools when needed. Always resolve titles to a tmdb_id via search_movies before
saving favorites, watching entries, or reviews. Never invent results or claim an action
succeeded without confirmation. Treat similar-title results as candidates, not
guaranteed personalized recommendations.

RULES:
Avoid spoilers unless requested. Distinguish facts from opinions. Be honest when uncertain.
Answer directly, keep responses concise, and never recommend titles solely because
they are popular or highly rated.

Your goal is to understand the user's taste and help them discover films they'll love.
"""


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def get_llm(model_name: str | None = None) -> ChatGroq:
    # max_retries: let the client back off and retry on Groq's own 429s
    return ChatGroq(model=normalize_model_name(model_name), temperature=0.3, max_retries=4)


def extract_text(content) -> str:
    """Message content is a str for most models, but can be a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


# --------------------------------------------------------------------------- #
# AGENT — lazy singletons (tools + checkpointer shared, one graph per model)
# --------------------------------------------------------------------------- #
_lock = asyncio.Lock()
_tools = None
_checkpointer: AsyncSqliteSaver | None = None
_agents: dict[str, object] = {}


async def _load_mcp_tools():
    """Spawn the MCP server over stdio and return its tools."""
    client = MultiServerMCPClient(
        {
            "movies": {
                "command": sys.executable,
                "args": [str(MCP_SERVER_PATH)],
                "transport": "stdio",
                # A stdio server only gets a minimal environment by default, so it would
                # not see TMDB_API_KEY etc. from your .env. Pass the environment through.
                "env": dict(os.environ),
            }
        }
    )
    return await client.get_tools()


async def _get_checkpointer() -> AsyncSqliteSaver:
    # The graph is async (MCP tools are async-only), so it needs the ASYNC saver.
    # The sync SqliteSaver raises NotImplementedError on ainvoke/astream.
    global _checkpointer
    if _checkpointer is None:
        conn = await aiosqlite.connect(str(CHECKPOINT_DB))
        _checkpointer = AsyncSqliteSaver(conn)
        await _checkpointer.setup()
    return _checkpointer


async def build_agent(model_name: str | None = None):
    """Build (or return the cached) Cinephile graph for the chosen model."""
    global _tools

    model = normalize_model_name(model_name)

    if model in _agents:
        return _agents[model]

    async with _lock:
        if model in _agents:
            return _agents[model]

        if _tools is None:
            _tools = await _load_mcp_tools()

        llm = get_llm(model).bind_tools(_tools)

        # -------- node: call the LLM --------
        async def call_model(state: MessagesState):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response]}

        # -------- graph --------
        builder = StateGraph(MessagesState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", ToolNode(_tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")

        _agents[model] = builder.compile(checkpointer=await _get_checkpointer())
        return _agents[model]


async def shutdown() -> None:
    """Close the checkpoint database (call when your app exits)."""
    global _checkpointer
    if _checkpointer is not None:
        await _checkpointer.conn.close()
        _checkpointer = None


# --------------------------------------------------------------------------- #
# CONVENIENCE RUNNER — used by bridge.py /api/chat
# --------------------------------------------------------------------------- #
async def run_agent(message: str, thread_id: str = "default", model_name: str | None = None) -> str:
    """
    Invoke the agent for one turn.

    Args:
        message:    The user's message.
        thread_id:  Conversation id — used by the checkpointer to persist history.
        model_name: Optional model id (must be in ALLOWED_MODELS, else the default is used).

    Returns:
        The assistant's reply text.
    """
    agent = await build_agent(model_name)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
    except RateLimitError:
        return "I've hit the Groq rate limit. Give it a minute and try again."

    return extract_text(result["messages"][-1].content)


# --------------------------------------------------------------------------- #
# CLI — python agent.py
# --------------------------------------------------------------------------- #
async def _cli() -> None:
    print("Cinephile — type 'exit' to quit.\n")
    thread_id = "cli"

    try:
        while True:
            try:
                user = (await asyncio.to_thread(input, "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user.lower() in {"exit", "quit"}:
                break
            if not user:
                continue

            reply = await run_agent(user, thread_id)
            print(f"\nCinephile: {reply}\n")
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(_cli())