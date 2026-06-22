import os
import sys
import logging

# ============================================================================
# CRITICAL: Protect the MCP stdio transport.
#
# FastMCP uses stdin/stdout for JSON-RPC via stdio_server(), which grabs
# sys.stdout.buffer at *run time* (not import time). We redirect sys.stdout
# to sys.stderr in main() to prevent print()/ANSI from deps from corrupting
# the transport, but pass the real stdout buffer explicitly to stdio_server().
# ============================================================================

os.environ["NO_COLOR"] = "1"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging to stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("lit-review-council")

# Validate required env vars on startup.
_REQUIRED_ENV = ["OPENROUTER_API_KEY", "GITHUB_TOKEN", "TAVILY_API_KEY"]
_OPTIONAL_ENV = ["ENG_MODEL", "RESEARCH_MODEL", "JUDGE_MODEL", "OPENALEX_API_KEY", "MAX_SOURCES"]
_missing = [v for v in _REQUIRED_ENV if not os.getenv(v)]
if _missing:
    logger.error("Missing required env vars: %s. Set them in your MCP client config.", ", ".join(_missing))
for _var in _OPTIONAL_ENV:
    _val = os.getenv(_var)
    logger.info("Env var %s = %s", _var, _val if _val else "(default)")
logger.info("OPENROUTER_API_KEY is %s", "set" if os.getenv("OPENROUTER_API_KEY") else "MISSING")

# Force-suppress LiteLLM's verbose logging after transitive import.
from src.orchestration import orchestrate, Config, TopicConfig
from src.skill_prompt import SKILL_MARKDOWN

try:
    import litellm
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
except ImportError:
    pass

mcp = FastMCP("lit-review-council")

@mcp.prompt()
def lit_review_council_instructions() -> str:
    """Get the instructions and guidelines for using the lit-review-council orchestrator."""
    return SKILL_MARKDOWN

@mcp.tool()
async def conduct_literature_review(question: str, topics: list[dict], output_dir: str = ".") -> str:
    """
    Executes a multi-agent literature review on the provided research question and topics.
    
    Args:
        question: The overarching research question driving the literature review.
        topics: A list of topic dictionaries. Each topic must have:
            - slug: (str) a short hyphenated identifier
            - description: (str) explanation of the topic
            - search_keywords: (list[str]) 2-4 highly specific search queries
            - rationale: (str, optional) why this topic was chosen
        output_dir: (str) Directory to save the final OKF markdown bundle. Defaults to current directory.
        
    Returns:
        A success message with the path to the generated OKF bundle directory.
    """
    # Convert dicts to TopicConfig objects
    topic_configs = []
    for t in topics:
        topic_configs.append(TopicConfig(
            slug=t["slug"],
            description=t["description"],
            search_keywords=t["search_keywords"],
            rationale=t.get("rationale")
        ))
        
    config = Config(topics=topic_configs)
    
    logger.info("Starting literature review for: %s", question)
    logger.info("Output directory: %s", output_dir)
    
    await orchestrate(
        config=config,
        output_dir=output_dir,
        research_question=question
    )
    
    index_path = os.path.join(output_dir, "index.md")
    return f"Literature review completed successfully. The OKF bundle has been saved to {output_dir}. Start reading at {index_path}."

def main():
    """Run the MCP server with stdout safely redirected to stderr."""
    import anyio
    from io import TextIOWrapper
    from mcp.server.stdio import stdio_server

    # Capture the real stdout BEFORE redirecting.
    real_stdout = anyio.wrap_file(
        TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    )
    real_stdin = anyio.wrap_file(
        TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    )

    # NOW redirect stdout → stderr so all print() from deps is safe.
    sys.stdout = sys.stderr

    async def _run():
        async with stdio_server(stdin=real_stdin, stdout=real_stdout) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )

    anyio.run(_run)

if __name__ == "__main__":
    main()
