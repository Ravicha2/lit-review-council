import sys
import os
import pytest

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

@pytest.mark.asyncio
async def test_mcp_server_has_literature_review_tool():
    from src.mcp_server import mcp
    tools = await mcp.list_tools()
    
    # FastMCP list_tools returns a list of Tool objects
    assert any(t.name == "conduct_literature_review" for t in tools)
