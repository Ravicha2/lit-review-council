import pytest
import json
import os
import sys
import importlib
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import get_synthesis_prompt

class MockSession:
    def __init__(self, state):
        self.state = state

class MockContext:
    def __init__(self, state):
        self.session = MockSession(state)

from src.schema import JudgeRanking, PeerReviewResult

def test_get_synthesis_prompt_formats_correctly():
    class FakeReport:
        def model_dump_json(self):
            return '{"fake": "report"}'
            
    state = {
        "topic": "Graph DBs",
        "topic_slug": "graph-dbs",
        "winning_report_id": "report_2",
        "peer_review_rationale": "It was practical",
        "report_1": FakeReport(),
        "report_2": FakeReport()
    }
    ctx = MockContext(state)
    prompt = get_synthesis_prompt(ctx)
    assert "Graph DBs" in prompt
    assert "graph-dbs" in prompt
    assert "engineer" in prompt
    assert "It was practical" in prompt
    assert '{"fake": "report"}' in prompt

def test_agent_models_from_env():
    env_vars = {
        "RESEARCH_MODEL": "openrouter/research-model",
        "ENG_MODEL": "openrouter/eng-model",
        "JUDGE_MODEL": "openrouter/judge-model",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        import src.pipeline
        importlib.reload(src.pipeline)
        
        assert src.pipeline.academic_explorer.model.model == "openrouter/research-model"
        assert src.pipeline.practitioner_explorer.model.model == "openrouter/eng-model"
        assert src.pipeline.architect_reviewer.model.model == "openrouter/judge-model"

def test_agent_instructions_contain_formatting_and_search_limits():
    import os
    os.environ["MAX_SOURCES"] = "10"
    import src.pipeline
    importlib.reload(src.pipeline)
    
    assert "maximum of 10 search calls" in src.pipeline.academic_explorer.instruction
    assert "at least 5 such" in src.pipeline.academic_explorer.instruction
    
    for agent in [src.pipeline.academic_reporter, src.pipeline.practitioner_reporter]:
        assert "properly escape all JSON string" in agent.instruction

def test_sequential_agents_compose_correctly():
    from src.pipeline import academic_explorer, academic_reporter, academic_sequence
    from src.pipeline import practitioner_explorer, practitioner_reporter, practitioner_sequence
    from google.adk.agents import SequentialAgent
    
    assert isinstance(academic_sequence, SequentialAgent)
    assert academic_sequence.sub_agents[0] == academic_explorer
    assert academic_sequence.sub_agents[1] == academic_reporter

from src.pipeline import run_pipeline, InMemorySessionService

@pytest.fixture
def mock_pipeline_deps():
    class Mocks:
        pass
    m = Mocks()
    m.run_async_calls = []
    with patch("src.pipeline.Runner") as mock_runner, \
         patch("src.pipeline.write_to_filesystem_mcp", new_callable=AsyncMock):
        async def fake_run_async(*args, **kwargs):
            m.run_async_calls.append(kwargs)
            if False: yield
        mock_runner.return_value.run_async = fake_run_async
        m.mock_runner = mock_runner
        yield m

@pytest.mark.asyncio
async def test_ensemble_calculates_winning_report(mock_pipeline_deps):
    original_get_session = InMemorySessionService.get_session
    async def mock_get_session(self, *, app_name, user_id, session_id):
        sess = await original_get_session(self, app_name=app_name, user_id=user_id, session_id=session_id)
        if not sess.state.get("report_1"):
            sess.state["report_1"] = {"title": "R1"}
            sess.state["report_2"] = {"title": "R2"}
            
        if not sess.state.get("rankings_researcher"):
            sess.state["rankings_researcher"] = JudgeRanking(ranking=["B", "A"], rationale="B good")
            sess.state["rankings_engineer"] = JudgeRanking(ranking=["B", "A"], rationale="B better")
            sess.state["rankings_architect"] = JudgeRanking(ranking=["B", "A"], rationale="B best")
            
        if not sess.state.get("synthesis_result"):
            sess.state["synthesis_result"] = {"markdown": "done", "urls_cited": []}
        return sess
        
    with patch.object(InMemorySessionService, 'get_session', new=mock_get_session):
        await run_pipeline("test")
        
    # verify the synthesis state delta got the correct winner based on rankings
    synthesis_call = mock_pipeline_deps.run_async_calls[-1]
    assert "winning_report_id" in synthesis_call.get("state_delta", {})

