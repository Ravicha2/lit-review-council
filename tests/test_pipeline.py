import pytest
import json
import os
import sys
import importlib
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import before_judge, after_judge, get_synthesis_prompt

class MockSession:
    def __init__(self, state):
        self.state = state

class MockContext:
    def __init__(self, state):
        self.session = MockSession(state)

@pytest.mark.asyncio
async def test_before_judge_anonymizes_reports():
    # Setup state with fake reports
    state = {
        "report_1": {"title": "Researcher Output", "body": "Body 1"},
        "report_2": {"title": "Engineer Output", "body": "Body 2"}
    }
    ctx = MockContext(state)
    
    # Act
    await before_judge(callback_context=ctx)
    
    # Assert
    assert "anon_map_judge" in ctx.session.state
    anon_map = ctx.session.state["anon_map_judge"]
    
    assert set(anon_map.keys()) == {"A", "B"}
    assert set(anon_map.values()) == {"report_1", "report_2"}
    
    assert "anon_report_a" in ctx.session.state
    assert "anon_report_b" in ctx.session.state
    
    combined_text = ctx.session.state["anon_report_a"] + ctx.session.state["anon_report_b"]
    assert "Researcher Output" in combined_text
    assert "Engineer Output" in combined_text

from src.schema import JudgeRanking, PeerReviewResult
from src.pipeline import get_review_instruction

@pytest.mark.asyncio
async def test_after_judge_deanonymizes_winner():
    # Setup
    state = {
        "report_1": {"title": "Researcher Output"},
        "report_2": {"title": "Engineer Output"},
        "anon_map_judge": {"A": "report_1", "B": "report_2"},
        "rankings_judge": JudgeRanking(ranking=["B", "A"], rationale="B was better")
    }
    ctx = MockContext(state)
    
    # Act
    await after_judge(callback_context=ctx)
    
    # Assert
    assert ctx.session.state["winning_report_id"] == "report_2"
    assert ctx.session.state["judge_rationale"] == "B was better"

def test_get_synthesis_prompt_formats_correctly():
    # Setup
    class FakeReport:
        def model_dump_json(self):
            return '{"fake": "report"}'
            
    state = {
        "topic": "Graph DBs",
        "topic_slug": "graph-dbs",
        "winning_report_id": "report_2",
        "judge_rationale": "It was practical",
        "report_1": FakeReport(),
        "report_2": FakeReport()
    }
    ctx = MockContext(state)
    
    # Act
    prompt = get_synthesis_prompt(ctx)
    
    # Assert
    assert "Graph DBs" in prompt
    assert "graph-dbs" in prompt
    assert "engineer" in prompt
    assert "It was practical" in prompt
    assert '{"fake": "report"}' in prompt
    assert "ERROR TO FIX IN THIS RETRY" not in prompt

def test_get_synthesis_prompt_with_validation_error():
    state = {
        "topic": "Graph DBs",
        "topic_slug": "graph-dbs",
        "winning_report_id": "report_2",
        "judge_rationale": "It was practical",
        "report_1": "rep1",
        "report_2": "rep2",
        "validation_error": "Hallucinated URL: https://bad.com"
    }
    ctx = MockContext(state)
    prompt = get_synthesis_prompt(ctx)
    assert "ERROR TO FIX IN THIS RETRY:\nHallucinated URL: https://bad.com" in prompt

def test_agent_models_from_env():
    env_vars = {
        "RESEARCH_MODEL": "openrouter/research-model",
        "ENG_MODEL": "openrouter/eng-model",
        "JUDGE_MODEL": "openrouter/judge-model",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        import src.pipeline
        importlib.reload(src.pipeline)
        
        assert src.pipeline.researcher.model.model == "openrouter/research-model"
        assert src.pipeline.engineer.model.model == "openrouter/eng-model"
        assert src.pipeline.judge.model.model == "openrouter/judge-model"


# --- Peer Review Tests ---
from pydantic import ValidationError

def test_review_excludes_own_report():
    state = {
        "report_1": {"title": "Researcher Output"},
        "report_2": {"title": "Engineer Output"}
    }
    ctx = MockContext(state)
    instruction_researcher = get_review_instruction(ctx, "report_2")
    assert "Engineer Output" in instruction_researcher
    assert "Researcher Output" not in instruction_researcher

def test_review_output_matches_schema():
    with pytest.raises(ValidationError):
        PeerReviewResult(insight=5) # missing fields
    
    with pytest.raises(ValidationError):
        PeerReviewResult(accuracy=11, insight=5, defer=False, rationale="bad") # out of range

from src.pipeline import run_pipeline, InMemorySessionService
from unittest.mock import AsyncMock

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
async def test_both_approve_no_judge_call(mock_pipeline_deps):
    original_get_session = InMemorySessionService.get_session
    async def mock_get_session(self, *, app_name, user_id, session_id):
        sess = await original_get_session(self, app_name=app_name, user_id=user_id, session_id=session_id)
        if not sess.state.get("review_researcher"):
            sess.state["review_researcher"] = {"defer": True}
            sess.state["review_engineer"] = {"defer": False}
        if not sess.state.get("synthesis_result"):
            sess.state["synthesis_result"] = {"markdown": "done", "urls_cited": []}
        return sess
        
    with patch.object(InMemorySessionService, 'get_session', new=mock_get_session):
        await run_pipeline("test")
        
    agents_called = [call.kwargs.get("agent").name for call in mock_pipeline_deps.mock_runner.call_args_list if "agent" in call.kwargs]
    assert "judge" not in agents_called

@pytest.mark.asyncio
async def test_disagreement_triggers_judge(mock_pipeline_deps):
    original_get_session = InMemorySessionService.get_session
    async def mock_get_session(self, *, app_name, user_id, session_id):
        sess = await original_get_session(self, app_name=app_name, user_id=user_id, session_id=session_id)
        if not sess.state.get("review_researcher"):
            sess.state["review_researcher"] = {"defer": False}
            sess.state["review_engineer"] = {"defer": False}
        if not sess.state.get("synthesis_result"):
            sess.state["synthesis_result"] = {"markdown": "done", "urls_cited": []}
        return sess
        
    with patch.object(InMemorySessionService, 'get_session', new=mock_get_session):
        await run_pipeline("test")
        
    agents_called = [call.kwargs.get("agent").name for call in mock_pipeline_deps.mock_runner.call_args_list if "agent" in call.kwargs]
    assert "judge" in agents_called

@pytest.mark.asyncio
async def test_winning_report_id_set_after_resolution(mock_pipeline_deps):
    # Test agreement path
    original_get_session = InMemorySessionService.get_session
    async def mock_get_session(self, *, app_name, user_id, session_id):
        sess = await original_get_session(self, app_name=app_name, user_id=user_id, session_id=session_id)
        if not sess.state.get("review_researcher"):
            sess.state["review_researcher"] = {"defer": True}
            sess.state["review_engineer"] = {"defer": False}
        if not sess.state.get("synthesis_result"):
            sess.state["synthesis_result"] = {"markdown": "done", "urls_cited": []}
        return sess

    with patch.object(InMemorySessionService, 'get_session', new=mock_get_session):
        await run_pipeline("test")
        
    synthesis_call = mock_pipeline_deps.run_async_calls[-1]
    assert synthesis_call.get("state_delta", {}).get("winning_report_id") == "report_2"

def test_anon_label_never_leaks_to_state():
    # Tested by test_after_judge_deanonymizes_winner which asserts winning_report_id="report_2"
    # Here we can just verify the logic explicitly.
    state = {
        "anon_map_judge": {"A": "report_1", "B": "report_2"},
        "rankings_judge": JudgeRanking(ranking=["B", "A"], rationale="B was better")
    }
    import asyncio
    ctx = MockContext(state)
    asyncio.run(after_judge(callback_context=ctx))
    assert ctx.session.state.get("winning_report_id") == "report_2"
    assert "A" not in ctx.session.state.values()
    assert "B" not in ctx.session.state.values()

