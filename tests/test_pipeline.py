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
    assert set(anon_map.values()) == {"researcher", "engineer"}
    
    anon_text = ctx.session.state["anonymized_reports_text"]
    assert "--- Report A ---" in anon_text
    assert "--- Report B ---" in anon_text
    assert "Researcher Output" in anon_text
    assert "Engineer Output" in anon_text

from src.schema import JudgeRanking

@pytest.mark.asyncio
async def test_after_judge_deanonymizes_winner():
    # Setup
    state = {
        "report_1": {"title": "Researcher Output"},
        "report_2": {"title": "Engineer Output"},
        "anon_map_judge": {"A": "researcher", "B": "engineer"},
        "rankings_judge": JudgeRanking(ranking=["B", "A"], rationale="B was better")
    }
    ctx = MockContext(state)
    
    # Act
    await after_judge(callback_context=ctx)
    
    # Assert
    assert ctx.session.state["winning_role"] == "engineer"
    assert ctx.session.state["winning_report"] == {"title": "Engineer Output"}
    assert ctx.session.state["judge_rationale"] == "B was better"

def test_get_synthesis_prompt_formats_correctly():
    # Setup
    class FakeReport:
        def model_dump_json(self):
            return '{"fake": "report"}'
            
    state = {
        "topic": "Graph DBs",
        "topic_slug": "graph-dbs",
        "winning_role": "engineer",
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
        "winning_role": "engineer",
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

