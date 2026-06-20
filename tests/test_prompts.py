import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.prompts import build_synthesis_prompt, build_ensemble_instruction

class FakeReport:
    def model_dump_json(self):
        return '{"fake": "report"}'

def test_build_synthesis_prompt_formats_correctly():
    prompt = build_synthesis_prompt(
        topic="Graph DBs",
        topic_slug="graph-dbs",
        winning_report_id="report_2",
        rationale="It was practical",
        report_1=FakeReport(),
        report_2=FakeReport()
    )
    
    assert "Graph DBs" in prompt
    assert "graph-dbs" in prompt
    assert "engineer" in prompt  # report_2 winner means engineer
    assert "It was practical" in prompt
    assert '{"fake": "report"}' in prompt
    assert "ERROR TO FIX" not in prompt

def test_build_synthesis_prompt_includes_validation_error():
    prompt = build_synthesis_prompt(
        topic="Graph DBs",
        topic_slug="graph-dbs",
        winning_report_id="report_1",
        rationale="It was theoretical",
        report_1=FakeReport(),
        report_2=FakeReport(),
        validation_error="Hallucinated URLs found"
    )
    
    assert "researcher" in prompt  # report_1 winner means researcher
    assert "ERROR TO FIX IN THIS RETRY:\nHallucinated URLs found" in prompt

def test_build_ensemble_instruction():
    prompt = build_ensemble_instruction(
        role="Researcher",
        report_a_text="Report A Content",
        report_b_text="Report B Content"
    )
    
    assert "You are a Research Scientist." in prompt
    assert "Report A Content" in prompt
    assert "Report B Content" in prompt
