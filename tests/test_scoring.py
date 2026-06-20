import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.schema import Report, Reference, JudgeRanking
from src.scoring import generate_anonymization_map, tally_ensemble_rankings, extract_hallucinated_urls

def test_generate_anonymization_map():
    report_ids = ["report_1", "report_2"]
    anon_map = generate_anonymization_map(report_ids)
    
    assert set(anon_map.keys()) == {"A", "B"}
    assert set(anon_map.values()) == {"report_1", "report_2"}

def test_tally_ensemble_rankings_majority():
    rankings = [
        JudgeRanking(ranking=["A", "B"], rationale="R1"),
        JudgeRanking(ranking=["A", "B"], rationale="R2"),
        JudgeRanking(ranking=["B", "A"], rationale="R3")
    ]
    
    winner, reasons = tally_ensemble_rankings(rankings)
    assert winner == "A"
    assert len(reasons) == 3
    assert "R1" in reasons[0]

def test_tally_ensemble_rankings_tie_breaker():
    rankings = [
        JudgeRanking(ranking=["A", "B"], rationale="R1"),
        JudgeRanking(ranking=["B", "A"], rationale="R2")
    ]
    winner, _ = tally_ensemble_rankings(rankings)
    assert winner == "A" # A wins ties based on original pipeline logic

def test_tally_ensemble_rankings_missing_values():
    rankings = [
        None,
        JudgeRanking(ranking=[], rationale="R1"),
        JudgeRanking(ranking=["B"], rationale="R2")
    ]
    winner, _ = tally_ensemble_rankings(rankings)
    assert winner == "B"

def test_extract_hallucinated_urls():
    report_1 = Report(
        title="R1",
        body="...",
        references=[Reference(title="T1", url="http://example.com/a")]
    )
    report_2 = Report(
        title="R2",
        body="...",
        references=[Reference(title="T2", url="http://example.com/b")]
    )
    
    cited_urls = [
        "http://example.com/a",
        "HTTP://EXAMPLE.COM/B",
        "http://example.com/c",
        "http://example.com/d  "
    ]
    
    hallucinations = extract_hallucinated_urls(cited_urls, [report_1, report_2])
    assert set(hallucinations) == {"http://example.com/c", "http://example.com/d  "}
