import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.schema import Report, Reference, JudgeRanking, ExploredReference
from src.scoring import generate_anonymization_map, tally_ensemble_rankings, validate_synthesis_citations, check_blog_tier_ratio

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

def test_check_blog_tier_ratio():
    # 0% blog_or_forum
    refs_good = [
        Reference(title="T1", url="http://example.com/a", source_tier="peer_reviewed"),
        Reference(title="T2", url="http://example.com/b", source_tier="established_project")
    ]
    assert check_blog_tier_ratio(refs_good) is None
    
    # 50% blog_or_forum (should not flag)
    refs_borderline = [
        Reference(title="T1", url="http://example.com/a", source_tier="blog_or_forum"),
        Reference(title="T2", url="http://example.com/b", source_tier="established_project")
    ]
    assert check_blog_tier_ratio(refs_borderline) is None
    
    # >50% blog_or_forum (should flag)
    refs_bad = [
        Reference(title="T1", url="http://example.com/a", source_tier="blog_or_forum"),
        Reference(title="T2", url="http://example.com/b", source_tier="blog_or_forum"),
        Reference(title="T3", url="http://example.com/c", source_tier="established_project")
    ]
    flag = check_blog_tier_ratio(refs_bad)
    assert flag is not None
    assert "2 of 3 sources in this report are blog/forum tier" in flag

def test_validate_synthesis_citations():
    report_1 = Report(
        title="R1",
        body="... [T1](http://example.com/a)",
        references=[ExploredReference(title="T1", url="http://example.com/a", source_tier="peer_reviewed", usage="cited")]
    )
    report_2 = Report(
        title="R2",
        body="... [T2](http://example.com/b)",
        references=[ExploredReference(title="T2", url="http://example.com/b", source_tier="established_project", usage="cited")]
    )
    source_reports = [report_1, report_2]

    # Test 1: Dangling citations
    bad_markdown = "Here is a fact (Rasband, 2024)."
    synth_refs = [Reference(title="T1", url="http://example.com/a", source_tier="peer_reviewed")]
    err = validate_synthesis_citations(bad_markdown, synth_refs, source_reports)
    assert err is not None
    assert "Dangling citation format detected" in err

    bad_markdown2 = "Here is another fact [1]."
    err2 = validate_synthesis_citations(bad_markdown2, synth_refs, source_reports)
    assert err2 is not None
    assert "Dangling citation format detected" in err2

    # Test 2: Markdown URL not in synth_references
    bad_markdown3 = "Here is a fact [T3](http://example.com/c)."
    err3 = validate_synthesis_citations(bad_markdown3, synth_refs, source_reports)
    assert err3 is not None
    assert "not found in your references list" in err3

    # Test 3: synth_references has URL not in source_reports
    good_markdown = "Here is a fact [T1](http://example.com/a)."
    bad_synth_refs = [
        Reference(title="T1", url="http://example.com/a", source_tier="peer_reviewed"),
        Reference(title="T3", url="http://example.com/c", source_tier="blog_or_forum")
    ]
    err4 = validate_synthesis_citations(good_markdown, bad_synth_refs, source_reports)
    assert err4 is not None
    assert "not present in the original reports" in err4

    # Test 4: All good
    good_synth_refs = [Reference(title="T1", url="http://example.com/a", source_tier="peer_reviewed")]
    err5 = validate_synthesis_citations(good_markdown, good_synth_refs, source_reports)
    assert err5 is None
