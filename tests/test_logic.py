import pytest
from src.schema import Report, Reference, Review
from src.logic import anonymize_reports, deanonymize_reviews

def test_anonymize_reports():
    reports = {
        "report_1": Report(title="T1", body="B1", references=[]),
        "report_2": Report(title="T2", body="B2", references=[])
    }
    
    anon_map, anon_reports = anonymize_reports(reports)
    
    # anon_map should map new labels (A, B) to original keys
    assert set(anon_map.values()) == {"report_1", "report_2"}
    assert set(anon_map.keys()) == {"A", "B"}
    
    # anon_reports should use the new labels
    assert set(anon_reports.keys()) == {"A", "B"}
    
    # content should match the mapped original
    assert anon_reports["A"].title == reports[anon_map["A"]].title
    assert anon_reports["B"].title == reports[anon_map["B"]].title

def test_deanonymize_reviews():
    anon_reviews = {
        "A": Review(score=1.0, rationale="Great"),
        "B": Review(score=0.0, rationale="Bad")
    }
    anon_map = {
        "A": "report_1",
        "B": "report_2"
    }
    
    real_reviews = deanonymize_reviews(anon_reviews, anon_map)
    
    assert set(real_reviews.keys()) == {"report_1", "report_2"}
    assert real_reviews["report_1"].score == 1.0
    assert real_reviews["report_2"].score == 0.0

from src.logic import normalize_url, validate_citations

def test_normalize_url():
    assert normalize_url("https://www.example.com/path/") == "example.com/path"
    assert normalize_url("http://example.com/path") == "example.com/path"
    assert normalize_url("https://example.com/path?b=2&a=1") == "example.com/path?a=1&b=2"

def test_validate_citations():
    reports = [
        Report(title="T1", body="B1", references=[Reference(title="R1", url="https://example.com/test/")])
    ]
    
    # Should pass (normalized matches)
    validate_citations(["http://www.example.com/test"], reports)
    
    # Should fail (hallucinated)
    with pytest.raises(ValueError):
        validate_citations(["https://example.com/other"], reports)
