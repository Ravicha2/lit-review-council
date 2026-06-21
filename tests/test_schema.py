from pydantic import ValidationError
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import Reference, Report

def test_reference_requires_title_and_url_and_tier():
    # Valid reference
    ref = Reference(title="Test", url="https://example.com", source_tier="peer_reviewed")
    assert ref.title == "Test"
    assert ref.url == "https://example.com"
    assert ref.source_tier == "peer_reviewed"
    
    # Missing title
    with pytest.raises(ValidationError):
        Reference(url="https://example.com", source_tier="peer_reviewed")

    # Missing url
    with pytest.raises(ValidationError):
        Reference(title="Test", source_tier="peer_reviewed")

    # Missing source_tier
    with pytest.raises(ValidationError):
        Reference(title="Test", url="https://example.com")

    # Invalid source_tier
    with pytest.raises(ValidationError):
        Reference(title="Test", url="https://example.com", source_tier="invalid_tier")

def test_report_schema_parsing():
    valid_data = {
        "title": "A Great Report",
        "body": "This is a body of the report. [link](https://source1.com)",
        "references": [
            {"title": "Source 1", "url": "https://source1.com", "source_tier": "blog_or_forum", "usage": "cited"}
        ]
    }
    
    report = Report(**valid_data)
    assert report.title == "A Great Report"
    assert report.body == "This is a body of the report. [link](https://source1.com)"
    assert len(report.references) == 1
    assert report.references[0].url == "https://source1.com"
    assert report.references[0].source_tier == "blog_or_forum"

def test_report_missing_fields():
    # Missing references should fail
    with pytest.raises(ValidationError):
        Report(title="Title", body="Body")
        
    # Missing body should fail
    with pytest.raises(ValidationError):
        Report(title="Title", references=[])


