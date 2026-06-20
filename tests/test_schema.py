from pydantic import ValidationError
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import Reference, Report

def test_reference_requires_title_and_url():
    # Valid reference
    ref = Reference(title="Test", url="https://example.com")
    assert ref.title == "Test"
    assert ref.url == "https://example.com"
    
    # Missing title
    with pytest.raises(ValidationError):
        Reference(url="https://example.com")

    # Missing url
    with pytest.raises(ValidationError):
        Reference(title="Test")

def test_report_schema_parsing():
    valid_data = {
        "title": "A Great Report",
        "body": "This is a body of the report.",
        "references": [
            {"title": "Source 1", "url": "https://source1.com"}
        ]
    }
    
    report = Report(**valid_data)
    assert report.title == "A Great Report"
    assert report.body == "This is a body of the report."
    assert len(report.references) == 1
    assert report.references[0].url == "https://source1.com"

def test_report_missing_fields():
    # Missing references should fail
    with pytest.raises(ValidationError):
        Report(title="Title", body="Body")
        
    # Missing body should fail
    with pytest.raises(ValidationError):
        Report(title="Title", references=[])


