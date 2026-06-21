from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Literal
import re
import urllib.parse

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str

class Reference(BaseModel):
    title: str
    url: str
    source_tier: Literal["peer_reviewed", "established_project", "vendor_doc", "blog_or_forum"]

class ExploredReference(Reference):
    usage: Literal["cited", "rejected", "unevaluated"]

class Report(BaseModel):
    title: str
    body: str
    references: List[ExploredReference]

def validate_report(report: Report | dict) -> str | None:
    if isinstance(report, dict):
        report = Report(**report)
    raw_body_urls = re.findall(r'https?://[^\s)\]"\'<>]+', report.body)
    
    def normalize_url(url: str) -> str:
        u = urllib.parse.urlparse(url)
        path = u.path.rstrip('.,;!?/')
        netloc = u.netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return f"{netloc}{path}"
    
    body_urls = {normalize_url(u) for u in raw_body_urls}
    
    for ref in report.references:
        norm_ref = normalize_url(ref.url)
        if ref.usage == "cited" and norm_ref not in body_urls:
            return f"Reference '{ref.url}' is tagged as 'cited' but was not found in the body text. You MUST include the exact URL in the body text as an inline markdown link."
        elif ref.usage in ["rejected", "unevaluated"] and norm_ref in body_urls:
            return f"Reference '{ref.url}' is tagged as '{ref.usage}' but was actually cited in the body text. Change the usage tag to 'cited'."
            
    return None

class Review(BaseModel):
    score: float
    rationale: str

class PeerReviewResult(BaseModel):
    accuracy: int = Field(ge=0, le=10)
    insight: int = Field(ge=0, le=10)
    defer: bool
    rationale: str

class JudgeRanking(BaseModel):
    ranking: list[str]
    rationale: str

class SynthesisResult(BaseModel):
    markdown: str
    references: List[Reference] = Field(default_factory=list)

class GatheredSources(BaseModel):
    notes: str
    urls: list[str]
