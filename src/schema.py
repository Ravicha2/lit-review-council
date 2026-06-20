from pydantic import BaseModel, Field
from typing import List, Dict

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str

class Reference(BaseModel):
    title: str
    url: str

class Report(BaseModel):
    title: str
    body: str
    references: List[Reference]

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
    urls_cited: list[str]

class GatheredSources(BaseModel):
    notes: str
    urls: list[str]
