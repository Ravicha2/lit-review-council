from pydantic import BaseModel
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

class JudgeRanking(BaseModel):
    ranking: list[str]
    rationale: str

class SynthesisResult(BaseModel):
    markdown: str
    urls_cited: list[str]
