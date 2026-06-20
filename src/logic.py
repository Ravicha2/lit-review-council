import random
import string
from typing import Dict, Tuple
from .schema import Report, Review

def anonymize_reports(reports: Dict[str, Report]) -> Tuple[Dict[str, str], Dict[str, Report]]:
    """
    Randomly assigns labels (A, B, C...) to reports.
    Returns:
        anon_map: mapping from label to original report_id (e.g. {"A": "report_1"})
        anon_reports: dictionary of reports keyed by the label
    """
    labels = list(string.ascii_uppercase[:len(reports)])
    random.shuffle(labels)
    
    anon_map = {}
    anon_reports = {}
    
    for label, (report_id, report) in zip(labels, reports.items()):
        anon_map[label] = report_id
        anon_reports[label] = report
        
    return anon_map, anon_reports

def deanonymize_reviews(anon_reviews: Dict[str, Review], anon_map: Dict[str, str]) -> Dict[str, Review]:
    """
    Maps anonymized reviews back to their original report IDs.
    """
    return {anon_map[label]: review for label, review in anon_reviews.items()}

from urllib.parse import urlparse, parse_qsl, urlencode
from typing import List

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    
    # Sort query params
    query = parsed.query
    if query:
        params = parse_qsl(query, keep_blank_values=True)
        params.sort()
        query = urlencode(params)
    
    # Strip www.
    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
        
    # Strip trailing slash
    path = parsed.path
    if path.endswith("/"):
        path = path[:-1]
        
    normalized = netloc + path
    if query:
        normalized += "?" + query
        
    return normalized

def validate_citations(used_urls: List[str], reports: List[Report]) -> bool:
    available_normalized = set()
    for report in reports:
        for ref in report.references:
            available_normalized.add(normalize_url(ref.url))
            
    for url in used_urls:
        if normalize_url(url) not in available_normalized:
            raise ValueError(f"Citation validation failed: {url} was not present in any report.")
            
    return True
