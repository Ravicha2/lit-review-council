import random
from typing import List, Dict, Tuple, Set, Optional, Any
from src.schema import Report, JudgeRanking

def generate_anonymization_map(report_ids: List[str]) -> Dict[str, str]:
    """Generates a random mapping of generic labels 'A' and 'B' to report IDs."""
    labels = ["A", "B"]
    random.shuffle(labels)
    # Ensure we only map as many items as we have (assume 2 based on original logic)
    return {labels[0]: report_ids[0], labels[1]: report_ids[1]}

def tally_ensemble_rankings(rankings: List[Optional[Any]]) -> Tuple[str, List[str]]:
    """
    Applies Borda count (2 pts for 1st, 1 pt for 2nd) and returns the winning label 
    and the list of rationale strings.
    """
    scores = {"A": 0, "B": 0}
    reasons = []
    
    reviewers = ["Researcher", "Engineer", "Architect"]
    
    for i, r_obj in enumerate(rankings):
        reviewer_name = reviewers[i] if i < len(reviewers) else f"Reviewer_{i}"
        
        if r_obj:
            lst = r_obj.ranking if hasattr(r_obj, "ranking") else r_obj.get("ranking", [])
            rat = r_obj.rationale if hasattr(r_obj, "rationale") else r_obj.get("rationale", "")
            if lst and len(lst) >= 1:
                # 1st place gets 2 points, 2nd gets 1 point
                if lst[0] in scores: scores[lst[0]] += 2
                if len(lst) > 1 and lst[1] in scores: scores[lst[1]] += 1
            reasons.append(f"{reviewer_name} rationale: {rat}")

    top_label = "A" if scores["A"] >= scores["B"] else "B"
    return top_label, reasons

import re

def validate_synthesis_citations(markdown_text: str, synth_references: List[Optional[Any]], source_reports: List[Optional[Any]]) -> Optional[str]:
    """
    Validates the synthesis report for strict citation grounding.
    Returns an error message string if validation fails, or None if it passes.
    """
    # Step 1: Check for forbidden dangling formats (Author, Year) or [1]
    # \([A-Z][a-z\s]+, \d{4}\) catches (Smith, 2024) or (OpenAI, 2024)
    # \[\d+\] catches [1], [2], etc.
    dangling_regex = r'(\([A-Z][A-Za-z\s]+, \d{4}\)|\[\d+\](?!\())'
    if re.search(dangling_regex, markdown_text):
        return (
            "Dangling citation format detected. DO NOT use (Author, Year) or [1] style citations. "
            "You MUST use inline Markdown links: [Source Title](URL) for every citation."
        )

    import urllib.parse
    def normalize_url(u: str) -> str:
        p = urllib.parse.urlparse(u)
        path = p.path.rstrip('.,;!?/')
        netloc = p.netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return f"{netloc}{path}"

    # Step 2: Extract all Markdown URLs from the body text
    markdown_links = re.findall(r'\[.*?\]\((https?://[^\)]+)\)', markdown_text)
    markdown_urls = set(normalize_url(url.strip()) for url in markdown_links)

    # Step 3: Extract URLs from the synthesis references array
    synth_urls = set()
    synth_url_map = {} # Keep original for error messages
    for ref in synth_references:
        if ref:
            url = ref.url if hasattr(ref, "url") else ref.get("url", "")
            if url:
                norm_url = normalize_url(url.strip())
                synth_urls.add(norm_url)
                synth_url_map[norm_url] = url

    # Step 4: Verify every extracted URL exists in the synth_references array
    for url in markdown_urls:
        if url not in synth_urls:
            return f"URL {url} (normalized) was cited in the text but not found in your references list."

    # Step 5: Verify every URL in synth_references exists in the original source_reports
    valid_source_urls: Set[str] = set()
    for rep in source_reports:
        if rep:
            refs = rep.references if hasattr(rep, "references") else rep.get("references", [])
            for ref in refs:
                url = ref.url if hasattr(ref, "url") else ref.get("url", "")
                if url:
                    valid_source_urls.add(normalize_url(url.strip()))

    for norm_url in synth_urls:
        if norm_url not in valid_source_urls:
            orig_url = synth_url_map[norm_url]
            return f"URL {orig_url} is hallucinated or not present in the original reports' references. Only cite provided sources."

    return None

def check_blog_tier_ratio(references: List[Optional[Any]]) -> Optional[str]:
    """
    Checks if more than 50% of the given references are classified as 'blog_or_forum'.
    Returns a flag string if true, or None otherwise.
    """
    total = 0
    blog_count = 0
    
    for ref in references:
        if ref:
            total += 1
            tier = ref.source_tier if hasattr(ref, "source_tier") else ref.get("source_tier", "")
            if tier == "blog_or_forum":
                blog_count += 1
                
    if total > 0 and (blog_count / total) > 0.5:
        return f"⚠️ {blog_count} of {total} sources in this report are blog/forum tier."
    return None
