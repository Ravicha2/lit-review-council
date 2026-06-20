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

def extract_hallucinated_urls(cited_urls: List[str], source_reports: List[Optional[Any]]) -> List[str]:
    """
    Returns URLs that were cited but not present in the original reports.
    """
    valid_urls: Set[str] = set()
    for rep in source_reports:
        if rep:
            refs = rep.references if hasattr(rep, "references") else rep.get("references", [])
            for ref in refs:
                url = ref.url if hasattr(ref, "url") else ref.get("url", "")
                valid_urls.add(url.strip().lower())
    
    hallucinated = []
    for url in cited_urls:
        if url.strip().lower() not in valid_urls:
            hallucinated.append(url)
            
    return hallucinated
