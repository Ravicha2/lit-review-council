import os
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def is_retryable_http_error(exception):
    if isinstance(exception, requests.exceptions.HTTPError):
        status = exception.response.status_code
        # Retry on 429 Too Many Requests and 5xx Server Errors
        return status == 429 or status >= 500
    # Also retry on timeouts and connection errors
    return isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))

from .schema import SearchResult

class BaseProvider(ABC):
    @abstractmethod
    def search(self, query: str) -> list[SearchResult]:
        pass

class ArxivProvider(BaseProvider):
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception(is_retryable_http_error),
        reraise=True
    )
    def search(self, query: str) -> list[SearchResult]:
        # Pass query directly to allow ArXiv to handle multi-word bag-of-words natively.
        # Previously `all:{query}` forced exact phrase matching for multi-word queries.
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results=5"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        root = ET.fromstring(r.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            link = entry.find('atom:id', ns)
            
            title_text = title.text.strip().replace('\n', ' ') if title is not None else ""
            summary_text = summary.text.strip() if summary is not None else ""
            link_text = link.text.strip() if link is not None else ""
            
            results.append(SearchResult(
                title=title_text,
                url=link_text,
                snippet=summary_text[:500],
                content=summary_text
            ))
        return results

class GithubProvider(BaseProvider):
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception(is_retryable_http_error),
        reraise=True
    )
    def search(self, query: str) -> list[SearchResult]:
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc"
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
            
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        results = []
        for item in data.get("items", [])[:5]:
            repo_name = item.get("full_name")
            html_url = item.get("html_url")
            description = item.get("description") or ""
            
            # Fetch the raw README content
            readme_url = f"https://raw.githubusercontent.com/{repo_name}/HEAD/README.md"
            readme_resp = requests.get(readme_url, timeout=5)
            content = readme_resp.text[:4000] if readme_resp.status_code == 200 else description
            
            results.append(SearchResult(
                title=repo_name,
                url=html_url,
                snippet=description,
                content=content
            ))
        return results

class TavilyProvider(BaseProvider):
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception(is_retryable_http_error),
        reraise=True
    )
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("[!] TAVILY_API_KEY not set. Cannot search Tavily.")
            return []
            
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_raw_content": True
        }
        
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        results = []
        for item in data.get("results", []):
            title = item.get("title") or ""
            url_text = item.get("url") or ""
            content = item.get("content") or ""
            raw_content = item.get("raw_content") or content
            
            results.append(SearchResult(
                title=title,
                url=url_text,
                snippet=content,
                content=raw_content
            ))
        return results

def create_adk_tool(provider: BaseProvider, name: str, description: str) -> Callable[[str], list[dict]]:
    """
    Wraps a SearchProvider into an ADK-compatible tool function.
    Safely catches exceptions so the agent doesn't crash on network errors.
    """
    def _search_tool(query: str) -> list[dict]:
        print(f"[*] {name} searching for: '{query}'", flush=True)
        try:
            results = provider.search(query)
            print(f"[*] {name} found {len(results)} results", flush=True)
            return [r.model_dump() for r in results]
        except Exception as e:
            # Return a graceful error so the LLM knows the tool failed but can continue
            print(f"[!] {name} error: {str(e)}", flush=True)
            return [{"error": f"{name} API error: {str(e)}"}]
            
    _search_tool.__name__ = name
    _search_tool.__doc__ = description
    return _search_tool
