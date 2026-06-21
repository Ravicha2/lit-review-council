import pytest
from unittest.mock import patch, Mock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.providers import TavilyProvider, OpenAlexProvider

def test_tavily_provider_search():
    with patch("src.providers.requests.post") as mock_post:
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "title": "Tavily Result",
                    "url": "https://tavily.com",
                    "content": "Some content",
                    "raw_content": "Raw content here"
                }
            ]
        }
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"}):
            provider = TavilyProvider()
            results = provider.search("test query", max_results=2)
            
            assert len(results) == 1
            assert results[0].title == "Tavily Result"
            assert results[0].url == "https://tavily.com"
            assert results[0].snippet == "Some content"
            assert results[0].content == "Raw content here"
            
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert "api.tavily.com/search" in args[0]
            assert kwargs["json"]["api_key"] == "fake_key"
            assert kwargs["json"]["query"] == "test query"
            assert kwargs["json"]["max_results"] == 2

def test_tavily_provider_search_with_domains():
    with patch("src.providers.requests.post") as mock_post:
        mock_resp = Mock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"}):
            provider = TavilyProvider(include_domains=["github.com"], exclude_domains=["arxiv.org"])
            provider.search("test query")
            
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["include_domains"] == ["github.com"]
            assert kwargs["json"]["exclude_domains"] == ["arxiv.org"]

def test_tavily_provider_truncates_long_raw_content():
    with patch("src.providers.requests.post") as mock_post:
        mock_resp = Mock()
        long_content = "A" * 10000
        mock_resp.json.return_value = {
            "results": [
                {
                    "title": "Tavily Result",
                    "url": "https://tavily.com",
                    "content": "Short snippet",
                    "raw_content": long_content
                }
            ]
        }
        mock_resp.raise_for_status = Mock()
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"}):
            provider = TavilyProvider()
            results = provider.search("test query")
            
            assert len(results) == 1
            assert len(results[0].content) == 8000
            assert results[0].content == "A" * 8000

def test_openalex_provider_search():
    with patch("src.providers.requests.get") as mock_get:
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test OpenAlex Result",
                    "abstract_inverted_index": {
                        "This": [0],
                        "is": [1],
                        "a": [2],
                        "test.": [3]
                    }
                }
            ]
        }
        mock_resp.raise_for_status = Mock()
        mock_get.return_value = mock_resp

        with patch.dict(os.environ, {"OPENALEX_API_KEY": "fake_openalex_key"}):
            provider = OpenAlexProvider()
            results = provider.search("test query")
            
            assert len(results) == 1
            assert results[0].title == "Test OpenAlex Result"
            assert results[0].url == "https://openalex.org/W123"
            assert results[0].snippet == "This is a test."
            assert results[0].content == "This is a test."
            
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert "api.openalex.org/works" in args[0]
            assert kwargs["params"]["api_key"] == "fake_openalex_key"
            assert kwargs["params"]["search"] == "test query"
            assert kwargs["params"]["per-page"] == 5
