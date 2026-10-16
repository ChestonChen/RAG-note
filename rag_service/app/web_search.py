"""Optional web-search fallback for low-confidence questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings, get_settings
from .schemas import WebSource


@dataclass(slots=True)
class WebSearchResult:
    sources: list[WebSource]
    status: str
    reason: str | None = None
    error: str | None = None


class WebSearchService:
    """Call a configured provider without making web search mandatory."""

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or httpx

    def search(self, question: str) -> WebSearchResult:
        question = " ".join((question or "").split())
        if not question:
            return WebSearchResult([], "skipped", "empty question")
        if not self.settings.web_search_enabled:
            return WebSearchResult([], "disabled", "WEB_SEARCH_ENABLED is false")
        provider = self.settings.web_search_provider
        if provider == "none":
            return WebSearchResult([], "disabled", "WEB_SEARCH_PROVIDER is none")
        if provider == "tavily":
            return self._search_tavily(question)
        if provider == "proxy_gateway":
            return self._search_proxy_gateway(question)
        return WebSearchResult([], "unsupported", f"unsupported provider: {provider}")

    def _search_tavily(self, question: str) -> WebSearchResult:
        api_key = self.settings.tavily_api_key
        if not api_key:
            return WebSearchResult([], "not_configured", "TAVILY_API_KEY is empty")
        try:
            response = self.client.post(
                "https://api.tavily.com/search",
                json={
                    "query": question,
                    "search_depth": "basic",
                    "max_results": self.settings.web_search_max_results,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self.settings.web_search_timeout_seconds,
            )
            response.raise_for_status()
            return WebSearchResult(self._sources(response.json()), "ok")
        except httpx.TimeoutException as exc:
            return WebSearchResult([], "timeout", "web search timed out", type(exc).__name__)
        except Exception as exc:
            return WebSearchResult([], "error", "web search request failed", type(exc).__name__)

    def _search_proxy_gateway(self, question: str) -> WebSearchResult:
        url = self.settings.web_search_proxy_url
        if not url:
            return WebSearchResult([], "not_configured", "WEB_SEARCH_PROXY_URL is empty")
        try:
            headers = {}
            if self.settings.web_search_proxy_api_key:
                headers["Authorization"] = f"Bearer {self.settings.web_search_proxy_api_key}"
            response = self.client.post(
                url,
                json={"query": question, "max_results": self.settings.web_search_max_results},
                headers=headers,
                timeout=self.settings.web_search_timeout_seconds,
            )
            response.raise_for_status()
            return WebSearchResult(self._sources(response.json()), "ok")
        except httpx.TimeoutException as exc:
            return WebSearchResult([], "timeout", "web search timed out", type(exc).__name__)
        except Exception as exc:
            return WebSearchResult([], "error", "web search request failed", type(exc).__name__)

    def _sources(self, body: Any) -> list[WebSource]:
        if not isinstance(body, dict):
            return []
        raw_items = body.get("results") or body.get("sources") or []
        sources: list[WebSource] = []
        seen: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            url = str(item["url"])
            if url in seen:
                continue
            seen.add(url)
            snippet = item.get("content") or item.get("snippet") or item.get("summary")
            sources.append(
                WebSource(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(snippet)[:2000] if snippet is not None else None,
                )
            )
            if len(sources) >= self.settings.web_search_max_results:
                break
        return sources


__all__ = ["WebSearchResult", "WebSearchService"]
