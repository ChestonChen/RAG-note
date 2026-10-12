from rag_service.app.config import Settings
from rag_service.app.web_search import WebSearchService


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self):
        return self._body


class _FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_tavily_search_returns_normalized_sources():
    http = _FakeHttpClient(
        _FakeResponse(
            {
                "results": [
                    {
                        "title": "pgvector docs",
                        "url": "https://example.test/pgvector",
                        "content": "Vector extension documentation",
                    }
                ]
            }
        )
    )
    settings = Settings(
        web_search_enabled=True,
        web_search_provider="tavily",
        tavily_api_key="tavily-key",
    )

    result = WebSearchService(settings, client=http).search("pgvector latest release")

    assert result.status == "ok"
    assert result.sources[0].url == "https://example.test/pgvector"
    assert http.calls[0][0] == "https://api.tavily.com/search"
    assert http.calls[0][1]["headers"]["Authorization"] == "Bearer tavily-key"


def test_web_search_reports_missing_credentials_without_network_call():
    http = _FakeHttpClient(_FakeResponse({"results": []}))
    settings = Settings(web_search_enabled=True, web_search_provider="tavily")

    result = WebSearchService(settings, client=http).search("anything")

    assert result.status == "not_configured"
    assert result.sources == []
    assert http.calls == []
