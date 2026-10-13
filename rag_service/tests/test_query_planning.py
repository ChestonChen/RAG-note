import json

from rag_service.app.query_planning import QueryPlanner
from rag_service.app.term_dictionary import TermDictionary


def test_custom_dictionary_alias_changes_plan(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text(
        json.dumps(
            {
                "version": "test-1",
                "entities": {
                    "vector_store": {
                        "aliases": ["vstore"],
                        "expansions": ["nearest neighbor"],
                    }
                },
                "intent_expansions": {"definition": ["custom definition"]},
            }
        ),
        encoding="utf-8",
    )

    plan = QueryPlanner(TermDictionary.load(path)).plan("vstore 是什么")

    assert plan.entities[0].canonical == "vector_store"
    assert "nearest neighbor" in plan.expanded_terms
    assert "custom definition" in plan.expanded_terms


def test_malformed_dictionary_falls_back_to_builtin(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    dictionary = TermDictionary.load(path)
    plan = QueryPlanner(dictionary).plan("pgvector 是什么")

    assert dictionary.source == "builtin"
    assert plan.entities and plan.entities[0].canonical == "pgvector"


def test_debug_metadata_reports_dictionary_without_content_or_secrets(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps({"version": "safe-7", "entities": {"secret_topic": {"aliases": ["internal"]}}}),
        encoding="utf-8",
    )

    debug = QueryPlanner(TermDictionary.load(path)).plan("internal").to_debug_dict()

    assert debug["dictionary"]["version"] == "safe-7"
    assert debug["dictionary"]["source"] == "external"
    assert "secret_topic" in debug["selected_expansions"]
    assert "internal" not in str(debug["dictionary"])

