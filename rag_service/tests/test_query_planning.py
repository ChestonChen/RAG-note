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
    assert debug["entity_count"] == 1
    assert "secret_topic" not in str(debug)
    assert "internal" not in str(debug)


def test_schema_violations_fall_back_to_builtin(tmp_path):
    cases = [
        {"entities": {"x": {"aliases": "not-list"}}},
        {"intent_expansions": {"definition": "not-list"}},
        {"required_terms": "not-list"},
        {"version": True},
        {"version": "C:/private/secret"},
    ]
    for index, payload in enumerate(cases):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert TermDictionary.load(path).source == "builtin"


def test_required_terms_are_preserved_in_rewritten_query(tmp_path):
    path = tmp_path / "required.json"
    path.write_text(json.dumps({"required_terms": ["must-have-token"]}), encoding="utf-8")
    plan = QueryPlanner(TermDictionary.load(path)).plan("pgvector 是什么")
    assert "must-have-token" in plan.rewritten_query


def test_broad_and_technical_terms_are_selected_only_when_relevant(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"broad_terms": ["broad-secret"], "technical_hints": ["hint-secret"]}), encoding="utf-8")
    dictionary = TermDictionary.load(path)
    unrelated = QueryPlanner(dictionary).plan("天气怎么样")
    relevant = QueryPlanner(dictionary).plan("broad-secret 是什么")
    assert "broad-secret" not in unrelated.rewritten_query
    assert "hint-secret" not in unrelated.rewritten_query
    assert "broad-secret" in relevant.rewritten_query

