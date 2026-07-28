from pc_cng.verify_phase4_union_v2 import parse_source_mixtures


def test_parse_source_mixtures_tracks_scenarios():
    text = """
[union] === Scenario: author_lab ===
  [union_v2] train source mixture: {'learned_structured': 7, 'rule_pc_cng': 3}
[union] === Scenario: random ===
  [union_v2] train source mixture: {'learned_structured': 2, 'shuffled_parent': 8}
"""
    assert parse_source_mixtures(text) == {
        "author_lab": {"learned_structured": 7, "rule_pc_cng": 3},
        "random": {"learned_structured": 2, "shuffled_parent": 8},
    }
