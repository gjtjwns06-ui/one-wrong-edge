import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from owe.metamath import Statement, parse_text
from owe.retrieval import Neighborhood, traverse
from owe.finite_graphs import trail_witness, exact_trail_exists


FIXTURE = r"""
$c |- wff -> ( ) $.
$v ph ps ch $.
wph $f wff ph $. wps $f wff ps $. wch $f wff ch $.
wi $a wff ( ph -> ps ) $.
ax1 $a |- ( ph -> ( ps -> ph ) ) $.
unused $a |- ch $.
${ min $e |- ph $. maj $e |- ( ph -> ps ) $. mp $a |- ps $. $}
${ h $e |- ph $.
a1i $p |- ( ps -> ph ) $= ( wi ax1 mp unused ) ABADCABEF $.
$}
"""


def test_compressed_proof_ignores_unexecuted_reference():
    statements = parse_text(FIXTURE, verify=True)
    theorem = next(s for s in statements if s.label == "a1i")
    assert {statements[j].label for j in theorem.deps} == {"wi", "ax1", "mp"}
    assert [statements[j].label for j in theorem.hyps] == ["wph", "wps", "h"]


def test_checker_rejects_wrong_conclusion():
    broken = FIXTURE.replace("a1i $p |- ( ps -> ph )", "a1i $p |- ( ph -> ps )")
    with pytest.raises(ValueError, match="Final proof expression"):
        parse_text(broken, verify=True)


def test_checker_rejects_missing_and_out_of_scope_evidence():
    with pytest.raises(ValueError, match="Incomplete proof"):
        parse_text(FIXTURE.replace("ABADCABEF", "?"), verify=True)
    with pytest.raises(ValueError, match="Inactive hypothesis"):
        parse_text(FIXTURE.replace("( wi ax1 mp unused )", "( wi ax1 mp min )"), verify=True)


def test_checker_rejects_distinct_variable_violation():
    good = r"""$c wff |- R $. $v x y $. fx $f wff x $. fy $f wff y $.
    ${ $d x y $. ax $a |- R x y $. $}
    bad $p |- R x x $= fx fx ax $."""
    with pytest.raises(ValueError, match="Distinct-variable violation"):
        parse_text(good, verify=True)


def test_exact_interventions_equal_actual_graph_retrieval():
    # Independent BFS checks all deletions on many DAGs, including alternate paths.
    rng = np.random.default_rng(317)
    for _ in range(150):
        n = int(rng.integers(8, 45))
        adjacency = [sorted(v for v in range(u) if rng.random() < 0.14) for u in range(n)]
        seeds = rng.choice(n, size=4, replace=False).tolist()
        scores = rng.normal(size=n)
        neighborhood = Neighborhood(adjacency, seeds, scores)
        assert neighborhood.reached == traverse(adjacency, seeds)
        for edge in neighborhood.edges:
            expected = traverse(adjacency, seeds, deleted=frozenset([edge]))
            assert neighborhood.reached - neighborhood.lost_after_one(edge) == expected
            assert set(neighborhood.remove_one(edge, n)) == expected


def test_frozen_dataset_has_no_test_proof_or_self_leakage():
    path = Path("data/processed/scoped/dataset.pkl")
    if not path.exists():
        pytest.skip("Prepare the data first")
    data = pickle.loads(path.read_bytes())
    hidden = {q["signature"] for q in data["queries"] if q["split"] != "train"}
    assert all(data["nodes"][u]["signature"] not in hidden for u, _ in data["edges"])
    assert all(u > v for u, v in data["edges"])
    for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]:
        left = {q["signature"] for q in data["queries"] if q["split"] == a}
        right = {q["signature"] for q in data["queries"] if q["split"] == b}
        assert left.isdisjoint(right)
        assert {q["group"] for q in data["queries"] if q["split"] == a}.isdisjoint(
            {q["group"] for q in data["queries"] if q["split"] == b})


def test_euler_examples_preserve_connectivity_and_isolates():
    star = [(0, 1), (0, 2), (0, 3)]
    triangles = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    assert trail_witness(4, star) is None
    assert trail_witness(6, triangles) is None
    assert trail_witness(3, [(0, 1)]) == [0, 1]
    assert trail_witness(4, []) == [0]
    assert exact_trail_exists(3, [(0, 1)])
