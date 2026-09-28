"""Regression tests for globally-empty intermediate results, singleton-only ground truth and ties
(reported by an external review of v3)."""
import numpy as np
import pandas as pd
import pytest

from src.decide import assign_expected_f, assign_hybrid, assign_threshold, best_per_record
from src.features import _gap_to_best_other, labels

EMPTY = pd.DataFrame({"ri": np.zeros(0, np.int32), "si": np.zeros(0, np.int32), "p": np.zeros(0, np.float32)})


def test_empty_prediction_table():
    best = best_per_record(EMPTY)
    assert len(best) == 0 and {"ri", "si", "p"} <= set(best.columns)
    assert len(assign_threshold(best, 0.8)) == 0
    assert len(assign_hybrid(EMPTY, best, 5, 0.3, 0.75)) == 0
    assert len(assign_expected_f(EMPTY, best, 5)) == 0


def test_all_probabilities_below_minimum():
    pr = pd.DataFrame({"ri": [0, 1], "si": [0, 1], "p": np.float32([0.1, 0.2])})
    best = best_per_record(pr)
    assert len(assign_hybrid(pr, best, 3, 0.3, 0.75)) == 0
    assert len(assign_expected_f(pr, best, 3, p_min=0.3)) == 0


def test_empty_context():
    g, r, c = _gap_to_best_other(np.zeros(0, np.int64), np.zeros(0, np.float32))
    assert len(g) == len(r) == len(c) == 0


def test_singleton_only_ground_truth(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-b\t\n")
    s1 = pd.DataFrame({"id": ["S1-a", "S1-b"]})
    recs = pd.DataFrame({"id": ["S2-x", "S3-y"]})
    assert (labels(recs, s1, p) == -1).all()


def test_ties_are_order_independent():
    pr = pd.DataFrame({"ri": [0, 0, 1, 1], "si": [3, 1, 2, 0], "p": np.float32([0.9, 0.9, 0.7, 0.7])})
    a = best_per_record(pr)
    b = best_per_record(pr.iloc[::-1].reset_index(drop=True))
    key = lambda d: sorted(zip(d.ri.tolist(), d.si.tolist()))
    assert key(a) == key(b) == [(0, 1), (1, 0)]              # tie -> lowest S1 row, whatever the input order
    ea = assign_expected_f(pr, a, 4, p_min=0.1, use_empty=False)
    eb = assign_expected_f(pr.iloc[::-1].reset_index(drop=True), b, 4, p_min=0.1, use_empty=False)
    assert key(ea) == key(eb)


# ---------------------------------------------------------------- stale artifacts
def test_step_refuses_stale_output(tmp_path):
    from src import fingerprint as fpr
    from src.pipeline import step
    out = tmp_path / "x.parquet"
    calls = []
    step("s", out, lambda: (calls.append(1), out.write_text("a")), "fp1")
    step("s", out, lambda: calls.append(2), "fp1")                   # same fingerprint -> reused
    assert calls == [1]
    with pytest.raises(fpr.StaleArtifact):
        step("s", out, lambda: calls.append(3), "fp2")               # changed input/code/config -> refused
    out2 = tmp_path / "legacy.parquet"
    out2.write_text("old")                                           # output without fingerprint -> refused
    with pytest.raises(fpr.StaleArtifact):
        step("s", out2, lambda: None, "fp1")


def test_model_loader_requires_contiguous_folds(tmp_path):
    import json
    from src.model import StaleModels, load_boosters
    (tmp_path / "model_config.json").write_text(json.dumps({"n_folds": 3}))
    for k in (0, 2):
        (tmp_path / f"xgb_fold{k}.ubj").write_bytes(b"")
    with pytest.raises(StaleModels):
        load_boosters(tmp_path, "cpu")


def test_token_map_needs_distinct_entities():
    from src.token_map import learn_token_map
    s1 = ["Sharma Traders"] * 3
    rec = ["शर्मा Traders"] * 3
    assert learn_token_map(s1, rec, entity_ids=["e1", "e1", "e1"]) == {}        # 3 aliases of ONE entity
    assert learn_token_map(s1, rec, entity_ids=["e1", "e2", "e3"]) == {"शर्मा": "sharma"}
