import numpy as np
from subvox_win.dtw import dtw_distance
from subvox_win.classifier import TemplateClassifier
from subvox_win.evaluation import evaluate
from subvox_win.features import frame_vector, segment_features


def _word(kind, rng, T=30, D=6):
    t = np.linspace(0, 1, T)[:, None]
    base = np.sin(2 * np.pi * (kind + 1) * t * np.arange(1, D + 1) / D)
    return base + rng.normal(0, 0.1, base.shape)


def _samples(rng, n=8):
    return [(f"w{k}", _word(k, rng)) for k in range(3) for _ in range(n)]


def test_dtw_identity_and_warp():
    a = np.sin(np.linspace(0, 6, 40))[:, None]
    assert dtw_distance(a, a) == 0
    b = np.sin(np.linspace(0, 6, 55))[:, None]
    c = np.cos(np.linspace(0, 6, 55))[:, None]
    assert dtw_distance(a, b) < dtw_distance(a, c)


def test_classifier_and_reject():
    rng = np.random.default_rng(1)
    clf = TemplateClassifier().fit(_samples(rng))
    assert clf.predict(_word(2, rng)).word == "w2"
    assert TemplateClassifier(max_distance=1e-9).fit(_samples(rng)).predict(_word(1, rng)).word is None


def test_evaluate_significant_vs_noise():
    rng = np.random.default_rng(2)
    r = evaluate(_samples(rng), permutations=100)
    assert r["accuracy"] > 0.9 and r["p_value"] < 0.05
    noise = [(f"w{i % 3}", rng.normal(size=(30, 6))) for i in range(24)]
    assert evaluate(noise, permutations=100)["p_value"] > 0.05


def test_frame_vector_pose_invariant():
    rng = np.random.default_rng(3)
    lm = rng.random((478, 3))
    v = frame_vector(lm)
    ang = 0.4
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    lm2 = lm.copy(); lm2[:, :2] = lm[:, :2] @ R.T * 2.5 + 0.3
    assert np.allclose(v, frame_vector(lm2), atol=1e-6)


def test_segment_trims_and_centres():
    f = np.zeros((40, 4)); f[15:25] = np.random.default_rng(0).normal(size=(10, 4))
    s = segment_features(f)
    assert len(s) < 40 and abs(s.mean()) < 1
