"""KNN std floors on the features' physical scale: an empty training hue bin is no veto."""
from __future__ import annotations

import numpy as np
import pytest

from cube_task.perception.classify_knn import FEATURE_STD_FLOORS, KnnClassifier
from cube_task.perception.features import FEATURE_NAMES, HUE_BINS
from cube_task.sim.teach import collect_sim_samples

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _with_hue_mass(X, bin_index, frac):
    """Return X with frac of every row's hue histogram moved into bin_index (sum stays 1)."""
    Q = np.array(X, dtype=np.float64, copy=True)
    Q[:, :HUE_BINS] *= (1.0 - frac)
    Q[:, bin_index] += frac
    return Q


def _synthetic(rng, n_per_class=8):
    """Three fruit-like classes whose hue mass never falls in bins 5-7 (cyan / blue)."""
    protos = {
        'tomato': dict(hue={0: 0.7, 11: 0.3}, s=0.8, v=0.6, a=0.45, b=0.3, lwh=(0.06, 0.06, 0.05),
                       fill=0.78, circ=0.75),
        'lemon': dict(hue={1: 0.4, 2: 0.6}, s=0.75, v=0.85, a=0.0, b=0.55, lwh=(0.07, 0.05, 0.05),
                      fill=0.76, circ=0.7),
        'cucumber': dict(hue={3: 0.6, 4: 0.4}, s=0.6, v=0.4, a=-0.3, b=0.25,
                         lwh=(0.15, 0.035, 0.035), fill=0.8, circ=0.45),
    }
    X, y = [], []
    for lab, p in protos.items():
        for _ in range(n_per_class):
            v = np.zeros(len(FEATURE_NAMES))
            h = np.zeros(HUE_BINS)
            for b, m in p['hue'].items():
                h[b] = m
            h = np.clip(h + rng.normal(0.0, 0.02, HUE_BINS) * (h > 0), 0.0, None)
            v[:HUE_BINS] = h / h.sum()
            v[IDX['s_mean']] = p['s'] + rng.normal(0, 0.02)
            v[IDX['v_mean']] = p['v'] + rng.normal(0, 0.02)
            v[IDX['s_std']], v[IDX['v_std']] = 0.04 + rng.normal(0, 0.005), 0.12
            v[IDX['lab_a']], v[IDX['lab_b']] = p['a'] + rng.normal(0, 0.02), p['b']
            L, W, H = (np.array(p['lwh']) + rng.normal(0, 0.002, 3))
            v[IDX['length']], v[IDX['width']], v[IDX['height']] = L, W, H
            v[IDX['aspect']] = L / W
            v[IDX['fill']], v[IDX['circularity']] = p['fill'], p['circ'] + rng.normal(0, 0.02)
            X.append(v)
            y.append(lab)
    return np.array(X), y


def test_floors_cover_every_feature():
    assert set(FEATURE_STD_FLOORS) == set(FEATURE_NAMES)
    knn = KnnClassifier()
    assert knn.feature_floors == FEATURE_STD_FLOORS


def test_two_percent_mass_in_an_empty_bin_still_classifies_synthetic():
    rng = np.random.default_rng(5)
    X, y = _synthetic(rng)
    Xh, yh = _synthetic(np.random.default_rng(99), n_per_class=3)     # held out
    assert not np.any(X[:, 5:8])                                     # bins 5-7 empty
    knn = KnnClassifier(k=3, max_distance=3.0).fit(X, y, FEATURE_NAMES)
    assert np.all(knn.std[5:8] == 0.05)
    for b in (5, 6, 7):
        labels, scores, dists = knn.predict(_with_hue_mass(Xh, b, 0.02))
        assert labels == yh and np.all(scores >= 0.55), (b, labels, scores, dists)
    # The old 1e-3 floor turned exactly this into 'unknown'.
    old = KnnClassifier(k=3, max_distance=3.0, feature_floors={}).fit(X, y, FEATURE_NAMES)
    assert old.predict(Xh)[0] == yh
    assert set(old.predict(_with_hue_mass(Xh, 6, 0.02))[0]) == {'unknown'}
    # A different object is still rejected: twice the size and another color.
    far = Xh[:1].copy()
    far[0, IDX['length']:IDX['height'] + 1] *= 2.0
    far[0, :HUE_BINS] = 0.0
    far[0, 8] = 1.0
    assert knn.predict(far)[0] == ['unknown']


def test_old_model_files_load_and_get_the_floors(tmp_path):
    X, y = _synthetic(np.random.default_rng(1))
    old = KnnClassifier(feature_floors={}).fit(X, y, FEATURE_NAMES)
    path = old.save(tmp_path / 'knn.npz', created='2026-01-01T00:00:00')
    z = np.load(path, allow_pickle=False)
    assert int(z['version']) == 1 and np.allclose(z['std'][5:8], 1e-3)
    knn = KnnClassifier.load(path)
    assert knn.n_samples == len(y) and np.allclose(knn.std, KnnClassifier().fit(
        X, y, FEATURE_NAMES).std)
    assert np.all(knn.std[5:8] == 0.05)
    same = KnnClassifier.load(path, feature_floors={})
    assert np.allclose(same.std, old.std)
    knn.save(tmp_path / 'again.npz')
    assert np.allclose(KnnClassifier.load(tmp_path / 'again.npz').std, knn.std)
    with pytest.raises(ValueError, match='feature_floors'):
        KnnClassifier(feature_floors={'length': -1.0})
    # Unknown feature names keep the plain std_floor.
    custom = KnnClassifier(std_floor=0.01).fit(X[:, :3] * 0.0, y, ['f1', 'f2', 'f3'])
    assert np.allclose(custom.std, 0.01)


def test_held_out_sim_objects_with_two_percent_cyan_still_classify():
    X, y, names = collect_sim_samples(range(100, 112))
    Xh, yh, _ = collect_sim_samples(range(300, 304))
    assert names == FEATURE_NAMES and len(y) > 40 and len(yh) > 10
    assert np.max(X[:, 5:8]) < 0.005                       # no cyan / blue in training
    knn = KnnClassifier(k=3, max_distance=3.0).fit(X, y, names)
    base, _, _ = knn.predict(Xh)
    assert 'unknown' not in base
    assert sum(a == b for a, b in zip(base, yh)) >= 0.85 * len(yh)
    for b in (5, 6, 7):
        labels, scores, _ = knn.predict(_with_hue_mass(Xh, b, 0.02))
        assert labels == base and np.all(scores >= 0.55), b
    old = KnnClassifier(k=3, max_distance=3.0, feature_floors={}).fit(X, y, names)
    assert set(old.predict(_with_hue_mass(Xh, 6, 0.02))[0]) == {'unknown'}
