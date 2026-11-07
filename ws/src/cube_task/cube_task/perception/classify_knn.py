"""
k-nearest-neighbor instance classifier on features.feature_vector.

Distances are RMS standardized distances: each feature is divided by its
training standard deviation (at least its FEATURE_STD_FLOORS value, on the
feature's physical scale) and the Euclidean distance is divided by
sqrt(number of features), so max_distance means "average deviation per
feature in standard deviations" and does not depend on the feature count.
"""
from __future__ import annotations

import datetime
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .features import FEATURE_NAMES, feature_vector, HUE_BINS
from ..core.interfaces import InstanceClassifier
from ..core.types import Detection, Frame

LOG = logging.getLogger('cube_task.perception.classify_knn')

MODEL_VERSION = 1

#: Lower bound of the standardization std of each feature, on the feature's
#: physical scale: a difference this large is a small, plausible change
#: between the teaching day and the competition (white balance, a few pixels
#: of a blue tray, 3 mm of depth noise). Without it a feature that is
#: constant in training (a hue bin no training object has, e.g. cyan / blue
#: for fruit and vegetables) gets the 1e-3 floor and 2 % of hue mass in that
#: bin alone gives an RMS distance above 4, so the object becomes 'unknown'.
FEATURE_STD_FLOORS: Dict[str, float] = dict(
    [('hue_%02d' % i, 0.05) for i in range(HUE_BINS)]
    + [('s_mean', 0.05), ('v_mean', 0.05), ('s_std', 0.02), ('v_std', 0.02),
       ('lab_a', 0.05), ('lab_b', 0.05), ('length', 0.003), ('width', 0.003),
       ('height', 0.003), ('aspect', 0.1), ('fill', 0.05), ('circularity', 0.05)])


class KnnClassifier(InstanceClassifier):
    """
    Distance-weighted k-NN with unknown rejection.

    k: neighbors that vote (capped at the sample count). max_distance: RMS
    standardized distance to the nearest sample above which the label is
    'unknown'. std_floor: lower bound of every feature's standard deviation
    so constant training features do not explode distances; feature_floors:
    {feature name: floor} raising that bound per feature (None =
    FEATURE_STD_FLOORS, {} = std_floor only). score = vote share of the
    winner (weights 1 / (d + 1e-6)) times 1 / (1 + (d_nearest /
    max_distance)^2); score_kind = 'knn'.
    """

    def __init__(self, k: int = 3, max_distance: float = 3.0, std_floor: float = 1e-3,
                 feature_floors: Optional[Dict[str, float]] = None):
        if int(k) < 1:
            raise ValueError('k must be >= 1')
        if not (float(max_distance) > 0.0):
            raise ValueError('max_distance must be > 0')
        if not (float(std_floor) > 0.0):
            raise ValueError('std_floor must be > 0')
        floors = FEATURE_STD_FLOORS if feature_floors is None else feature_floors
        self.feature_floors: Dict[str, float] = {str(n): float(v) for n, v in floors.items()}
        if not all(math.isfinite(v) and v >= 0.0 for v in self.feature_floors.values()):
            raise ValueError('feature_floors must be finite and >= 0')
        self.k = int(k)
        self.max_distance = float(max_distance)
        self.std_floor = float(std_floor)
        self.X = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
        self.y: List[str] = []
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.mean = np.zeros(len(FEATURE_NAMES))
        self.std = np.ones(len(FEATURE_NAMES))
        self.created = ''
        self.labels: List[str] = []

    # ------------------------------------------------------------------ training

    @property
    def n_samples(self) -> int:
        """Return the number of training samples."""
        return int(self.X.shape[0])

    def _check_xy(self, X: Any, y: Sequence[str], dim: int) -> Tuple[np.ndarray, List[str]]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        y = [str(v) for v in y]
        if X.ndim != 2 or X.shape[1] != dim:
            raise ValueError('X must be N x %d, got %s' % (dim, X.shape))
        if X.shape[0] != len(y):
            raise ValueError('X has %d rows but y has %d labels' % (X.shape[0], len(y)))
        if not np.all(np.isfinite(X)):
            raise ValueError('training features must be finite (drop samples with NaN)')
        if any(not v or v == 'unknown' for v in y):
            raise ValueError("training labels must be non-empty and not 'unknown'")
        return X, y

    def std_floors(self) -> np.ndarray:
        """Return the std lower bound of each feature (feature_names order)."""
        return np.array([max(self.std_floor, self.feature_floors.get(n, 0.0))
                         for n in self.feature_names], dtype=np.float64)

    def _update_stats(self) -> None:
        if self.n_samples == 0:
            raise ValueError('no training samples')
        self.mean = self.X.mean(axis=0)
        self.std = np.maximum(self.X.std(axis=0), self.std_floors())
        self.labels = sorted(set(self.y))

    def fit(self, X: Any, y: Sequence[str],
            feature_names: Optional[Sequence[str]] = None) -> 'KnnClassifier':
        """Replace the training set with X (N x D) and labels y; return self."""
        names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        X, y = self._check_xy(X, y, len(names))
        if X.shape[0] == 0:
            raise ValueError('no training samples')
        self.feature_names = names
        self.X, self.y = X.copy(), list(y)
        self._update_stats()
        return self

    def add_samples(self, X: Any, y: Sequence[str]) -> 'KnnClassifier':
        """Append samples (same features) and recompute the standardization; return self."""
        X, y = self._check_xy(X, y, len(self.feature_names))
        self.X = np.vstack([self.X, X])
        self.y = self.y + list(y)
        self._update_stats()
        return self

    # ------------------------------------------------------------------ inference

    def _distances(self, Xq: np.ndarray) -> np.ndarray:
        A = (Xq - self.mean) / self.std
        B = (self.X - self.mean) / self.std
        d2 = (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
              - 2.0 * A @ B.T)
        return np.sqrt(np.maximum(d2, 0.0) / A.shape[1])

    def _vote(self, d: np.ndarray, labels: List[str]) -> Tuple[str, float, Dict[str, Any]]:
        k = min(self.k, d.size)
        idx = np.argsort(d, kind='stable')[:k]
        votes: Dict[str, float] = {}
        for i in idx:
            votes[labels[i]] = votes.get(labels[i], 0.0) + 1.0 / (float(d[i]) + 1e-6)
        total = sum(votes.values())
        best = max(votes, key=lambda q: (votes[q], q))
        d0 = float(d[idx[0]])
        share = votes[best] / total
        score = share / (1.0 + (d0 / self.max_distance) ** 2)
        info = {'knn_distance': round(d0, 3), 'knn_share': round(share, 3),
                'knn_neighbors': [labels[i] for i in idx]}
        if d0 > self.max_distance:
            info['classify_reason'] = 'too_far'
            return 'unknown', float(score), info
        return best, float(score), info

    def predict(self, X: Any) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Classify feature rows X (N x D).

        Returns (labels, scores in [0, 1], nearest RMS standardized distance).
        Rows containing NaN get 'unknown' with score 0 and distance inf.
        """
        if self.n_samples == 0:
            raise ValueError('KnnClassifier has no training samples')
        Xq = np.asarray(X, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        if Xq.shape[1] != len(self.feature_names):
            raise ValueError('expected %d features, got %d'
                             % (len(self.feature_names), Xq.shape[1]))
        labels = ['unknown'] * Xq.shape[0]
        scores = np.zeros(Xq.shape[0])
        dists = np.full(Xq.shape[0], np.inf)
        ok = np.all(np.isfinite(Xq), axis=1)
        if np.any(ok):
            D = self._distances(Xq[ok])
            for row, i in enumerate(np.nonzero(ok)[0]):
                lab, sc, info = self._vote(D[row], self.y)
                labels[i], scores[i], dists[i] = lab, sc, info['knn_distance']
        return labels, scores, dists

    def classify(self, frame: Frame, detections: List[Detection]) -> None:
        """Fill label / score / score_kind of each detection in place."""
        if not detections:
            return
        if self.n_samples == 0:
            raise ValueError('KnnClassifier has no training samples')
        feats = [feature_vector(frame, d)[0] for d in detections]
        X = np.vstack(feats).astype(np.float64)
        ok = np.all(np.isfinite(X), axis=1)
        D = self._distances(X[ok]) if np.any(ok) else None
        row = 0
        for det, good in zip(detections, ok):
            det.score_kind = 'knn'
            if not good:
                det.label, det.score = 'unknown', 0.0
                det.extra['classify_reason'] = 'missing_features'
                continue
            lab, sc, info = self._vote(D[row], self.y)
            row += 1
            det.label, det.score = lab, sc
            det.extra.update(info)

    # ------------------------------------------------------------------ evaluation

    def leave_one_out(self) -> Dict[str, Any]:
        """
        Leave-one-out evaluation on the training set.

        Each sample is classified by the others (standardization from the full
        set). Returns {'n', 'accuracy', 'n_unknown', 'per_class': {label:
        {'correct', 'total', 'accuracy'}}, 'confusion': {true: {pred: count}}}.
        """
        n = self.n_samples
        rep: Dict[str, Any] = {'n': n, 'accuracy': float('nan'), 'n_unknown': 0,
                               'per_class': {}, 'confusion': {}}
        if n < 2:
            return rep
        D = self._distances(self.X)
        correct = 0
        for i in range(n):
            keep = np.arange(n) != i
            labels = [self.y[j] for j in range(n) if j != i]
            pred, _, _ = self._vote(D[i, keep], labels)
            t = self.y[i]
            pc = rep['per_class'].setdefault(t, {'correct': 0, 'total': 0, 'accuracy': 0.0})
            pc['total'] += 1
            row = rep['confusion'].setdefault(t, {})
            row[pred] = row.get(pred, 0) + 1
            if pred == 'unknown':
                rep['n_unknown'] += 1
            if pred == t:
                correct += 1
                pc['correct'] += 1
        for pc in rep['per_class'].values():
            pc['accuracy'] = pc['correct'] / float(pc['total'])
        rep['accuracy'] = correct / float(n)
        return rep

    # ------------------------------------------------------------------ persistence

    def save(self, path: Union[str, Path], created: Optional[str] = None) -> Path:
        """
        Write the model to an .npz file atomically and return its path.

        Stored arrays: X, y, mean, std, feature_names, version, created (ISO
        time string; pass created for reproducible files).
        """
        if self.n_samples == 0:
            raise ValueError('cannot save an empty model')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.created = created if created is not None else \
            datetime.datetime.now().isoformat(timespec='seconds')
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'wb') as f:
            np.savez(f, X=self.X, y=np.array(self.y, dtype=str), mean=self.mean,
                     std=self.std, feature_names=np.array(self.feature_names, dtype=str),
                     version=np.array(MODEL_VERSION), created=np.array(self.created))
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path], k: int = 3, max_distance: float = 3.0,
             std_floor: float = 1e-3,
             feature_floors: Optional[Dict[str, float]] = None) -> 'KnnClassifier':
        """
        Load a model written by save().

        Raises FileNotFoundError when the file is missing and ValueError when
        it is malformed, has another version, or was trained with features
        different from the current features.FEATURE_NAMES. mean / std are
        recomputed from X with the current floors (identical to the stored
        ones for the same std_floor / feature_floors), so a model saved
        before FEATURE_STD_FLOORS existed loads unchanged and gets them.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('KNN model not found: %s' % path)
        with np.load(path, allow_pickle=False) as z:
            missing = [k_ for k_ in ('X', 'y', 'mean', 'std', 'feature_names', 'version')
                       if k_ not in z.files]
            if missing:
                raise ValueError('%s: missing arrays %s' % (path, missing))
            version = int(z['version'])
            if version != MODEL_VERSION:
                raise ValueError('%s: model version %d, expected %d; re-run teach_classes'
                                 % (path, version, MODEL_VERSION))
            names = [str(s) for s in z['feature_names']]
            X = np.asarray(z['X'], dtype=np.float64)
            y = [str(s) for s in z['y']]
            created = str(z['created']) if 'created' in z.files else ''
        if names != list(FEATURE_NAMES):
            raise ValueError('%s: trained with different features; re-run teach_classes'
                             % path)
        m = cls(k=k, max_distance=max_distance, std_floor=std_floor,
                feature_floors=feature_floors)
        m.fit(X, y, names)
        m.created = created
        LOG.info('loaded KNN model %s: %d samples, labels %s', path, m.n_samples, m.labels)
        return m


def nan_free(X: np.ndarray) -> np.ndarray:
    """Return the bool mask of rows of X without NaN / inf (to filter training data)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    return np.all(np.isfinite(X), axis=1)
