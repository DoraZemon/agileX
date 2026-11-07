"""
Build the classifier, detector and Perception pipeline from the config.

Backend selection: tasks.<task>.classifier when set, else
perception.classifier.backend. Backends: 'color' (HSV ranges), 'knn'
(trained model file), 'onnx' (2D detector only), 'onnx+knn' (detector, KNN
for instances the detector misses).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .classify_color import ColorClassifier
from .classify_knn import KnnClassifier
from .pipeline import Perception
from ..core.interfaces import Detector2D, InstanceClassifier

LOG = logging.getLogger('cube_task.perception.factory')

BACKENDS = ('color', 'knn', 'onnx', 'onnx+knn')


def classifier_backend(cfg: Any, task: Optional[str]) -> str:
    """Return the classifier backend name for task; raise ValueError if unknown."""
    b = cfg.get('tasks.%s.classifier' % task, None) if task else None
    if not b:
        b = cfg.get('perception.classifier.backend', 'color')
    b = str(b).strip().lower()
    if b not in BACKENDS:
        raise ValueError('unknown classifier backend %r (expected one of %s)'
                         % (b, ', '.join(BACKENDS)))
    return b


def task_labels(cfg: Any, task: Optional[str]) -> list:
    """Return tasks.<task>.labels as a list of strings (empty when unset)."""
    if not task:
        return []
    labels = cfg.get('tasks.%s.labels' % task, None) or []
    return [str(v) for v in labels]


def make_classifier(cfg: Any, task: Optional[str]) -> Optional[InstanceClassifier]:
    """
    Return the instance classifier for task, or None for the 'onnx' backend.

    'color': ColorClassifier with perception.classifier.color.classes,
    restricted to tasks.<task>.labels when that list is non-empty (every
    label needs ranges, else ValueError). 'knn' / 'onnx+knn': KnnClassifier
    loaded from perception.classifier.knn.model_file; FileNotFoundError when
    the file is missing.
    """
    backend = classifier_backend(cfg, task)
    if backend == 'color':
        classes = cfg.get('perception.classifier.color.classes', None) or {}
        if not classes:
            raise ValueError('perception.classifier.color.classes is empty; run color_calib')
        labels = task_labels(cfg, task)
        return ColorClassifier(classes,
                               float(cfg.get('perception.classifier.color.min_fraction', 0.35)),
                               allowed_labels=labels or None)
    if backend in ('knn', 'onnx+knn'):
        path = cfg.path('perception.classifier.knn.model_file')
        if path is None:
            raise ValueError('perception.classifier.knn.model_file is not set')
        if not path.is_file():
            raise FileNotFoundError(
                'KNN model %s not found: run "teach_classes" to record samples for task %s '
                'first, or set tasks.%s.classifier: color' % (path, task, task))
        return KnnClassifier.load(path, k=int(cfg.get('perception.classifier.knn.k', 3)),
                                  max_distance=float(cfg.get(
                                      'perception.classifier.knn.max_distance', 3.0)))
    return None


def make_detector(cfg: Any, task: Optional[str]) -> Optional[Detector2D]:
    """
    Return the ONNX 2D detector when the backend contains 'onnx', else None.

    Imports detector_onnx lazily (it needs optional runtime pieces); an
    ImportError is re-raised with a hint because the configured backend
    cannot run without it.
    """
    if 'onnx' not in classifier_backend(cfg, task):
        return None
    try:
        from .detector_onnx import OnnxDetector
    except ImportError as e:
        raise ImportError('classifier backend needs perception/detector_onnx.py: %s' % e) from e
    return OnnxDetector(cfg)


def make_perception(cfg: Any, task: Optional[str]) -> Perception:
    """Build Perception(cfg, make_classifier(cfg, task), make_detector(cfg, task))."""
    classifier = make_classifier(cfg, task)
    detector = make_detector(cfg, task)
    LOG.info('perception for %s: backend %s, classifier %s, detector %s', task,
             classifier_backend(cfg, task), type(classifier).__name__ if classifier else None,
             type(detector).__name__ if detector else None)
    return Perception(cfg, classifier=classifier, detector=detector)
