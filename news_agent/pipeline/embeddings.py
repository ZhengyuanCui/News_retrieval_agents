from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_lock = threading.Lock()


def get_model():
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info("Loading sentence-transformers model '%s'", _MODEL_NAME)
                    _model = SentenceTransformer(_MODEL_NAME)
                except ImportError:
                    logger.warning(
                        "sentence-transformers not installed; semantic features disabled"
                    )
    return _model
