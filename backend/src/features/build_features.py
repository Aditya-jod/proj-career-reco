import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = Path("models/hf/all-MiniLM-L6-v2")


def _resolve_model_source(
    model_name: str,
    local_model_path: Union[str, Path, None] = None,
) -> str:
    """Return the best available model source path or name."""
    env_path = os.getenv("SENTENCE_BERT_MODEL_DIR")
    candidates = [local_model_path, env_path, DEFAULT_LOCAL_MODEL]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return model_name


@lru_cache(maxsize=1)
def _load_sentence_transformer(model_source: str, device: str) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model process-wide (one load per process)."""
    logger.info("Loading Sentence-BERT model from %s", model_source)
    return SentenceTransformer(model_source, device=device)


class FeatureBuilder:
    """
    Sentence-BERT text encoder.

    The underlying SentenceTransformer is loaded once per process via
    ``lru_cache`` so multiple FeatureBuilder instances share a single model
    without hidden class-level mutable state.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        local_model_path: Union[str, Path, None] = None,
    ) -> None:
        model_source = _resolve_model_source(model_name, local_model_path)
        self._model: SentenceTransformer = _load_sentence_transformer(model_source, device)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

