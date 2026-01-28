import logging
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


DEFAULT_LOCAL_MODEL = Path("models/hf/all-MiniLM-L6-v2")


class FeatureBuilder:
    _model = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        local_model_path: str | Path | None = None,
    ):
        if FeatureBuilder._model is None:
            model_source = self._resolve_model_source(model_name, local_model_path)
            logger.info("Loading Sentence-BERT model from %s", model_source)
            FeatureBuilder._model = SentenceTransformer(model_source, device=device)

        self.model = FeatureBuilder._model

    @staticmethod
    def _resolve_model_source(model_name: str, local_model_path: str | Path | None) -> str:
        env_path = os.getenv("SENTENCE_BERT_MODEL_DIR")
        candidates = [local_model_path, env_path, DEFAULT_LOCAL_MODEL]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                return str(path)
        return model_name

    def encode(self, texts, batch_size=64, normalize=True):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings
