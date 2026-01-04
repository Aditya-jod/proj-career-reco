import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class FeatureBuilder:
    _model = None 

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        if FeatureBuilder._model is None:
            logger.info(f"Loading Sentence-BERT model: {model_name}")
            FeatureBuilder._model = SentenceTransformer(model_name, device=device)

        self.model = FeatureBuilder._model

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
