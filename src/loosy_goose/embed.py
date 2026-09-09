from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]

# Two distinct defaults on purpose: selection and scoring must never share a model, or the
# distortion metric grades the selector's own similarity function (self-grading).
SELECT_MODEL = "BAAI/bge-small-en-v1.5"
SCORE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _model(name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def embed_texts(
    texts: list[str], model_name: str = SELECT_MODEL, batch_size: int = 32
) -> FloatArray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float64)
    vectors = _model(model_name).encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float64)
