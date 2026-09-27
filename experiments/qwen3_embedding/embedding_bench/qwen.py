"""The real Qwen3-Embedding-0.6B embedder. The only module that imports torch.

CPU only, full float32, one text per forward pass (no padding), L2-normalized
output. The caller formats queries (``retrieval.format_query``); this class
embeds exactly the text it is given, with no prompt of its own.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
_INPUT_PROBE = "zomah input probe"


class QwenEmbedder:
    def __init__(self, model_id: str = MODEL_ID, *, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        # local_files_only: load time must never include a download
        # (run `run_benchmark.py --download-only` first).
        self._model = SentenceTransformer(
            model_id, device="cpu", revision=revision, local_files_only=True
        )
        self.loaded_dtype = str(next(self._model.parameters()).dtype)
        # Pin precision so the checkpoint dtype / transformers default is not a variable.
        self._model.to(torch.float32)
        self.input_probe = self._check_plain_text_input()

    def embed(self, text: str) -> list[float]:
        # prompt="" also bypasses any default prompt in the model's ST config.
        vectors = self._model.encode(
            [text],
            prompt="",
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors[0].tolist()

    def token_count(self, text: str) -> int:
        return len(self._model.tokenizer(text)["input_ids"])

    @property
    def max_tokens(self) -> int | None:
        return self._model.max_seq_length

    def describe(self) -> dict[str, Any]:
        return {
            "embedder": "qwen3-embedding",
            "model_id": self.model_id,
            "revision_requested": self.revision,
            "snapshot": cached_snapshot(self.model_id, self.revision),
            "device": str(self._model.device),
            "dtype_loaded": self.loaded_dtype,
            "dtype_used": str(next(self._model.parameters()).dtype),
            "torch_threads": torch.get_num_threads(),
            "modules": [type(module).__name__ for module in self._model],
            "pooling": _pooling_mode(self._model),
            "input_probe": self.input_probe,
            "packages": package_versions(),
        }

    def _check_plain_text_input(self) -> str:
        """Fail loudly if sentence-transformers would rewrite our text.

        sentence-transformers 5 wraps inputs in the tokenizer's chat template
        when a model declares a "message" modality. The documented Qwen3-Embedding
        input is the raw string, so any wrapping would invalidate the run.
        """

        features = self._model.preprocess([_INPUT_PROBE], prompt="")
        ids = features["input_ids"][0]
        plain = self._model.tokenizer.decode(ids, skip_special_tokens=True)
        if plain.strip() != _INPUT_PROBE:
            raise RuntimeError(
                "sentence-transformers changed the input text before embedding "
                f"({plain!r}); refusing to run a benchmark on rewritten inputs"
            )
        return self._model.tokenizer.decode(ids, skip_special_tokens=False)


def download(model_id: str = MODEL_ID, *, revision: str | None = None) -> str | None:
    """Fetch exactly what sentence-transformers loads into the Hugging Face cache."""

    SentenceTransformer(model_id, device="cpu", revision=revision)
    return cached_snapshot(model_id, revision)


def is_available_offline(model_id: str, revision: str | None = None) -> bool:
    if Path(model_id).expanduser().is_dir():
        return True
    return cached_snapshot(model_id, revision) is not None


def cached_snapshot(model_id: str, revision: str | None = None) -> str | None:
    """The cached snapshot commit for a Hub model id, or None (e.g. a local dir)."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        path = try_to_load_from_cache(model_id, "modules.json", revision=revision)
    except Exception:  # invalid repo ids (e.g. local paths) raise validation errors
        return None
    if not isinstance(path, str):
        return None
    parts = Path(path).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "sentence-transformers", "tokenizers", "huggingface-hub", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _pooling_mode(model: SentenceTransformer) -> str | None:
    for module in model:
        mode = getattr(module, "pooling_mode", None)
        if isinstance(mode, str):
            return mode
    return None
