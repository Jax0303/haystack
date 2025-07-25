# SPDX-FileCopyrightText: 2025-present <your_name_or_org>
#
# SPDX-License-Identifier: Apache-2.0

"""Gemini Pro text generation component.

This component provides a minimal wrapper around the Google Generative AI
(Gemini-Pro) model so that it can be used inside Haystack pipelines just like
`OpenAIGenerator`. It focuses on **text-in / text-out** usage and supports an
*automatic API-key fallback*: a list of keys can be supplied; if a request
fails due to quota exhaustion or rate-limit errors the component will
transparently switch to the next key.

Only the synchronous `run()` interface is implemented for now. If you need to
stream responses or use the async API you can extend this class further.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret, deserialize_secrets_inplace

# Google GenerativeAI is an optional dependency
with LazyImport(message="Run 'pip install google-generativeai>=0.4.0' to use GeminiProGenerator") as genai_import:
    import google.generativeai as genai  # noqa: F401 – imported lazily

logger = logging.getLogger(__name__)


@component
class GeminiProGenerator:
    """Generate text with Google Gemini-Pro LLM.

    Parameters
    ----------
    api_keys
        One or more API keys. The component will iterate over these keys if it
        hits rate-limits / quota errors so that long-running experiments can
        continue automatically.
    model
        Gemini model name. Defaults to ``"gemini-pro"``.
    generation_kwargs
        Extra keyword arguments forwarded to ``GenerativeModel.generate_content``.
    """

    def __init__(
        self,
        api_keys: List[Secret],
        model: str = "gemini-pro",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        genai_import.check()

        if not api_keys:
            raise ValueError("At least one API key is required for GeminiProGenerator.")

        self._api_keys = api_keys
        self._model_name = model
        self._generation_kwargs = generation_kwargs or {}

        # Runtime attributes (not persisted)
        self._current_key_idx: int = 0
        self._model = None  # type: ignore
        self._init_client(key_idx=0)

    # ---------------------------------------------------------------------
    # Haystack serialization helpers
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            api_keys=[key.to_dict() for key in self._api_keys],
            model=self._model_name,
            generation_kwargs=self._generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiProGenerator":
        init_params = data.get("init_parameters", {})
        # Deserialize Secret objects
        if "api_keys" in init_params:
            init_params["api_keys"] = [Secret.from_dict(k) for k in init_params["api_keys"]]
        deserialize_secrets_inplace(init_params, keys=[])  # Already handled above
        return default_from_dict(cls, data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _init_client(self, *, key_idx: int) -> None:
        """Configure google-generativeai client for the given key index."""
        import google.generativeai as genai

        key_val = self._api_keys[key_idx].resolve_value()
        genai.configure(api_key=key_val)
        self._model = genai.GenerativeModel(self._model_name)
        self._current_key_idx = key_idx
        logger.debug("GeminiProGenerator initialised with key #%d", key_idx)

    def _switch_key(self) -> bool:
        """Switch to the next API key.

        Returns ``True`` if a new key was activated, ``False`` if no more keys
        are available.
        """
        next_idx = self._current_key_idx + 1
        if next_idx >= len(self._api_keys):
            return False
        self._init_client(key_idx=next_idx)
        return True

    # ------------------------------------------------------------------
    # Haystack runtime interface
    # ------------------------------------------------------------------
    @component.output_types(replies=List[str])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """Generate a single completion for *prompt*.

        The method will try each API key until the request succeeds or all keys
        are exhausted.
        """
        gen_kwargs = {**self._generation_kwargs, **(generation_kwargs or {})}

        while True:
            try:
                # We use a single-message prompt; Gemini accepts plain strings.
                response = self._model.generate_content(prompt, **gen_kwargs)
                # The library returns a `GenerateContentResponse`. Use `.text`.
                text = response.text  # type: ignore[attr-defined]
                return {"replies": [text]}

            except Exception as err:  # Broad catch – libraries raise diverse errors
                err_msg = str(err)
                # Detect quota/rate-limit; fall back to next key if available
                quota_exhausted = any(word in err_msg.lower() for word in ["quota", "rate", "exceed", "limit"])

                if quota_exhausted and self._switch_key():
                    logger.warning("GeminiProGenerator: quota error – switching to next API key.")
                    continue
                # No fallback possible → propagate
                raise

    # ------------------------------------------------------------------
    # Telemetry (optional override)
    # ------------------------------------------------------------------
    def _get_telemetry_data(self) -> Dict[str, Any]:
        return {"model": self._model_name} 