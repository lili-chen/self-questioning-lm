"""
Sampler module for V-SQLM.

Provides unified interface for sampling solutions from different LLM backends.
"""

from typing import TypedDict, Any, Protocol
import random
import logging

logger = logging.getLogger(__name__)


class Sample(TypedDict):
    """A single sampled solution from the solver."""
    answer: str
    rationale: str
    confidence: float  # NaN if not available
    meta: dict


class SolverBackend(Protocol):
    """Protocol defining the interface for solver backends."""

    def sample(
        self,
        prompt: str,
        seed: int | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        variant: str = "cot"
    ) -> Sample:
        """
        Sample a single solution from the solver.

        Parameters
        ----------
        prompt : str
            The input problem prompt.
        seed : int | None
            Random seed for reproducibility.
        temperature : float
            Sampling temperature (default: 0.6).
        top_p : float | None
            Nucleus sampling parameter.
        variant : str
            Prompting variant: 'cot' (chain-of-thought), 'concise', etc.

        Returns
        -------
        Sample
            A sampled solution with answer, rationale, confidence, and metadata.
        """
        ...


class DummySolverBackend:
    """
    Dummy solver backend for testing.

    Returns synthetic solutions for testing purposes.
    """

    def __init__(self, name: str = "dummy"):
        """
        Initialize dummy solver.

        Parameters
        ----------
        name : str
            Name identifier for this backend.
        """
        self.name = name

    def sample(
        self,
        prompt: str,
        seed: int | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        variant: str = "cot"
    ) -> Sample:
        """Sample a dummy solution."""
        if seed is not None:
            random.seed(seed)

        # Generate synthetic answer based on prompt hash for determinism
        answer_num = hash(prompt + str(seed)) % 100

        return Sample(
            answer=f"{answer_num}",
            rationale=f"[{variant}] This is a synthetic rationale for testing.",
            confidence=float('nan'),
            meta={"backend": self.name, "seed": seed, "variant": variant}
        )


class TransformersBackend:
    """
    HuggingFace Transformers backend for local models.

    Uses transformers library to run inference on local models.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 512
    ):
        """
        Initialize Transformers backend.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        device : str
            Device to run on ('cuda' or 'cpu').
        max_new_tokens : int
            Maximum tokens to generate.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info(f"Initializing TransformersBackend with model: {model_name}")
        # Lazy import to avoid dependency issues
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        except ImportError:
            logger.warning("transformers not installed, backend will fail at inference")
            self.model = None
            self.tokenizer = None

    def sample(
        self,
        prompt: str,
        seed: int | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        variant: str = "cot"
    ) -> Sample:
        """Sample solution using HuggingFace model."""
        if self.model is None:
            raise RuntimeError("transformers library not available")

        import torch

        if seed is not None:
            torch.manual_seed(seed)

        # Format prompt based on variant
        formatted_prompt = self._format_prompt(prompt, variant)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p if top_p is not None else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse answer and rationale
        answer, rationale = self._parse_output(generated)

        return Sample(
            answer=answer,
            rationale=rationale,
            confidence=float('nan'),
            meta={
                "backend": "transformers",
                "model": self.model_name,
                "seed": seed,
                "variant": variant
            }
        )

    def _format_prompt(self, prompt: str, variant: str) -> str:
        """Format prompt based on variant."""
        if variant == "cot":
            return f"{prompt}\n\nLet's think step by step:"
        elif variant == "concise":
            return f"{prompt}\n\nAnswer concisely:"
        else:
            return prompt

    def _parse_output(self, text: str) -> tuple[str, str]:
        """
        Parse generated text into answer and rationale.

        Simple heuristic: last line is answer, rest is rationale.
        """
        lines = text.strip().split('\n')
        if len(lines) == 0:
            return "", ""
        elif len(lines) == 1:
            return lines[0], ""
        else:
            return lines[-1], '\n'.join(lines[:-1])


class OpenAIBackend:
    """
    OpenAI API backend for GPT models.

    Uses OpenAI API to run inference on GPT models.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        max_tokens: int = 512
    ):
        """
        Initialize OpenAI backend.

        Parameters
        ----------
        model_name : str
            OpenAI model identifier (e.g., 'gpt-3.5-turbo', 'gpt-4').
        api_key : str | None
            OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        max_tokens : int
            Maximum tokens to generate.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens

        logger.info(f"Initializing OpenAIBackend with model: {model_name}")

        try:
            import openai
            if api_key:
                openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai library not installed, backend will fail at inference")
            self.client = None

    def sample(
        self,
        prompt: str,
        seed: int | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        variant: str = "cot"
    ) -> Sample:
        """Sample solution using OpenAI API."""
        if self.client is None:
            raise RuntimeError("openai library not available")

        formatted_prompt = self._format_prompt(prompt, variant)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=top_p if top_p is not None else 1.0,
            seed=seed
        )

        generated = response.choices[0].message.content

        # Parse answer and rationale
        answer, rationale = self._parse_output(generated)

        return Sample(
            answer=answer,
            rationale=rationale,
            confidence=float('nan'),
            meta={
                "backend": "openai",
                "model": self.model_name,
                "seed": seed,
                "variant": variant,
                "usage": response.usage.model_dump() if response.usage else {}
            }
        )

    def _format_prompt(self, prompt: str, variant: str) -> str:
        """Format prompt based on variant."""
        if variant == "cot":
            return f"{prompt}\n\nLet's think step by step:"
        elif variant == "concise":
            return f"{prompt}\n\nAnswer concisely:"
        else:
            return prompt

    def _parse_output(self, text: str) -> tuple[str, str]:
        """Parse generated text into answer and rationale."""
        lines = text.strip().split('\n')
        if len(lines) == 0:
            return "", ""
        elif len(lines) == 1:
            return lines[0], ""
        else:
            return lines[-1], '\n'.join(lines[:-1])


def create_backend(backend_type: str, **kwargs) -> SolverBackend:
    """
    Factory function to create solver backends from configuration.

    Parameters
    ----------
    backend_type : str
        Type of backend: 'dummy', 'transformers', 'openai'.
    **kwargs
        Additional arguments passed to the backend constructor.

    Returns
    -------
    SolverBackend
        Initialized solver backend.

    Examples
    --------
    >>> backend = create_backend('dummy', name='test')
    >>> backend = create_backend('transformers', model_name='gpt2')
    >>> backend = create_backend('openai', model_name='gpt-4')
    """
    backends = {
        'dummy': DummySolverBackend,
        'transformers': TransformersBackend,
        'openai': OpenAIBackend,
    }

    if backend_type not in backends:
        raise ValueError(f"Unknown backend type: {backend_type}. Available: {list(backends.keys())}")

    return backends[backend_type](**kwargs)
