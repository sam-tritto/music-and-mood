"""
API Cost Estimator for Google Gemini Models.
=============================================
Provides local, 0-latency token and cost estimations for Gemini API requests
before they are sent, preventing unexpected charges.
"""

import logging

logger = logging.getLogger(__name__)

# Pricing rates as of June 2026 (USD per 1,000,000 tokens)
PRICING_RATES = {
    "gemini-embedding-001": {
        "input": 0.15,
        "output": 0.0,
    },
    "text-embedding-004": {
        "input": 0.02,
        "output": 0.0,
    },
    "gemini-3.5-flash": {
        "input": 1.50,
        "output": 9.00,
    },
}

DEFAULT_EMBEDDING_RATE = 0.15
DEFAULT_NARRATIVE_INPUT_RATE = 1.50
DEFAULT_NARRATIVE_OUTPUT_RATE = 9.00


def estimate_tokens_locally(text: str) -> int:
    """
    Estimate the number of tokens in a string using a local word-based heuristic
    (1 word ≈ 1.33 tokens). This is fast and requires no API calls.
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.33)


def estimate_embedding_cost(texts: list[str], model: str) -> dict:
    """
    Calculate the estimated token count and cost (USD) for embedding a list of texts.

    Parameters
    ----------
    texts : list of strings to embed
    model : the embedding model name

    Returns
    -------
    dict with 'tokens', 'cost_usd', and 'model'
    """
    total_tokens = sum(estimate_tokens_locally(t) for t in texts)
    
    rate = PRICING_RATES.get(model, {}).get("input", DEFAULT_EMBEDDING_RATE)
    cost = (total_tokens / 1_000_000) * rate
    
    return {
        "model": model,
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6),
    }


def estimate_narrative_cost(prompt: str, system_instruction: str, max_output_tokens: int, model: str) -> dict:
    """
    Calculate the estimated token count and cost (USD) for a narrative generation request.

    Parameters
    ----------
    prompt : the user prompt string
    system_instruction : the system instruction string
    max_output_tokens : the maximum output tokens requested
    model : the model name

    Returns
    -------
    dict with input/output token estimates and cost details
    """
    input_text = f"{system_instruction}\n{prompt}"
    input_tokens = estimate_tokens_locally(input_text)
    
    # We estimate output tokens based on max_output_tokens, but usually it generates less.
    # We will assume it outputs around 60% of max_output_tokens or a baseline of 250 tokens.
    expected_output_tokens = min(250, max_output_tokens)
    
    rates = PRICING_RATES.get(model, {
        "input": DEFAULT_NARRATIVE_INPUT_RATE,
        "output": DEFAULT_NARRATIVE_OUTPUT_RATE,
    })
    
    input_cost = (input_tokens / 1_000_000) * rates.get("input", DEFAULT_NARRATIVE_INPUT_RATE)
    output_cost = (expected_output_tokens / 1_000_000) * rates.get("output", DEFAULT_NARRATIVE_OUTPUT_RATE)
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": expected_output_tokens,
        "estimated_cost_usd": round(total_cost, 6),
    }
