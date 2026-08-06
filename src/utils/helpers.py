"""General utility helpers.

Common functions used across multiple modules.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime
import time


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division avoiding ZeroDivisionError."""
    try:
        return numerator / denominator if denominator != 0 else default
    except Exception:
        return default


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def retry_with_backoff(
    func,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Execute function with exponential backoff retry.

    Args:
        func: Callable to execute.
        max_attempts: Maximum retry attempts.
        base_delay: Initial delay in seconds.
        multiplier: Delay multiplier per attempt.
        exceptions: Tuple of exceptions to catch.

    Returns:
        Function result.

    Raises:
        Exception: Last exception after all retries exhausted.
    """
    delay = base_delay
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(delay)
                delay *= multiplier
    raise last_exc


def dict_to_markdown_table(data: Dict[str, Any]) -> str:
    """Convert dictionary to markdown table."""
    if not data:
        return ""
    lines = ["| Key | Value |", "|-----|-------|"]
    for k, v in data.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
