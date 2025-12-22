"""
Churn Compass - UUID
"""

from datetime import datetime, timezone


def generate_run_id(prefix: str) -> str:
    """
    Generate a sortable, human-readable run ID.

    Example:
        train_20251220_201026
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
