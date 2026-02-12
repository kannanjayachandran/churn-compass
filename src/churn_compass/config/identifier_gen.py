"""
Identifier generation utilities for Churn Compass.

This module provides functions for generating different types of identifiers:
- Run IDs: For tracking pipeline executions and logging context
- Customer UUIDs: For persistent, unique customer identification
- batch UUIDs: UUIDs for tracking batch jobs
"""

import uuid
from datetime import datetime, timezone
from typing import Final

# Application-specific namespace for deterministic UUID generation
# This should be generated once and kept constant across the application
# Generate with: python -c "import uuid; print(uuid.uuid4())"
CUSTOMER_UUID_NAMESPACE: Final[uuid.UUID] = uuid.UUID(
    "59e0ac34-a656-49e5-b518-c971aad9c4d5"
)


def generate_run_id(prefix: str, include_microsecond: bool = True) -> str:
    """
    Generate a sortable, human-readable run ID for pipeline tracking.

    Creates time-based identifiers suitable for logging, monitoring, and
    debugging pipeline executions. Includes microseconds by default to
    prevent collisions when multiple runs start simultaneously.

    Args:
    prefix: Descriptive stage name (e.g., "ingestion", "training", "scoring")
    include_microseconds: Include microsecond precision for uniqueness.
                        Set to False for cleaner logs if collision risk is low.

    Returns:
        Run ID in format: {prefix}_{YYYYMMDD}_{HHMMSS}[_{MMMMMM}]

    Examples:
        >>> generate_run_id("ingestion")
        'ingestion_20251220_201026_847392'

        >>> generate_run_id("training", include_microseconds=False)
        'training_20251220_201026'

    Note:
        - All timestamps are in UTC
        - IDs are lexicographically sortable by time
        - Use for ephemeral tracking, not persistent entity identification
    """
    if include_microsecond:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return f"{prefix}_{timestamp}"


def generate_customer_uuid(customer_id: int) -> uuid.UUID:
    """
    Generate a deterministic UUID for a customer based on their bank CustomerId.

    Uses UUID v5 (SHA-1 hash of namespace + customer_id) to ensure:
    - **Idempotency**: Same CustomerId always produces the same UUID
    - **Uniqueness**: Different CustomerIds always produce different UUIDs
    - **Consistency**: Same customer across different batches has same UUID
    - **Standard compliance**: Valid UUID compatible with PostgreSQL UUID type

    This enables deduplication and consistent customer tracking even when
    the same customer appears in multiple data batches.

    :param customer_id: The bank's customer identifier from raw data (CustomerId column)
    :type customer_id: int
    :return: UUID (uuid.UUID object) that uniquely and deterministically represents
    this customer. Can be stored as UUID type in PostgreSQL.
    :rtype: UUID

    Examples:
        >>> generate_customer_uuid(12345)
        UUID('f8e7d6c5-b4a3-5291-8807-6f5e4d3c2b1a')

        >>> # Idempotent - same input always gives same output
        >>> uuid1 = generate_customer_uuid(12345)
        >>> uuid2 = generate_customer_uuid(12345)
        >>> uuid1 == uuid2
        True

        >>> # Different customers get different UUIDs
        >>> generate_customer_uuid(12345) == generate_customer_uuid(67890)
        False

    Usage in pipeline:
    ```python
        # During ingestion
        df['customer_uuid'] = df['CustomerId'].apply(generate_customer_uuid)

        # Store mapping in PostgreSQL
        INSERT INTO customers (uuid, bank_customer_id, ...)
        VALUES (customer_uuid, CustomerId, ...)
        ON CONFLICT (bank_customer_id) DO NOTHING;

        # Drop sensitive columns for training
        df_training = df.drop(columns=['CustomerId', 'Surname', 'RowNumber'])
        # df_training still has 'customer_uuid' for traceability
    ```

    Note:
        - The namespace UUID is application-specific and must remain constant
        - Changing CUSTOMER_UUID_NAMESPACE will change all generated UUIDs
        - Never use this for run tracking - use generate_run_id() instead
    """
    return uuid.uuid5(CUSTOMER_UUID_NAMESPACE, str(customer_id))


def generate_batch_job_id() -> uuid.UUID:
    """
    Generate a random UUID for tracking batch prediction jobs.

    Uses UUID v4 (random) for one-time job identifiers where determinism
    is not required. Suitable for tracking asynchronous batch processing jobs.

    Returns:
        UUID (uuid.UUID object) for batch job tracking

    Example:
        >>> job_id = generate_batch_job_id()
        >>> # Store in batch_jobs table
        >>> INSERT INTO batch_jobs (job_id, status, ...) VALUES (job_id, 'pending', ...)

    Note:
        - Use for ephemeral job tracking, not customer identification
        - For customer UUIDs, always use generate_customer_uuid() for consistency
    """
    return uuid.uuid4()
