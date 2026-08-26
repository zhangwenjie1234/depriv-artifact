"""Public, deterministic preprocessing shared by all Criteo splits.

The transform deliberately has no ``fit`` step: private train/test data and
the independently released public auxiliary data use exactly the same public
constants and therefore cannot acquire incompatible category vocabularies.
"""

from __future__ import annotations

import hashlib
import math


CRITEO_PREPROCESSING_VERSION = "fia-criteo-hash-v1"
CRITEO_HASH_SALT = "fia-criteo-v1"
CRITEO_CATEGORY_BUCKET_SIZE = 10_000
CRITEO_INTEGER_COUNT = 13
CRITEO_CATEGORY_COUNT = 26
CRITEO_FEATURE_COUNT = CRITEO_INTEGER_COUNT + CRITEO_CATEGORY_COUNT

CRITEO_INTEGER_INPUT_COLUMNS = [
    f"integer_feature_{index}" for index in range(1, CRITEO_INTEGER_COUNT + 1)
]
CRITEO_CATEGORY_INPUT_COLUMNS = [
    f"categorical_feature_{index}" for index in range(1, CRITEO_CATEGORY_COUNT + 1)
]
CRITEO_AUX_INPUT_COLUMNS = (
    ["label"] + CRITEO_INTEGER_INPUT_COLUMNS + CRITEO_CATEGORY_INPUT_COLUMNS
)
CRITEO_FEATURE_NAMES = (
    [f"I{index}" for index in range(1, CRITEO_INTEGER_COUNT + 1)]
    + [f"C{index}" for index in range(1, CRITEO_CATEGORY_COUNT + 1)]
)
CRITEO_RAW_COLUMNS = ["label"] + CRITEO_FEATURE_NAMES

# Numeric fields only ever index embedding row zero; categorical field zero is
# reserved for missing values and 1..bucket_size are non-missing hash buckets.
CRITEO_FEATURE_SIZES = (
    [1] * CRITEO_INTEGER_COUNT
    + [CRITEO_CATEGORY_BUCKET_SIZE + 1] * CRITEO_CATEGORY_COUNT
)
CRITEO_NUMERIC_MASK = (
    [True] * CRITEO_INTEGER_COUNT + [False] * CRITEO_CATEGORY_COUNT
)


def normalize_integer(value: object) -> float:
    """Apply the public numeric transform without fitting private statistics."""
    if value is None or str(value).strip() == "":
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return math.log1p(max(number, 0.0))


def stable_category_bucket(
    value: object,
    field_index: int,
    bucket_size: int = CRITEO_CATEGORY_BUCKET_SIZE,
    hash_salt: str = CRITEO_HASH_SALT,
) -> int:
    """Map a category to a stable public bucket; zero denotes missing."""
    if value is None or str(value).strip() == "":
        return 0
    if not 1 <= int(field_index) <= CRITEO_CATEGORY_COUNT:
        raise ValueError(f"Criteo category field index is invalid: {field_index}")
    if int(bucket_size) <= 1:
        raise ValueError("Criteo category bucket size must be greater than 1.")
    payload = f"{hash_salt}|C{field_index}|{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (int.from_bytes(digest, byteorder="big", signed=False) % bucket_size) + 1
