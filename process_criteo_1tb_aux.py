"""Convert the sampled Criteo 1TB auxiliary data into deterministic features.

Integer fields are mapped to log1p(max(x, 0)).  Categorical fields use a
stable BLAKE2b hash into a public fixed-size bucket space, with bucket zero
reserved for missing values.  The output Parquet and NPZ files contain the
same 39 features in the original Criteo order.

The private Kaggle Criteo loader imports the same transformation implementation,
so train, test, and auxiliary fields are encoded identically without fitting
statistics from private data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from utils.criteo_preprocessing import (
    CRITEO_AUX_INPUT_COLUMNS as EXPECTED_INPUT_COLUMNS,
    CRITEO_CATEGORY_BUCKET_SIZE,
    CRITEO_CATEGORY_INPUT_COLUMNS as CATEGORICAL_INPUT_COLUMNS,
    CRITEO_FEATURE_NAMES as OUTPUT_FEATURE_COLUMNS,
    CRITEO_HASH_SALT,
    CRITEO_INTEGER_INPUT_COLUMNS as INTEGER_INPUT_COLUMNS,
    CRITEO_PREPROCESSING_VERSION,
    normalize_integer,
    stable_category_bucket,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize integer fields and deterministically hash categorical "
            "fields in the sampled Criteo 1TB public auxiliary data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/criteo_1tb_public/public_aux_raw.parquet"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/criteo_1tb_public"),
    )
    parser.add_argument("--parquet_name", default="public_aux_processed.parquet")
    parser.add_argument("--npz_name", default="public_aux_processed.npz")
    parser.add_argument(
        "--metadata_name", default="public_aux_processed_metadata.json"
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=CRITEO_CATEGORY_BUCKET_SIZE,
        help="Number of non-missing buckets per categorical field.",
    )
    parser.add_argument("--hash_salt", default=CRITEO_HASH_SALT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed artifacts after successful conversion.",
    )
    return parser.parse_args()


def require_dependencies():
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: pip install numpy pyarrow"
        ) from exc
    return np, pa, pq


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.bucket_size <= 1:
        raise ValueError("--bucket_size must be greater than 1.")
    if not args.hash_salt:
        raise ValueError("--hash_salt must not be empty.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Raw auxiliary file not found: {args.input}")

    np, pa, pq = require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / args.parquet_name
    npz_path = args.output_dir / args.npz_name
    metadata_path = args.output_dir / args.metadata_name
    if not args.overwrite:
        existing = [
            path
            for path in (parquet_path, npz_path, metadata_path)
            if path.exists()
        ]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output already exists: {joined}. Use --overwrite to replace it."
            )

    print(f"[1/4] Reading raw public auxiliary data: {args.input}")
    raw_table = pq.read_table(args.input, columns=EXPECTED_INPUT_COLUMNS)
    missing_columns = [
        name for name in EXPECTED_INPUT_COLUMNS if name not in raw_table.column_names
    ]
    if missing_columns:
        raise ValueError(f"Input file is missing columns: {missing_columns}")
    rows = raw_table.to_pylist()
    if not rows:
        raise ValueError("Input auxiliary file is empty.")

    print(
        "[2/4] Applying public deterministic preprocessing: "
        f"BLAKE2b buckets={args.bucket_size}, salt={args.hash_salt!r}"
    )
    feature_matrix = np.empty((len(rows), 39), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    integer_missing = [0] * 13
    categorical_missing = [0] * 26
    categorical_unique_raw = [set() for _ in range(26)]
    categorical_unique_bucket = [set() for _ in range(26)]

    for row_index, row in enumerate(rows):
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Unexpected label {label!r} at row {row_index}.")
        labels[row_index] = label

        for feature_index, column in enumerate(INTEGER_INPUT_COLUMNS):
            raw_value = row[column]
            if raw_value is None:
                integer_missing[feature_index] += 1
            feature_matrix[row_index, feature_index] = normalize_integer(raw_value)

        for feature_index, column in enumerate(CATEGORICAL_INPUT_COLUMNS):
            raw_value = row[column]
            if raw_value is None or str(raw_value) == "":
                categorical_missing[feature_index] += 1
            else:
                categorical_unique_raw[feature_index].add(str(raw_value))
            bucket = stable_category_bucket(
                value=raw_value,
                field_index=feature_index + 1,
                bucket_size=args.bucket_size,
                hash_salt=args.hash_salt,
            )
            feature_matrix[row_index, 13 + feature_index] = float(bucket)
            categorical_unique_bucket[feature_index].add(bucket)

    processed_columns = {"label": pa.array(labels, type=pa.int64())}
    for feature_index in range(13):
        processed_columns[f"I{feature_index + 1}"] = pa.array(
            feature_matrix[:, feature_index], type=pa.float32()
        )
    for feature_index in range(26):
        processed_columns[f"C{feature_index + 1}"] = pa.array(
            feature_matrix[:, 13 + feature_index].astype(np.int64), type=pa.int64()
        )
    processed_table = pa.table(processed_columns)

    print(f"[3/4] Writing processed artifacts under: {args.output_dir}")
    temporary_parquet = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    pq.write_table(processed_table, temporary_parquet, compression="snappy")
    temporary_parquet.replace(parquet_path)

    temporary_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=feature_matrix,
            labels=labels,
            feature_names=np.asarray(OUTPUT_FEATURE_COLUMNS),
            preprocessing_version=np.asarray(CRITEO_PREPROCESSING_VERSION),
            hash_salt=np.asarray(args.hash_salt),
            bucket_size=np.asarray(args.bucket_size),
        )
    temporary_npz.replace(npz_path)

    label_counts = {
        "0": int(np.count_nonzero(labels == 0)),
        "1": int(np.count_nonzero(labels == 1)),
    }
    categorical_stats = []
    for feature_index in range(26):
        raw_unique = len(categorical_unique_raw[feature_index])
        bucket_unique_nonmissing = len(
            {value for value in categorical_unique_bucket[feature_index] if value != 0}
        )
        categorical_stats.append(
            {
                "field": f"C{feature_index + 1}",
                "missing": categorical_missing[feature_index],
                "unique_raw_nonmissing": raw_unique,
                "unique_buckets_nonmissing": bucket_unique_nonmissing,
                "observed_collisions": max(raw_unique - bucket_unique_nonmissing, 0),
            }
        )

    metadata = {
        "artifact": "criteo_1tb_public_aux_processed",
        "preprocessing_version": CRITEO_PREPROCESSING_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input),
        "input_sha256": sha256_file(args.input),
        "rows": len(rows),
        "features": 39,
        "label_counts": label_counts,
        "feature_order": OUTPUT_FEATURE_COLUMNS,
        "integer_transform": {
            "formula": "log1p(max(value, 0)); missing/nonfinite -> 0",
            "missing_by_field": {
                f"I{index + 1}": count
                for index, count in enumerate(integer_missing)
            },
        },
        "categorical_transform": {
            "algorithm": "blake2b-64",
            "payload": "<hash_salt>|C<field_index>|<raw_value>",
            "hash_salt": args.hash_salt,
            "bucket_size": args.bucket_size,
            "nonmissing_bucket_range": [1, args.bucket_size],
            "missing_bucket": 0,
            "fields": categorical_stats,
        },
        "privacy_contract": {
            "fits_private_statistics": False,
            "requires_same_transform_for_private_train_and_test": True,
        },
        "parquet_file": parquet_path.name,
        "parquet_sha256": sha256_file(parquet_path),
        "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
    }
    atomic_write_json(metadata_path, metadata)

    print("[4/4] Processed public auxiliary artifacts are ready")
    print(f"  features_shape={feature_matrix.shape}")
    print(f"  labels={label_counts}")
    print(f"  parquet={parquet_path}")
    print(f"  npz={npz_path}")
    print(f"  metadata={metadata_path}")
    print(
        "  Shared transform: private train/test use the same public constants "
        "from utils/criteo_preprocessing.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
