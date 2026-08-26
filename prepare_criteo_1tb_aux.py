"""Download and reproducibly sample the public Criteo 1TB auxiliary data.

Only one pinned day/shard is downloaded.  The script scans that shard with a
fixed-seed, class-stratified reservoir sampler and writes a small raw Parquet
artifact plus provenance metadata.  It intentionally preserves the original
feature values; model encoding is handled by ``process_criteo_1tb_aux.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_ID = "criteo/CriteoClickLogs"
DEFAULT_SOURCE_DAY = "2015-02-15"
DEFAULT_SOURCE_FILE = (
    "data/day=2015-02-15/"
    "part-00015-99c339d5-fbac-4110-9dcf-75453a61a5c1.c000.snappy.parquet"
)
EXPECTED_COLUMNS = (
    ["label"]
    + [f"integer_feature_{index}" for index in range(1, 14)]
    + [f"categorical_feature_{index}" for index in range(1, 27)]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download one official Criteo 1TB Parquet shard and create a "
            "fixed, class-stratified public auxiliary subset."
        )
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help=(
            "Optional Hugging Face-compatible endpoint, for example "
            "https://hf-mirror.com. Defaults to the HF_ENDPOINT environment variable."
        ),
    )
    parser.add_argument(
        "--local_source",
        type=Path,
        default=None,
        help=(
            "Use an already-downloaded Parquet shard and perform sampling fully "
            "offline. When set, no Hugging Face API request is made."
        ),
    )
    parser.add_argument("--source_day", default=DEFAULT_SOURCE_DAY)
    parser.add_argument("--source_file", default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--sample_size", type=int, default=10_000)
    parser.add_argument(
        "--positive_fraction",
        type=float,
        default=0.5,
        help="Requested fraction of public click-label 1 examples.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=16_384)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/criteo_1tb_public"),
    )
    parser.add_argument("--output_name", default="public_aux_raw.parquet")
    parser.add_argument("--metadata_name", default="public_aux_metadata.json")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output artifacts after a successful new build.",
    )
    return parser.parse_args()


def require_dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: "
            "pip install pyarrow huggingface_hub"
        ) from exc
    return pa, pq, HfApi, hf_hub_download


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_args(args: argparse.Namespace) -> tuple[int, int]:
    if args.sample_size <= 1:
        raise ValueError("--sample_size must be greater than 1.")
    if not 0.0 < args.positive_fraction < 1.0:
        raise ValueError("--positive_fraction must be strictly between 0 and 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    positive_target = int(round(args.sample_size * args.positive_fraction))
    negative_target = args.sample_size - positive_target
    if positive_target <= 0 or negative_target <= 0:
        raise ValueError("Both label reservoirs must contain at least one example.")
    return negative_target, positive_target


def reservoir_update(
    reservoir: list[dict],
    row: dict,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < limit:
        reservoir.append(row)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = row


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    negative_target, positive_target = validate_args(args)
    pa, pq, HfApi, hf_hub_download = require_dependencies()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    metadata_path = args.output_dir / args.metadata_name
    if not args.overwrite:
        existing = [path for path in (output_path, metadata_path) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output already exists: {joined}. Use --overwrite to replace it."
            )

    if args.local_source is not None:
        downloaded = args.local_source.expanduser().resolve()
        if not downloaded.is_file():
            raise FileNotFoundError(f"Local source shard not found: {downloaded}")
        resolved_revision = "local-source-see-sha256"
        print("[1/5] Offline mode: skipping Hugging Face revision lookup")
        print(f"[2/5] Using local source shard: {downloaded}")
    else:
        endpoint_text = args.endpoint or "https://huggingface.co"
        print(
            f"[1/5] Resolving dataset revision: {args.repo_id}@{args.revision} "
            f"via {endpoint_text}"
        )
        try:
            api = HfApi(endpoint=args.endpoint) if args.endpoint else HfApi()
            dataset_info = api.dataset_info(args.repo_id, revision=args.revision)
            resolved_revision = dataset_info.sha
        except Exception as exc:
            print(
                "  WARNING: revision lookup failed; attempting the file download "
                f"with revision={args.revision!r}: {exc}",
                file=sys.stderr,
            )
            resolved_revision = args.revision

        print(f"[2/5] Downloading fixed source shard: {args.source_file}")
        download_kwargs = {
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "filename": args.source_file,
            "revision": resolved_revision,
        }
        if args.endpoint:
            download_kwargs["endpoint"] = args.endpoint
        try:
            downloaded = Path(hf_hub_download(**download_kwargs))
        except Exception as exc:
            raise RuntimeError(
                "Unable to download the public Criteo shard. The server cannot "
                "reach the configured Hugging Face endpoint. Try "
                "--endpoint https://hf-mirror.com, or download the single shard "
                "elsewhere, upload it, and rerun with --local_source PATH."
            ) from exc
    source_sha256 = sha256_file(downloaded)
    source_size = downloaded.stat().st_size

    parquet_file = pq.ParquetFile(downloaded)
    actual_columns = parquet_file.schema_arrow.names
    missing_columns = [name for name in EXPECTED_COLUMNS if name not in actual_columns]
    if missing_columns:
        raise ValueError(f"Source shard is missing columns: {missing_columns}")

    print(
        "[3/5] Scanning shard with stratified reservoir sampling: "
        f"label0={negative_target}, label1={positive_target}, seed={args.seed}"
    )
    reservoirs: dict[int, list[dict]] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    rng = random.Random(args.seed)
    scanned_rows = 0

    for batch in parquet_file.iter_batches(
        batch_size=args.batch_size,
        columns=EXPECTED_COLUMNS,
        use_threads=False,
    ):
        for row in batch.to_pylist():
            label = int(row["label"])
            if label not in reservoirs:
                raise ValueError(f"Unexpected public label {label!r}; expected 0 or 1.")
            seen[label] += 1
            limit = negative_target if label == 0 else positive_target
            reservoir_update(reservoirs[label], row, seen[label], limit, rng)
            scanned_rows += 1
        if scanned_rows and scanned_rows % 1_000_000 < len(batch):
            print(
                f"  scanned={scanned_rows:,} "
                f"seen0={seen[0]:,} seen1={seen[1]:,}",
                flush=True,
            )

    if len(reservoirs[0]) != negative_target or len(reservoirs[1]) != positive_target:
        raise RuntimeError(
            "The selected shard does not contain enough examples: "
            f"available label0={seen[0]}, label1={seen[1]}."
        )

    selected_rows = reservoirs[0] + reservoirs[1]
    rng.shuffle(selected_rows)
    output_table = pa.Table.from_pylist(selected_rows, schema=parquet_file.schema_arrow)
    if output_table.num_rows != args.sample_size:
        raise AssertionError("Internal error: sampled row count is incorrect.")

    print(f"[4/5] Writing raw public subset: {output_path}")
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(output_table, temporary_output, compression="snappy")
    temporary_output.replace(output_path)
    output_sha256 = sha256_file(output_path)

    metadata = {
        "artifact": "criteo_1tb_public_aux_raw",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-NC-SA-4.0",
        "source_repo": args.repo_id,
        "source_endpoint": args.endpoint,
        "local_source_mode": args.local_source is not None,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "source_day": args.source_day,
        "source_file": args.source_file,
        "source_file_bytes": source_size,
        "source_sha256": source_sha256,
        "sampling": {
            "algorithm": "per_label_reservoir_sampling",
            "seed": args.seed,
            "scanned_rows": scanned_rows,
            "seen_label_0": seen[0],
            "seen_label_1": seen[1],
            "sample_size": args.sample_size,
            "sampled_label_0": len(reservoirs[0]),
            "sampled_label_1": len(reservoirs[1]),
            "positive_fraction": args.positive_fraction,
        },
        "schema": {
            "label": "label",
            "integer_features": [f"integer_feature_{i}" for i in range(1, 14)],
            "categorical_features": [
                f"categorical_feature_{i}" for i in range(1, 27)
            ],
            "missing_values_preserved": True,
        },
        "output_file": output_path.name,
        "output_sha256": output_sha256,
    }
    atomic_write_json(metadata_path, metadata)

    print("[5/5] Public auxiliary raw artifact is ready")
    print(f"  rows={output_table.num_rows:,} columns={output_table.num_columns}")
    print(f"  labels={{0: {len(reservoirs[0])}, 1: {len(reservoirs[1])}}}")
    print(f"  source_revision={resolved_revision}")
    print(f"  output_sha256={output_sha256}")
    print(f"  metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; no completed output metadata was produced.", file=sys.stderr)
        raise SystemExit(130)
