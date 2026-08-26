"""Prepare labeled private Criteo train/test files for FIA experiments.

The official Display Advertising Challenge archive contains a labeled
``train.txt`` and an unlabeled ``test.txt``.  FIA evaluates accuracy, so its
test split must not use the official unlabeled file.  This script streams the
labeled training data and makes disjoint chronological train/test subsets.
It can read ``dac.tar.gz`` directly and does not extract the full 11 GB file.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_COLUMNS = 40  # label + 13 integer fields + 26 categorical fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create disjoint labeled Criteo train/test files from the official "
            "labeled train.txt or dac.tar.gz archive."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/criteo_raw/dac.tar.gz"),
        help="Official dac.tar.gz archive or its extracted labeled train.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/criteo"),
    )
    parser.add_argument("--train_rows", type=int, default=1_000_000)
    parser.add_argument("--test_rows", type=int, default=1_000_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output train.txt/test.txt files.",
    )
    return parser.parse_args()


@contextlib.contextmanager
def open_labeled_train(source: Path):
    if tarfile.is_tarfile(source):
        archive = tarfile.open(source, mode="r:*")
        try:
            candidates = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name == "train.txt"
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Expected exactly one train.txt in the archive, found {}.".format(
                        len(candidates)
                    )
                )
            binary = archive.extractfile(candidates[0])
            if binary is None:
                raise OSError("Could not read train.txt from the archive.")
            text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
            try:
                yield text, "archive:{}".format(candidates[0].name)
            finally:
                text.close()
        finally:
            archive.close()
    else:
        with source.open("r", encoding="utf-8", newline="") as handle:
            yield handle, str(source)


def validate_row(line: str, row_number: int) -> int:
    values = line.rstrip("\r\n").split("\t")
    if len(values) != EXPECTED_COLUMNS:
        raise ValueError(
            "Malformed source row {}: expected {} columns, got {}.".format(
                row_number, EXPECTED_COLUMNS, len(values)
            )
        )
    if values[0] not in {"0", "1"}:
        raise ValueError(
            "Malformed source row {}: label must be 0 or 1, got {!r}.".format(
                row_number, values[0]
            )
        )
    return int(values[0])


def main() -> int:
    args = parse_args()
    if args.train_rows <= 0 or args.test_rows <= 0:
        raise ValueError("--train_rows and --test_rows must both be positive.")
    if not args.source.is_file():
        raise FileNotFoundError(
            "Criteo source not found: {}. Download the official dac.tar.gz and "
            "place it there, or pass --source /path/to/train.txt.".format(args.source)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.txt"
    test_path = args.output_dir / "test.txt"
    metadata_path = args.output_dir / "private_split_metadata.json"
    if args.source.resolve() in {train_path.resolve(), test_path.resolve()}:
        raise ValueError(
            "--source must be kept outside --output_dir; otherwise publishing the "
            "split could overwrite the original labeled data."
        )
    existing = [path for path in (train_path, test_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists: {}. Use --overwrite to replace it.".format(
                ", ".join(str(path) for path in existing)
            )
        )

    train_tmp = train_path.with_suffix(".txt.tmp")
    test_tmp = test_path.with_suffix(".txt.tmp")
    required_rows = args.train_rows + args.test_rows
    label_counts = {"train": [0, 0], "test": [0, 0]}

    print("[1/3] Streaming labeled Criteo rows from: {}".format(args.source))
    written = 0
    source_member = None
    try:
        with open_labeled_train(args.source) as (source_handle, source_member):
            with train_tmp.open("w", encoding="utf-8", newline="") as train_out:
                with test_tmp.open("w", encoding="utf-8", newline="") as test_out:
                    for row_number, line in enumerate(source_handle, start=1):
                        label = validate_row(line, row_number)
                        if written < args.train_rows:
                            train_out.write(line if line.endswith("\n") else line + "\n")
                            label_counts["train"][label] += 1
                        elif written < required_rows:
                            test_out.write(line if line.endswith("\n") else line + "\n")
                            label_counts["test"][label] += 1
                        written += 1
                        if written >= required_rows:
                            break
        if written < required_rows:
            raise ValueError(
                "Source has only {} labeled rows; {} are required. Reduce "
                "--train_rows/--test_rows or use the full dataset.".format(
                    written, required_rows
                )
            )
        if min(label_counts["train"]) == 0 or min(label_counts["test"]) == 0:
            raise ValueError("Both generated splits must contain labels 0 and 1.")

        print("[2/3] Publishing disjoint private train/test splits")
        train_tmp.replace(train_path)
        test_tmp.replace(test_path)
    except Exception:
        for temporary in (train_tmp, test_tmp):
            if temporary.exists():
                temporary.unlink()
        raise

    metadata = {
        "artifact": "fia_criteo_private_split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_member": source_member,
        "split_strategy": "chronological_contiguous",
        "overlap_rows": 0,
        "train_rows": args.train_rows,
        "test_rows": args.test_rows,
        "train_label_counts": {
            "0": label_counts["train"][0],
            "1": label_counts["train"][1],
        },
        "test_label_counts": {
            "0": label_counts["test"][0],
            "1": label_counts["test"][1],
        },
        "official_unlabeled_test_used": False,
    }
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    with metadata_tmp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    metadata_tmp.replace(metadata_path)

    print("[3/3] Private Criteo data is ready")
    print("  train={} labels={}".format(train_path, metadata["train_label_counts"]))
    print("  test={} labels={}".format(test_path, metadata["test_label_counts"]))
    print("  metadata={}".format(metadata_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
