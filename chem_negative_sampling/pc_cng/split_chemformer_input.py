"""Split Chemformer input tables into resumable chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from typing import Dict, List, Sequence


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_markdown(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Chemformer Input Chunks",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Input | `{payload['input_path']}` |",
        f"| Total rows | `{payload['total_rows']}` |",
        f"| Chunk size | `{payload['chunk_size']}` |",
        f"| Chunk count | `{payload['chunk_count']}` |",
        "",
        "| Chunk | Rows | SHA256 | Path |",
        "|---:|---:|---|---|",
    ]
    for chunk in payload["chunks"]:  # type: ignore[index]
        row = chunk  # type: ignore[assignment]
        lines.append(f"| {row['chunk_index']} | {row['rows']} | `{row['sha256']}` | `{row['path']}` |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def split_chemformer_input(input_path: str, output_dir: str, chunk_size: int, prefix: str) -> Dict[str, object]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    os.makedirs(output_dir, exist_ok=True)
    chunks: List[Dict[str, object]] = []
    total_rows = 0

    with open(input_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Missing header in {input_path}")

        writer = None
        out_handle = None
        chunk_rows = 0
        chunk_index = -1
        current_path = ""
        try:
            for row in reader:
                if writer is None or chunk_rows >= chunk_size:
                    if out_handle is not None:
                        out_handle.close()
                        chunks[-1]["sha256"] = sha256_file(current_path)
                    chunk_index += 1
                    chunk_rows = 0
                    current_path = os.path.join(output_dir, f"{prefix}_chunk_{chunk_index:04d}.tsv")
                    out_handle = open(current_path, "w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter="\t")
                    writer.writeheader()
                    chunks.append({"chunk_index": chunk_index, "path": current_path, "rows": 0, "sha256": ""})
                writer.writerow(row)
                chunk_rows += 1
                total_rows += 1
                chunks[-1]["rows"] = chunk_rows
        finally:
            if out_handle is not None:
                out_handle.close()
                chunks[-1]["sha256"] = sha256_file(current_path)

    manifest_json = os.path.join(output_dir, f"{prefix}_chunks_manifest.json")
    manifest_md = os.path.join(output_dir, f"{prefix}_chunks_manifest.md")
    payload: Dict[str, object] = {
        "input_path": input_path,
        "input_sha256": sha256_file(input_path),
        "output_dir": output_dir,
        "chunk_size": chunk_size,
        "total_rows": total_rows,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "outputs": {
            "manifest_json": manifest_json,
            "manifest_md": manifest_md,
        },
    }
    with open(manifest_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_markdown(manifest_md, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--prefix", default="chemformer_forward_input")
    args = parser.parse_args()

    payload = split_chemformer_input(
        input_path=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        prefix=args.prefix,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
