"""Audit active PC-CNG execution queues and long-running watchers."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class QueueSpec:
    queue_id: str
    pid: Optional[int]
    gpu: Optional[int]
    dependency: str


DEFAULT_QUEUES: Sequence[QueueSpec] = (
    QueueSpec("type1_v2_coslr_warm5_20260712", 1512519, 4, "none"),
    QueueSpec("type1_v2_filtered_baseline_20260712", 2312365, 5, "none; first in GPU5 serial queue"),
    QueueSpec("type1_v2_valtop1_ckpt_smoke_20260712", 2312371, 5, "type1_v2_filtered_baseline_20260712"),
    QueueSpec("type1_v2_representation_scale_smoke_20260712", 2312379, 5, "type1_v2_valtop1_ckpt_smoke_20260712"),
    QueueSpec("type1_v2_pairwise_margin_smoke_20260712", 2312385, 5, "type1_v2_representation_scale_smoke_20260712"),
)

ARTIFACT_PATTERNS = (
    "**/summary.json",
    "**/ranking_metrics.json",
    "**/*_per_seed.csv",
    "**/metrics.json",
    "**/train.log",
)


def tail_lines(path: str, n: int) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    return lines[-n:]


def pid_alive(pid: Optional[int]) -> Optional[bool]:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def find_artifacts(result_dir: str, limit: int) -> List[str]:
    if not os.path.isdir(result_dir):
        return []
    paths: List[str] = []
    for pattern in ARTIFACT_PATTERNS:
        paths.extend(glob.glob(os.path.join(result_dir, pattern), recursive=True))
    return sorted(set(paths))[:limit]


def query_gpu_status() -> List[Dict[str, object]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: List[Dict[str, object]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, mem_used, mem_total, util = parts
        rows.append(
            {
                "index": index,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "utilization_pct": util,
            }
        )
    return rows


def summarize_train_queue(root: str, spec: QueueSpec, tail_n: int, artifact_limit: int) -> Dict[str, object]:
    log_path = os.path.join(root, "results", "logs", f"{spec.queue_id}.queue.log")
    result_dir = os.path.join(root, "results", spec.queue_id)
    artifacts = find_artifacts(result_dir, artifact_limit)
    log_tail = tail_lines(log_path, tail_n)
    return {
        "id": spec.queue_id,
        "pid": spec.pid,
        "pid_alive": pid_alive(spec.pid),
        "gpu": spec.gpu,
        "dependency": spec.dependency,
        "queue_log": log_path,
        "last_log_line": log_tail[-1] if log_tail else "",
        "log_tail": log_tail,
        "result_dir": result_dir,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def read_pid_file(path: str) -> Optional[int]:
    try:
        text = open(path, encoding="utf-8").read().strip()
        return int(text)
    except Exception:
        return None


def summarize_beam_watcher(root: str, tail_n: int) -> Dict[str, object]:
    pid_file = os.path.join(root, "results", "logs", "external_product_prediction_25k_chunked_beams.pid")
    queue_log = os.path.join(root, "results", "logs", "external_product_prediction_25k_chunked_beams.queue.log")
    status_json = os.path.join(
        root,
        "results",
        "external_product_prediction_benchmark_25k_20260713",
        "chemformer_beam_chunks",
        "chemformer_forward_beam_chunks_status.json",
    )
    status: Dict[str, object] = {}
    if os.path.exists(status_json):
        with open(status_json, encoding="utf-8") as handle:
            status = json.load(handle)
    pid = read_pid_file(pid_file)
    log_tail = tail_lines(queue_log, tail_n)
    return {
        "id": "external_product_prediction_25k_chunked_beams",
        "pid": pid,
        "pid_alive": pid_alive(pid),
        "pid_file": pid_file,
        "queue_log": queue_log,
        "last_log_line": log_tail[-1] if log_tail else "",
        "log_tail": log_tail,
        "status_json": status_json,
        "complete_chunks": status.get("complete_chunks"),
        "chunk_count": status.get("chunk_count"),
        "all_chunks_complete": status.get("all_chunks_complete"),
        "merged_beam_exists": status.get("merged_beam_exists"),
        "chunks": status.get("chunks", []),
    }


def write_markdown(path: str, payload: Dict[str, object]) -> None:
    lines = [
        "# Active Execution Status",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Generated at | `{payload['generated_at']}` |",
        f"| Root | `{payload['root']}` |",
        f"| GPU rows | `{len(payload['gpu_status'])}` |",
        "",
        "## Training Queues",
        "",
        "| Queue | PID | Alive | GPU | Artifacts | Last log line |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in payload["training_queues"]:  # type: ignore[index]
        lines.append(
            "| {id} | `{pid}` | `{pid_alive}` | `{gpu}` | `{artifact_count}` | `{last_log_line}` |".format(**row)
        )
    beam = payload["beam_watcher"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "## Beam Watcher",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| PID | `{beam.get('pid')}` |",
            f"| Alive | `{beam.get('pid_alive')}` |",
            f"| Complete chunks | `{beam.get('complete_chunks')}/{beam.get('chunk_count')}` |",
            f"| Merged beam exists | `{beam.get('merged_beam_exists')}` |",
            f"| Last log line | `{beam.get('last_log_line')}` |",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_audit(root: str, output_dir: str, tail_n: int, artifact_limit: int) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": root,
        "gpu_status": query_gpu_status(),
        "training_queues": [
            summarize_train_queue(root, spec, tail_n=tail_n, artifact_limit=artifact_limit)
            for spec in DEFAULT_QUEUES
        ],
        "beam_watcher": summarize_beam_watcher(root, tail_n=tail_n),
    }
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "active_execution_status.json")
    md_path = os.path.join(output_dir, "active_execution_status.md")
    payload["outputs"] = {"json": json_path, "summary_md": md_path}
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_markdown(md_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/cunyuliu/pc_cng_research")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-lines", type=int, default=8)
    parser.add_argument("--artifact-limit", type=int, default=20)
    args = parser.parse_args()
    payload = run_audit(
        root=args.root,
        output_dir=args.output_dir,
        tail_n=args.tail_lines,
        artifact_limit=args.artifact_limit,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
