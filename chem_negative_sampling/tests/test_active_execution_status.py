from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ActiveExecutionStatusTest(unittest.TestCase):
    def test_audit_collects_queue_and_beam_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            out_dir = Path(tmp) / "out"
            logs = root / "results" / "logs"
            logs.mkdir(parents=True)
            result_dir = root / "results" / "type1_v2_coslr_warm5_20260712" / "seed0"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            (logs / "type1_v2_coslr_warm5_20260712.queue.log").write_text(
                "old\nlatest gpu4 mem=1MiB util=0%\n",
                encoding="utf-8",
            )
            (logs / "external_product_prediction_25k_chunked_beams.pid").write_text("99999999\n", encoding="utf-8")
            (logs / "external_product_prediction_25k_chunked_beams.queue.log").write_text(
                "[wait] chunk 0\n[gpu] busy\n",
                encoding="utf-8",
            )
            status_dir = (
                root
                / "results"
                / "external_product_prediction_benchmark_25k_20260713"
                / "chemformer_beam_chunks"
            )
            status_dir.mkdir(parents=True)
            (status_dir / "chemformer_forward_beam_chunks_status.json").write_text(
                json.dumps(
                    {
                        "complete_chunks": 1,
                        "chunk_count": 5,
                        "all_chunks_complete": False,
                        "merged_beam_exists": False,
                        "chunks": [{"chunk_index": 0, "beam_valid": True}],
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.audit_active_execution_status",
                    "--root",
                    str(root),
                    "--output-dir",
                    str(out_dir),
                    "--tail-lines",
                    "2",
                ],
                check=True,
            )

            payload = json.loads((out_dir / "active_execution_status.json").read_text())
            self.assertEqual(payload["beam_watcher"]["complete_chunks"], 1)
            self.assertEqual(payload["beam_watcher"]["chunk_count"], 5)
            queues = {row["id"]: row for row in payload["training_queues"]}
            self.assertEqual(queues["type1_v2_coslr_warm5_20260712"]["artifact_count"], 1)
            self.assertIn("latest", queues["type1_v2_coslr_warm5_20260712"]["last_log_line"])
            self.assertTrue((out_dir / "active_execution_status.md").exists())


if __name__ == "__main__":
    unittest.main()
