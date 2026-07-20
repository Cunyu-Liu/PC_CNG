# pc_cng_tmp_sync [DEPRECATED]

> **DEPRECATED** (2026-07-19): This is a 2026-07-10 11:28 staging snapshot from early development. Kept for historical reference only. Do not use.

The canonical, newer versions live in:
- `run_learned_boundary_generation.py` → `/home/cunyuliu/pc_cng_research/chem_negative_sampling/pc_cng/run_learned_boundary_generation.py` (2026-07-10 11:52, adds `output_dim` + `decoder_head_scores` API)
- `scripts_run_edit_decoder_v3_pipeline.sh` → `/home/cunyuliu/pc_cng_research/chem_negative_sampling/scripts_run_edit_decoder_v3_pipeline.sh` (identical content)

**Diff summary** (tmp_sync vs canonical `run_learned_boundary_generation.py`):
- tmp_sync imports only `EditDecoderMLP`; canonical imports `EditDecoderMLP, decoder_head_scores`
- tmp_sync constructs `EditDecoderMLP(...)` without `output_dim`; canonical reads `output_dim` from checkpoint
- tmp_sync calls `model(x)` directly; canonical calls `decoder_head_scores(model, x, head="hard_negative")`

No code references this directory. Safe to delete after P1-13 review.
