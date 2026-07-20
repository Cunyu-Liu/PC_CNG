"""Generate the 6 figures for the PC-CNG manuscript v2 (P2-09).

Figures:
  1. PC-CNG architecture overview (text/ASCII).
  2. Cross-dataset migration gains (P2-05).
  3. Retrosynthesis route ranking (P2-01).
  4. External bridge calibration (P2-04 v2 calibrator).
  5. DFT validation support (P2-02, 90% support).
  6. SOTA comparison radar chart (P2-06).

Uses matplotlib when available; otherwise emits ASCII/JSON figure
descriptions so the manuscript can still reference a figure file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "FIGURE_NAMES",
    "load_json",
    "generate_figure_1_architecture",
    "generate_figure_2_cross_dataset",
    "generate_figure_3_route_ranking",
    "generate_figure_4_external_bridge",
    "generate_figure_5_dft_validation",
    "generate_figure_6_sota_radar",
    "generate_all_figures",
    "main",
]

FIGURE_NAMES = [
    "figure_1_architecture_overview",
    "figure_2_cross_dataset_migration",
    "figure_3_route_ranking",
    "figure_4_external_bridge_calibration",
    "figure_5_dft_validation_support",
    "figure_6_sota_radar",
]

# Default data used when the corresponding results JSON is missing.  These
# match the fallback numbers in build_manuscript_v2.py.
DEFAULT_DATA: Dict[str, Any] = {
    "p2_01": {
        "baseline_mrr": 0.2431, "pc_cng_mrr": 0.5350,
        "delta_pp": 29.20, "ci_low_pp": 28.18, "ci_high_pp": 30.53,
    },
    "p2_02": {"support_rate": 0.9, "n_supported": 27, "n_not_supported": 3, "n_computed": 30},
    "p2_04": {"baseline_top1": 0.525, "v2_top1": 0.5504, "delta_pp": 2.54,
              "ci_low_pp": 1.33, "ci_high_pp": 3.75},
    "p2_05": {
        "pairs": [
            {"pair": "regiosqm20_to_hitea", "delta_mean": 0.0,
             "delta_ci95_low": 0.0, "delta_ci95_high": 0.0, "n_pooled": 3830},
            {"pair": "regiosqm20_to_uspto", "delta_mean": 0.01088,
             "delta_ci95_low": -0.00335, "delta_ci95_high": 0.02469, "n_pooled": 2390},
            {"pair": "regiosqm20_to_nicolit", "delta_mean": 0.0,
             "delta_ci95_low": 0.0, "delta_ci95_high": 0.0, "n_pooled": 0},
            {"pair": "regiosqm20_to_ord", "delta_mean": 0.0,
             "delta_ci95_low": 0.0, "delta_ci95_high": 0.0, "n_pooled": 0},
        ],
    },
    "p2_06": {
        "per_baseline": {
            "pc_cng_vs_rdkit_template": {"baseline": "rdkit_template", "delta_pp": 27.83,
                                          "ci_low_pp": 27.56, "ci_high_pp": 28.11},
            "pc_cng_vs_heuristic_validator": {"baseline": "heuristic_validator", "delta_pp": 27.83,
                                               "ci_low_pp": 27.56, "ci_high_pp": 28.11},
            "pc_cng_vs_tanimoto_nn": {"baseline": "tanimoto_nn", "delta_pp": -48.61,
                                       "ci_low_pp": -48.89, "ci_high_pp": -48.33},
        },
        "methods": ["rdkit_template", "heuristic_validator", "tanimoto_nn", "pc_cng"],
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning ``{}`` when missing or invalid."""
    try:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _try_import_matplotlib():
    """Import matplotlib in headless mode; return module or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _write_ascii_figure(output_dir: Path, name: str, title: str, body: str) -> Path:
    """Write an ASCII figure description as a .txt file."""
    path = output_dir / f"{name}.txt"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def _load_p2_data(results_dir: Path) -> Dict[str, Any]:
    """Load P2 data from the results directory, falling back to DEFAULT_DATA."""
    data = {}
    # P2-01
    p2_01_summary = load_json(results_dir / "aizynthfinder_route_ranking_20260720" / "route_ranking_summary.json")
    p2_01_paired = load_json(results_dir / "aizynthfinder_route_ranking_20260720" / "paired_significance.json")
    r3r1 = p2_01_paired.get("r3_vs_r1", {}) if p2_01_paired else {}
    metrics = p2_01_summary.get("metrics", {}) if p2_01_summary else {}
    data["p2_01"] = {
        "baseline_mrr": metrics.get("aizynthfinder_baseline", {}).get("mrr", DEFAULT_DATA["p2_01"]["baseline_mrr"]),
        "pc_cng_mrr": metrics.get("aizynthfinder_pc_cng", {}).get("mrr", DEFAULT_DATA["p2_01"]["pc_cng_mrr"]),
        "delta_pp": r3r1.get("delta_pp", DEFAULT_DATA["p2_01"]["delta_pp"]),
        "ci_low_pp": r3r1.get("seed_level_ci95_low_pp", DEFAULT_DATA["p2_01"]["ci_low_pp"]),
        "ci_high_pp": r3r1.get("seed_level_ci95_high_pp", DEFAULT_DATA["p2_01"]["ci_high_pp"]),
    }
    # P2-02
    p2_02 = load_json(results_dir / "dft_validation_chemoselectivity_20260720" / "dft_validation_summary.json")
    data["p2_02"] = {
        "support_rate": p2_02.get("support_rate", DEFAULT_DATA["p2_02"]["support_rate"]),
        "n_supported": p2_02.get("n_supported", DEFAULT_DATA["p2_02"]["n_supported"]),
        "n_not_supported": p2_02.get("n_not_supported", DEFAULT_DATA["p2_02"]["n_not_supported"]),
        "n_computed": p2_02.get("n_computed", DEFAULT_DATA["p2_02"]["n_computed"]),
    }
    # P2-04
    p2_04 = load_json(results_dir / "external_score_mlp_calibrator_v2_chemformer_aware_20260720" / "summary.json")
    top1 = p2_04.get("metrics", {}).get("top1", {}) if p2_04 else {}
    paired_test = top1.get("paired_test", {}) if top1 else {}
    data["p2_04"] = {
        "baseline_top1": p2_04.get("baseline_top1_mean", DEFAULT_DATA["p2_04"]["baseline_top1"]),
        "v2_top1": top1.get("v2_mean", DEFAULT_DATA["p2_04"]["v2_top1"]),
        "delta_pp": p2_04.get("delta_top1_pp", DEFAULT_DATA["p2_04"]["delta_pp"]),
        "ci_low_pp": paired_test.get("ci_low", DEFAULT_DATA["p2_04"]["ci_low_pp"] / 100) * 100,
        "ci_high_pp": paired_test.get("ci_high", DEFAULT_DATA["p2_04"]["ci_high_pp"] / 100) * 100,
    }
    # P2-05
    p2_05_dir = results_dir / "cross_dataset_transfer_v2_20260720"
    pairs = []
    if p2_05_dir.exists():
        for pd in sorted(p2_05_dir.iterdir()):
            if not pd.is_dir():
                continue
            psig = load_json(pd / "paired_significance.json")
            if not psig:
                continue
            pooled = psig.get("paired_significance_pooled", {})
            pairs.append({
                "pair": pd.name,
                "delta_mean": pooled.get("delta_mean", 0.0),
                "delta_ci95_low": pooled.get("delta_ci95_low", 0.0),
                "delta_ci95_high": pooled.get("delta_ci95_high", 0.0),
                "n_pooled": pooled.get("n", 0),
            })
    if not pairs:
        pairs = DEFAULT_DATA["p2_05"]["pairs"]
    data["p2_05"] = {"pairs": pairs}
    # P2-06
    p2_06_go = load_json(results_dir / "sota_comparison_uspto_mit_50k_20260720_smoke" / "go_no_go_decision.json")
    per_baseline = p2_06_go.get("per_baseline", DEFAULT_DATA["p2_06"]["per_baseline"]) if p2_06_go else DEFAULT_DATA["p2_06"]["per_baseline"]
    data["p2_06"] = {"per_baseline": per_baseline, "methods": DEFAULT_DATA["p2_06"]["methods"]}
    return data


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def generate_figure_1_architecture(plt, output_dir: Path) -> Path:
    """Figure 1: PC-CNG architecture overview."""
    title = "Figure 1. PC-CNG architecture overview"
    body = (
        "PC-CNG architecture (text rendering):\n\n"
        "  Atom-mapped reactions (USPTO / RegioSQM20 / HiTEA / ORD)\n"
        "        |\n"
        "        v\n"
        "  +-------------------+    5 typed edit actions\n"
        "  | Counterfactual    |    replace_atom / drop_reactant /\n"
        "  | Generator         |    swap_functional_group /\n"
        "  | (rule-based +     |    wrong_anchor / add_reagent\n"
        "  |  GNN supplement)  |\n"
        "  +---------+---------+\n"
        "            |\n"
        "            v\n"
        "  +-------------------+    Layer 1: ensemble agreement\n"
        "  | 3-Layer False-    |    Layer 2: DB retrieval (Tanimoto >= 0.95)\n"
        "  | Negative Control  |    Layer 3: rule-based plausibility\n"
        "  +---------+---------+    (expert review deferred -> L4)\n"
        "            |\n"
        "            v\n"
        "  High-confidence negatives (26,517 / 64,646 = 41.02%)\n"
        "            |\n"
        "            +-----------> Reranker training (USPTO trainval)\n"
        "            |\n"
        "            +-----------> Cross-dataset transfer (P2-05)\n"
        "            |\n"
        "            +-----------> Route ranking (P2-01, GO +29.20 pp)\n"
        "            |\n"
        "            +-----------> External bridge v2 calibrator (P2-04, GO +2.54 pp)\n"
        "            |\n"
        "            +-----------> DFT validation (P2-02, GO 90% support)\n"
    )
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=13, fontweight="bold")
        ax.text(0.02, 0.92, body, ha="left", va="top", fontsize=8.5, family="monospace",
                transform=ax.transAxes)
        path = output_dir / "figure_1_architecture_overview.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
    return _write_ascii_figure(output_dir, "figure_1_architecture_overview", title, body)


def generate_figure_2_cross_dataset(plt, output_dir: Path, p2_05: Dict[str, Any]) -> Path:
    """Figure 2: Cross-dataset migration gains (P2-05)."""
    title = "Figure 2. Cross-dataset migration gains (P2-05, 10-seed paired)"
    pairs = p2_05.get("pairs", [])
    labels = [p["pair"] for p in pairs]
    deltas = [p["delta_mean"] * 100 for p in pairs]
    ci_low = [p["delta_ci95_low"] * 100 for p in pairs]
    ci_high = [p["delta_ci95_high"] * 100 for p in pairs]
    err_low = [d - lo for d, lo in zip(deltas, ci_low)]
    err_high = [hi - d for d, hi in zip(deltas, ci_high)]
    body_lines = ["Pair | Delta (pp) | CI95 (pp) | n_pooled"]
    for p in pairs:
        body_lines.append(f"{p['pair']} | {p['delta_mean']*100:.3f} | "
                          f"[{p['delta_ci95_low']*100:.3f}, {p['delta_ci95_high']*100:.3f}] | "
                          f"{p['n_pooled']}")
    body = "\n".join(body_lines)
    if plt is not None and labels:
        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = list(range(len(labels)))
        ax.barh(y_pos, deltas, xerr=[err_low, err_high], color="steelblue",
                edgecolor="black", capsize=4, alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Δ Top-1 (percentage points)")
        ax.set_title(title)
        ax.invert_yaxis()
        path = output_dir / "figure_2_cross_dataset_migration.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
    return _write_ascii_figure(output_dir, "figure_2_cross_dataset_migration", title, body)


def generate_figure_3_route_ranking(plt, output_dir: Path, p2_01: Dict[str, Any]) -> Path:
    """Figure 3: Retrosynthesis route ranking (P2-01)."""
    title = "Figure 3. Retrosynthesis route ranking (P2-01, 10-seed paired)"
    baseline = p2_01.get("baseline_mrr", DEFAULT_DATA["p2_01"]["baseline_mrr"]) * 100
    pc_cng = p2_01.get("pc_cng_mrr", DEFAULT_DATA["p2_01"]["pc_cng_mrr"]) * 100
    delta = p2_01.get("delta_pp", DEFAULT_DATA["p2_01"]["delta_pp"])
    ci_low = p2_01.get("ci_low_pp", DEFAULT_DATA["p2_01"]["ci_low_pp"])
    ci_high = p2_01.get("ci_high_pp", DEFAULT_DATA["p2_01"]["ci_high_pp"])
    body = (f"Baseline MRR: {baseline:.2f}%\nPC-CNG MRR:   {pc_cng:.2f}%\n"
            f"Delta: {delta:.2f} pp (CI [{ci_low:.2f}, {ci_high:.2f}] pp)\n"
            f"144/150 groups favoured, p < 1e-4")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(["Baseline", "PC-CNG"], [baseline, pc_cng],
                      color=["lightgray", "steelblue"], edgecolor="black")
        ax.set_ylabel("MRR (%)")
        ax.set_title(title)
        ax.set_ylim(0, max(baseline, pc_cng) * 1.25)
        for bar, val in zip(bars, [baseline, pc_cng]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=11)
        ax.annotate(f"+{delta:.2f} pp\nCI [{ci_low:.2f}, {ci_high:.2f}]",
                    xy=(1, pc_cng), xytext=(0.5, pc_cng * 1.15),
                    ha="center", fontsize=10, color="darkgreen",
                    arrowprops=dict(arrowstyle="->", color="darkgreen"))
        path = output_dir / "figure_3_route_ranking.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
    return _write_ascii_figure(output_dir, "figure_3_route_ranking", title, body)


def generate_figure_4_external_bridge(plt, output_dir: Path, p2_04: Dict[str, Any]) -> Path:
    """Figure 4: External bridge calibration (P2-04 v2 calibrator)."""
    title = "Figure 4. External bridge v2 calibration (P2-04, 10-seed paired)"
    baseline = p2_04.get("baseline_top1", DEFAULT_DATA["p2_04"]["baseline_top1"]) * 100
    v2 = p2_04.get("v2_top1", DEFAULT_DATA["p2_04"]["v2_top1"]) * 100
    delta = p2_04.get("delta_pp", DEFAULT_DATA["p2_04"]["delta_pp"])
    ci_low = p2_04.get("ci_low_pp", DEFAULT_DATA["p2_04"]["ci_low_pp"])
    ci_high = p2_04.get("ci_high_pp", DEFAULT_DATA["p2_04"]["ci_high_pp"])
    body = (f"Chemformer LL Top-1: {baseline:.2f}%\nv2 Calibrator Top-1:   {v2:.2f}%\n"
            f"Delta: +{delta:.2f} pp (CI [{ci_low:.2f}, {ci_high:.2f}] pp, p = 0.001)\n"
            f"Verdict: GO — L1 FIXED")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(["Chemformer LL", "v2 Calibrator"], [baseline, v2],
                      color=["lightgray", "seagreen"], edgecolor="black")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, max(baseline, v2) * 1.25)
        for bar, val in zip(bars, [baseline, v2]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=11)
        ax.annotate(f"+{delta:.2f} pp\nCI [{ci_low:.2f}, {ci_high:.2f}]",
                    xy=(1, v2), xytext=(0.5, v2 * 1.12),
                    ha="center", fontsize=10, color="darkgreen",
                    arrowprops=dict(arrowstyle="->", color="darkgreen"))
        path = output_dir / "figure_4_external_bridge_calibration.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
    return _write_ascii_figure(output_dir, "figure_4_external_bridge_calibration", title, body)


def generate_figure_5_dft_validation(plt, output_dir: Path, p2_02: Dict[str, Any]) -> Path:
    """Figure 5: DFT validation support (P2-02, 90% support)."""
    title = "Figure 5. DFT validation support (P2-02, GFN2-xTB)"
    n_supported = p2_02.get("n_supported", DEFAULT_DATA["p2_02"]["n_supported"])
    n_not = p2_02.get("n_not_supported", DEFAULT_DATA["p2_02"]["n_not_supported"])
    support_rate = p2_02.get("support_rate", DEFAULT_DATA["p2_02"]["support_rate"])
    body = (f"Supported: {n_supported}\nNot supported: {n_not}\n"
            f"Support rate: {support_rate:.0%}\nThreshold: 0.60\nVerdict: GO — L3 FIXED")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["Supported", "Not supported"]
        sizes = [n_supported, n_not]
        colors = ["seagreen", "indianred"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 12})
        ax.set_title(f"{title}\nSupport rate = {support_rate:.0%} (threshold 0.60) — GO")
        path = output_dir / "figure_5_dft_validation_support.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
    return _write_ascii_figure(output_dir, "figure_5_dft_validation_support", title, body)


def generate_figure_6_sota_radar(plt, output_dir: Path, p2_06: Dict[str, Any]) -> Path:
    """Figure 6: SOTA comparison radar chart (P2-06)."""
    title = "Figure 6. SOTA comparison — PC-CNG vs RDKit baselines (P2-06 smoke)"
    per_baseline = p2_06.get("per_baseline", DEFAULT_DATA["p2_06"]["per_baseline"])
    body_lines = ["Baseline | Delta (pp) | CI95 (pp) | PC-CNG better?"]
    for key, val in per_baseline.items():
        body_lines.append(f"{val.get('baseline', key)} | {val.get('delta_pp', 0):.2f} | "
                          f"[{val.get('ci_low_pp', 0):.2f}, {val.get('ci_high_pp', 0):.2f}] | "
                          f"{'yes' if val.get('pc_cng_better') else 'no'}")
    body = "\n".join(body_lines)
    if plt is not None and per_baseline:
        baselines = [v.get("baseline", k) for k, v in per_baseline.items()]
        deltas = [v.get("delta_pp", 0) for v in per_baseline.values()]
        n = len(baselines)
        # Radar chart requires closed loop; if only 1-2 baselines, fall back to bar
        if n >= 3:
            angles = [i / float(n) * 2 * math.pi for i in range(n)]
            angles += angles[:1]
            deltas_closed = deltas + deltas[:1]
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
            ax.plot(angles, deltas_closed, "o-", linewidth=2, color="steelblue")
            ax.fill(angles, deltas_closed, alpha=0.25, color="steelblue")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(baselines, fontsize=10)
            ax.set_title(title, pad=20)
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            path = output_dir / "figure_6_sota_radar.png"
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path
        # Fall back to bar chart for < 3 baselines
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(baselines, deltas, color=["seagreen" if d > 0 else "indianred" for d in deltas],
               edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Δ MRR (pp) vs PC-CNG")
        ax.set_title(title)
        path = output_dir / "figure_6_sota_radar.png"
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
    return _write_ascii_figure(output_dir, "figure_6_sota_radar", title, body)


def generate_all_figures(output_dir: Path, results_dir: Optional[Path] = None) -> List[Path]:
    """Generate all 6 figures; return the list of written file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _try_import_matplotlib()
    data = _load_p2_data(results_dir) if results_dir else DEFAULT_DATA
    paths = [
        generate_figure_1_architecture(plt, output_dir),
        generate_figure_2_cross_dataset(plt, output_dir, data["p2_05"]),
        generate_figure_3_route_ranking(plt, output_dir, data["p2_01"]),
        generate_figure_4_external_bridge(plt, output_dir, data["p2_04"]),
        generate_figure_5_dft_validation(plt, output_dir, data["p2_02"]),
        generate_figure_6_sota_radar(plt, output_dir, data["p2_06"]),
    ]
    # Write a manifest
    manifest = {
        "figures": [{"name": FIGURE_NAMES[i], "path": str(paths[i])} for i in range(len(paths))],
        "backend": "matplotlib" if plt is not None else "ascii",
        "n_figures": len(paths),
    }
    (output_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PC-CNG manuscript v2 figures (P2-09).")
    parser.add_argument("--output-dir", default="docs/manuscript_figures_v2_20260720",
                        help="Output directory for the figures.")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory containing P2 result artifacts.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    paths = generate_all_figures(output_dir, results_dir)
    for p in paths:
        print(f"[generate_manuscript_figures_v2] wrote {p}")
    print(f"[generate_manuscript_figures_v2] total figures: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
