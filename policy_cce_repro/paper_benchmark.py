"""Paper benchmark displays from saved results and independently replayed audits.

No manuscript values are inputs. Historical reporting records are comparison
targets only. No solver, simulator, cloud service, or manuscript writer is used.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from cmfg_cce.evaluation.revision_statistics import support_diversity
from cmfg_cce.experiments.run_revision_cnc import _distribution_from_payload
from policy_cce_repro.offline import require
from policy_cce_repro.recompute_audits import _differences


SCALES = ((4, 6), (4, 8), (5, 6), (5, 8), (6, 6), (6, 8))
LARGE = ((8, 8), (10, 8))
SOLVERS = ("FullTensor-CCE-LP", "ExhaustiveCG-CCE", "MWU-PolicyTrace", "REPAIR-SAD-CCE")
SPARSE_SOLVERS = SOLVERS[2:]
NAMES = dict(zip(SOLVERS, ("Full-space LP", "Exact CG", "MWU", "DSS-CCE"), strict=True))
TABLE_NAMES = dict(zip(SOLVERS, ("FullSpace-CCE-LP", "ColumnGen-CCE", "MWU", "DSS-CCE"), strict=True))
COLORS = dict(zip(SOLVERS, ("#405d80", "#8093a6", "#b37c27", "#aa3d46"), strict=True))
MARKERS = dict(zip(SOLVERS, ("o", "s", "^", "D"), strict=True))
CELL_FIELDS = ("relative_nominal_gap_percent", "runtime_seconds", "training_profile_count", "profile_saving_percent", "support_size", "effective_support_size")
UNCERTAINTY_FIELDS = ("max_payoff_standard_error", "max_replacement_gain_standard_error", "relative_nominal_gap_percent", "max_t_relative_gap_ucb95_percent")


def _ms(values):
    values = [float(value) for value in values]
    require(bool(values) and bool(np.all(np.isfinite(values))), "Empty or nonfinite benchmark values")
    return {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else 0.0, "min": min(values), "max": max(values), "n": len(values)}


def _cell(rows, scale, solver):
    selected = [row for row in rows if (row["N"], row["J"]) == scale and row["solver"] == solver]
    require(len(selected) == (12 if scale in SCALES else 6), f"Incomplete benchmark cell: {scale}/{solver}")
    expected_mechanisms = {"M1_price_first", "M2_price_critical", "M3_delivery_first", "M4_delivery_critical"} if scale in SCALES else {"M1_price_first", "M4_delivery_critical"}
    require({(row["mechanism"], row["seed"]) for row in selected} == {(mechanism, seed) for mechanism in expected_mechanisms for seed in range(3)}, "Duplicate or missing mechanism/replicate cell")
    return {"N": scale[0], "J": scale[1], "solver": solver, **{field: _ms(row[field] for row in selected) for field in CELL_FIELDS}}


def _runtime_contract(result):
    require(result["runtime_contract"]["empty_cache_per_solver"] is True, "Solver runtime did not start from an empty cache")
    require(result["runtime_contract"]["verification_runtime_included"] is False, "Verification time included in solver runtime")
    require(result["runtime_environment"]["workers"] == 32, "Unexpected runtime worker count")


def _sha(dataset, job_id):
    return dataset.read_json(dataset.results[job_id]["provenance_path"])["result_sha256"]


def _joined_job(dataset, result, audit_entry):
    job_id = result["job_id"]
    n, j = int(result["N"]), int(result["J"])
    require(audit_entry["job_id"] == job_id and audit_entry["status"] == "pass" and not audit_entry["mismatches"], "Audit replay missing or failed")
    link = result["runtime_q_provenance"]
    runtime_id = link["runtime_job_id"]
    runtime = dataset.result(runtime_id)
    require(runtime["family"] == result["family"] + "_runtime", "Runtime/audit family mismatch")
    require((runtime["reported_N"], runtime["reported_J"], runtime["mechanism"], runtime["seed"]) == (n, j, result["mechanism"], result["seed"]), "Runtime/audit case identity mismatch")
    metadata = dataset.jobs[runtime_id]["metadata"]
    require((metadata["n_agents"], metadata["policies_per_agent"]) == (n, j), "Runtime scale differs from frozen job specification")
    _runtime_contract(runtime)
    require(result["train_rollouts"] == 200 and result["audit_rollouts"] == 2000, "Unexpected benchmark rollout counts")
    runtime_sha = _sha(dataset, runtime_id)
    require(link["runtime_result_sha256"] == runtime_sha, "Runtime provenance SHA mismatch")
    expected = SOLVERS if result["family"] == "solver_benchmark" else SPARSE_SOLVERS
    require(set(result["solver_results"]) == set(runtime["solver_results"]) == set(expected), "Unexpected benchmark solver set")
    require(set(audit_entry["recomputed"]["solver_audits"]) == set(expected), "Audit replay solver set mismatch")
    rows, mismatches = [], []
    for solver in expected:
        source, timed = result["solver_results"][solver], runtime["solver_results"][solver]
        q = _distribution_from_payload(source["distribution"], n_agents=n, policy_ids=result["policy_ids"])
        timed_q = _distribution_from_payload(timed["distribution"], n_agents=n, policy_ids=result["policy_ids"])
        fresh = audit_entry["recomputed"]["solver_audits"][solver]
        require(q.q_hash == timed_q.q_hash == source["formal_audit"]["q_hash_before"] == source["formal_audit"]["q_hash_after"] == fresh["q_hash"] == link["q_hashes"][solver], "Runtime/audit/replay q identity mismatch")
        require(source["runtime_q_source_sha256"] == runtime_sha and source["runtime_q_source_job_id"] == runtime_id, "Per-solver runtime source mismatch")
        require(source["q_restored_without_resolving"] is True, "Audit q was resolved instead of restored")
        stats = fresh["statistics"]
        mismatches.extend(_differences(stats, source["formal_audit"], f"{job_id}.{solver}.audit"))
        diversity = asdict(support_diversity(q.probabilities))
        mismatches.extend(_differences(diversity, source["support_diversity"], f"{job_id}.{solver}.support_diversity"))
        elapsed = float(timed["runtime_seconds"])
        count = int(timed["training_profile_count"])
        require(np.isfinite(elapsed) and elapsed > 0 and 0 < count <= j ** n, "Invalid timed runtime/profile count")
        rows.append({
            "job_id": job_id, "family": result["family"], "N": n, "J": j,
            "mechanism": result["mechanism"], "seed": int(result["seed"]), "solver": solver,
            "q_hash": q.q_hash, "support_size": len(q.support),
            "effective_support_size": diversity["effective_support_size"], "largest_probability": diversity["largest_probability"],
            **{field: stats[field] for field in (*UNCERTAINTY_FIELDS, "max_t_relative_gap_lcb95_percent", "payoff_denominator")},
            "runtime_seconds": elapsed, "training_profile_count": count,
            "profile_saving_percent": 100 * (1 - count / j ** n),
            "runtime_job_id": runtime_id, "runtime_result_sha256": runtime_sha,
            "audit_result_sha256": _sha(dataset, job_id), "q_frozen_and_runtime_matched": True,
            "audit_rollouts": 2000, "train_rollouts": 200,
        })
    return rows, mismatches


def _aggregate(rows, pure_games):
    benchmark = [row for row in rows if row["family"] == "solver_benchmark"]
    by_job = {}
    for row in benchmark:
        by_job.setdefault(row["job_id"], {})[row["solver"]] = row
    require(len(by_job) == 72 and all(set(group) == set(SOLVERS) for group in by_job.values()), "Incomplete paired benchmark")
    solver_rows = []
    for solver in SOLVERS:
        solver_rows.append({
            "solver": solver,
            "fresh_gap_percent": _ms(group[solver]["relative_nominal_gap_percent"] for group in by_job.values()),
            "within_game_time_ratio_to_dss": _ms(group[solver]["runtime_seconds"] / group["REPAIR-SAD-CCE"]["runtime_seconds"] for group in by_job.values()),
        })
    ratio68 = mean(group["FullTensor-CCE-LP"]["runtime_seconds"] / group["REPAIR-SAD-CCE"]["runtime_seconds"] for group in by_job.values() if (group["REPAIR-SAD-CCE"]["N"], group["REPAIR-SAD-CCE"]["J"]) == (6, 8))
    return {
        "solver_rows": solver_rows, "game_count": len(by_job),
        "pure_nash_training_game_count": sum(game["pure_nash_count"] > 0 for game in pure_games),
        "dss_singleton_output_count": sum(group["REPAIR-SAD-CCE"]["support_size"] == 1 for group in by_job.values()),
        "mwu_mean_paired_time_overrun_percent": 100 * (solver_rows[2]["within_game_time_ratio_to_dss"]["mean"] - 1),
        "fullspace_to_dss_mean_paired_time_ratio_n6j8": ratio68,
    }


def collect_benchmark(dataset, audit_report):
    """Validate and calculate numeric evidence; this function creates no files."""
    require(audit_report["status"] == "pass", "Benchmark displays require a passing audit replay")
    audit_jobs = {entry["job_id"]: entry for entry in audit_report["jobs"]}
    require(len(audit_jobs) == len(audit_report["jobs"]), "Duplicate audit-report job ID")
    selected = sorted(job_id for job_id, spec in dataset.jobs.items() if spec["family"] in {"solver_benchmark", "scalability_sparse"})
    require(len(selected) == 84, "Expected the 84 original benchmark/sparse jobs")
    rows, pure_games, mismatches = [], [], []
    for job_id in selected:
        result = dataset.result(job_id)
        require((result["N"], result["J"]) in (SCALES if result["family"] == "solver_benchmark" else LARGE), "Unreported benchmark/scalability scale")
        job_rows, job_mismatches = _joined_job(dataset, result, audit_jobs[job_id])
        rows.extend(job_rows)
        mismatches.extend(job_mismatches)
        if result["family"] == "solver_benchmark":
            diagnostic = result["pure_policy_game_diagnostics"]
            require(diagnostic["complete_table"] is True and isinstance(diagnostic["pure_nash_count"], int), "Missing complete-training-game pure Nash diagnostic")
            pure_games.append({"job_id": job_id, "pure_nash_count": diagnostic["pure_nash_count"]})
    require(len(rows) == 312 and len(pure_games) == 72 and len({row["runtime_job_id"] for row in rows}) == 84, "Incomplete runtime/audit joins")
    rows.sort(key=lambda row: (row["N"], row["J"], row["mechanism"], row["seed"], SOLVERS.index(row["solver"])))
    cells = [_cell(rows, scale, solver) for scale in SCALES for solver in SOLVERS]
    sparse_cells = [_cell(rows, scale, solver) for solver in SPARSE_SOLVERS for scale in LARGE]
    nonpure_ids = {game["job_id"] for game in pure_games if game["pure_nash_count"] == 0}
    require(len(nonpure_ids) == 6, "Expected all six non-pure complete-training benchmark games")
    nonpure = [row for row in rows if row["job_id"] in nonpure_ids and row["solver"] == "REPAIR-SAD-CCE"]
    uncertainty = []
    for n, j in SCALES:
        group = [row for row in rows if (row["N"], row["J"]) == (n, j) and row["solver"] == "REPAIR-SAD-CCE"]
        uncertainty.append({"N": n, "J": j, "game_count": len(group), **{field: _ms(row[field] for row in group) for field in UNCERTAINTY_FIELDS}})
    n7_rows = []
    n7_ids = sorted(job_id for job_id, spec in dataset.jobs.items() if spec["family"] == "scalability_exact_runtime")
    require(len(n7_ids) == 12, "Expected 12 N7 runtime-only jobs")
    for job_id in n7_ids:
        result = dataset.result(job_id)
        require((result["reported_N"], result["reported_J"]) == (7, 6) and set(result["solver_results"]) == set(SOLVERS), "Unexpected N7 scale/solver scope")
        metadata = dataset.jobs[job_id]["metadata"]
        require((metadata["n_agents"], metadata["policies_per_agent"]) == (7, 6), "N7 runtime scale differs from frozen job specification")
        _runtime_contract(result)
        for solver in SOLVERS:
            timed = result["solver_results"][solver]
            require(not timed.get("formal_audit"), "N7 runtime-only table unexpectedly contains fresh audit")
            q = _distribution_from_payload(timed["distribution"], n_agents=7, policy_ids=tuple(f"A{i}" for i in range(1, 7)))
            elapsed, count = float(timed["runtime_seconds"]), int(timed["training_profile_count"])
            require(np.isfinite(elapsed) and elapsed > 0 and 0 < count <= 6 ** 7, "Invalid N7 runtime/profile count")
            n7_rows.append({"N": 7, "J": 6, "solver": solver, "job_id": job_id, "mechanism": result["mechanism"], "seed": int(result["seed"]), "runtime_seconds": elapsed, "training_profile_count": count, "profile_saving_percent": 100 * (1 - count / 6 ** 7), "q_hash": q.q_hash})
    n7 = []
    for solver in SOLVERS:
        group = [row for row in n7_rows if row["solver"] == solver]
        require(len({(row["mechanism"], row["seed"]) for row in group}) == 12, "Duplicate N7 mechanism/replicate")
        n7.append({"N": 7, "J": 6, "solver": solver, "runtime_seconds": _ms(row["runtime_seconds"] for row in group), "mean_training_profile_count": mean(row["training_profile_count"] for row in group), "mean_profile_saving_percent": mean(row["profile_saving_percent"] for row in group)})
    aggregate = _aggregate(rows, pure_games)
    reference = dataset.reference("solver_artifact_evidence.json")
    uncertainty_reference = dataset.reference("solver_uncertainty_evidence.json")
    comparison = {"benchmark_cells": cells, "selected_sparse_cells": sparse_cells, "runtime_only_extension": n7}
    mismatches.extend(_differences(comparison, reference, "historical_solver_evidence"))
    # Compare case identity as a set, not by historical source-file ordering.
    require({row["job_id"] for row in reference["nonpure_cases"]} == nonpure_ids, "Historical non-pure case set mismatch")
    mismatches.extend(_differences({"uncertainty_cells": uncertainty, "benchmark_summary": aggregate}, uncertainty_reference, "historical_uncertainty_evidence"))
    return {
        "status": "pass" if not mismatches else "fail", "mismatches": mismatches,
        "matched_rows": rows, "benchmark_cells": cells, "selected_sparse_cells": sparse_cells,
        "nonpure_cases": nonpure, "uncertainty_cells": uncertainty, "benchmark_summary": aggregate,
        "runtime_only_extension": n7, "runtime_only_rows": n7_rows,
        "checks": {"matched_audit_jobs": 84, "matched_solver_outputs": len(rows), "benchmark_games": 72, "sparse_games": 12, "n7_runtime_games": 12, "all_runtime_q_hashes_match": True, "no_n7_fresh_gap": True, "source_of_pure_nash_classification": "saved complete training-table diagnostics; no tensor/LP re-execution", "numeric_reference_role": "comparison only; never a calculation input"},
    }


def _body(headers, displayed):
    return " & ".join(headers) + r" \\" + "\n\\midrule\n" + "\n".join(" & ".join(row) + r" \\" for row in displayed) + "\n"


def format_tables(evidence):
    """Return current manuscript table precision and units without file writes."""
    from policy_cce_repro.paper_common import table
    tables = {}
    def add(label, caption, columns, headers, displayed, rows, notes="", wide=False):
        key = "tab:" + label
        tables[key] = {"caption": caption, "filename": label + ".tex", "rows": rows, "display_rows": displayed, "headers": headers, "latex": table(caption, key, columns, _body(headers, displayed), notes=notes, wide=wide)}
    shown = []
    for index, row in enumerate(evidence["benchmark_cells"]):
        gap = row["relative_nominal_gap_percent"]
        shown.append([f'$({row["N"]},{row["J"]})$' if index % 4 == 0 else "", TABLE_NAMES[row["solver"]], f'${gap["mean"]:.2f} \\pm {gap["sd"]:.2f}$', f'{row["runtime_seconds"]["mean"] / 60:.1f}', f'{row["training_profile_count"]["mean"]:,.0f}', f'{row["profile_saving_percent"]["mean"]:.1f}', f'{row["support_size"]["mean"]:.2f}'])
    add("solver_results", "Solver accuracy and computational effort across the six benchmark scales.", "llrrrrr", [r"$(N,J)$", "Method", r"Independent gap (\%)", "Time (min)", "$E$", r"Saving (\%)", "$|S|$"], shown, evidence["benchmark_cells"], r"Independent gap is the relative gap in Eq.~\eqref{eq:relative_gap}, evaluated with new simulations after fixing \(q\). Gaps are reported as mean \(\pm\) sample standard deviation at each scale; other entries are means. Runtime excludes verification. The evaluated-profile count \(E\) gives the saving rate \(100(1-E/J^N)\), and \(|S|\) counts profiles with positive probability.", wide=True)
    aggregate = evidence["benchmark_summary"]
    shown = [[TABLE_NAMES[row["solver"]], f'{row["fresh_gap_percent"]["mean"]:.2f}', f'{row["fresh_gap_percent"]["max"]:.2f}', f'{row["within_game_time_ratio_to_dss"]["mean"]:.2f}'] for row in aggregate["solver_rows"]]
    notes = r"Time ratios are calculated relative to DSS-CCE separately in each paired game, then averaged. " + f'MWU time overrun: {aggregate["mwu_mean_paired_time_overrun_percent"]:.1f}\\%; FullSpace-CCE-LP/DSS-CCE time ratio at $(6,8)$: {aggregate["fullspace_to_dss_mean_paired_time_ratio_n6j8"]:.1f}.\n\\par\nTraining games with a pure Nash equilibrium: {aggregate["pure_nash_training_game_count"]}/{aggregate["game_count"]}; single-profile DSS-CCE outputs: {aggregate["dss_singleton_output_count"]}/{aggregate["game_count"]}.'
    add("solver_benchmark_summary", "Aggregate solver results across the 72 paired benchmark games.", "lrrr", ["Method", r"\shortstack{Mean gap\\(\%)}", r"\shortstack{Max gap\\(\%)}", r"\shortstack{Time\\ratio}"], shown, aggregate["solver_rows"], notes)
    shown = [[f'$({row["N"]},{row["J"]})$', *[f'{row[field]["min"]:.2f}--{row[field]["max"]:.2f}' for field in UNCERTAINTY_FIELDS]] for row in evidence["uncertainty_cells"]]
    add("solver_uncertainty", "Simulation uncertainty in the independently evaluated DSS-CCE results.", "lrrrr", ["$(N,J)$", r"\shortstack{Max payoff\\SE}", r"\shortstack{Max gain\\SE}", r"\shortstack{Independent\\gap (\%)}", r"\shortstack{Upper bound\\(\%)}"], shown, evidence["uncertainty_cells"], r"Each cell gives the minimum and maximum across the 12 games at that scale. The SE columns use each game's largest payoff or replacement-gain standard error, measured in normalized payoff units. The independent gap and its one-sided 95\% upper bound are percentages. The bounds account for uncertainty across the policy replacements and in the payoff normalization (Section~\ref{sec:evaluation_settings}). They show how simulation error affects the estimated gaps.")
    shown = [[f'$({row["N"]},{row["J"]})$', "AUC" + row["mechanism"].split("_")[0][1:], str(row["seed"] + 1), str(row["support_size"]), f'{row["effective_support_size"]:.3f}', f'{row["relative_nominal_gap_percent"]:.3f}'] for row in evidence["nonpure_cases"]]
    add("nonpure_cases", "DSS-CCE results for the six benchmark games without a pure Nash equilibrium in the training game.", "llrrrr", ["$(N,J)$", "Mechanism", "Replicate", "$|S|$", r"\shortstack{Effective\\support}", r"\shortstack{Independent\\gap (\%)}"], shown, evidence["nonpure_cases"], r"Replicate identifies which of the three benchmark replications the result comes from. Effective support, \(\exp(H(q))\), also reflects how evenly probability is spread over the profiles, with \(H(q)\) denoting entropy.")
    shown = [[TABLE_NAMES[row["solver"]], f'${row["runtime_seconds"]["mean"] / 60:.1f} \\pm {row["runtime_seconds"]["sd"] / 60:.1f}$', f'{row["mean_training_profile_count"]:,.0f}', f'{row["mean_profile_saving_percent"]:.1f}'] for row in evidence["runtime_only_extension"]]
    add("scalability_exact_runtime", r"Computational effort at $(N,J)=(7,6)$.", "lrrr", ["Method", "Time (min)", "$E$", r"\shortstack{Saving\\(\%)}"], shown, evidence["runtime_only_extension"], r"The comparison covers four mechanisms with three replications each. Times are means \(\pm\) sample standard deviations; evaluated-profile counts \(E\) and saving rates are means. Each solver starts from an empty payoff cache, and runtime excludes verification. These outputs were checked against the training table; independent gap evaluation is outside this comparison's scope.")
    return tables


def _base_axes(ax, title, ylabel):
    ax.set_title(title, loc="left", pad=7)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.grid(axis="y", color="#dfdfdf", linewidth=0.45, zorder=0)
    ax.tick_params(width=0.6, length=3)


def _frontier(rows):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.72))
    fig.subplots_adjust(left=0.07, right=0.993, bottom=0.25, top=0.88, wspace=0.37)
    x = np.arange(len(SCALES))
    for si, solver in enumerate(SOLVERS):
        summaries = [_cell(rows, scale, solver) for scale in SCALES]
        for ai, key in enumerate(("relative_nominal_gap_percent", "runtime_seconds", "profile_saving_percent")):
            y = np.array([summary[key]["mean"] for summary in summaries])
            sd = np.array([summary[key]["sd"] for summary in summaries])
            kwargs = dict(color=COLORS[solver], marker=MARKERS[solver], markersize=3.5, linewidth=1.05, linestyle="--" if si == 1 else "-", label=NAMES[solver], zorder=3 + si)
            if ai == 0:
                axes[ai].errorbar(x + (si - 1.5) * 0.055, y, yerr=np.vstack([np.minimum(y, sd), sd]), capsize=1.8, elinewidth=0.75, **kwargs)
            else:
                axes[ai].plot(x + (si - 1.5) * 0.055, y, **kwargs)
    _base_axes(axes[0], "(a) Independent audit", "Nominal relative gap (%)")
    axes[0].set_yscale("symlog", linthresh=1, linscale=0.7, base=10)
    axes[0].set_ylim(-0.025, 130)
    axes[0].set_yticks([0, 0.5, 1, 10, 100])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    _base_axes(axes[1], "(b) Empty-cache runtime", "Wall-clock time (s)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(35, 22000)
    _base_axes(axes[2], "(c) Profile saving", "Unevaluated training profiles (%)")
    axes[2].set_ylim(-4, 104)
    axes[2].set_yticks([0, 25, 50, 75, 100])
    for ax in axes:
        ax.set_xticks(x, [f"{n},{j}" for n, j in SCALES])
        ax.set_xlabel("Scale $(N,J)$", labelpad=4)
        ax.set_xlim(-0.30, 5.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=4, frameon=False, handlelength=1.8, columnspacing=1.6)
    return fig


def _scalability(rows):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.72))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.87, wspace=0.35)
    metrics = ("runtime_seconds", "training_profile_count", "relative_nominal_gap_percent")
    for si, solver in enumerate(SPARSE_SOLVERS):
        offset = (si - 0.5) * 0.22
        for xi, scale in enumerate(LARGE):
            summary = _cell(rows, scale, solver)
            group = [row for row in rows if (row["N"], row["J"]) == scale and row["solver"] == solver]
            for ai, key in enumerate(metrics):
                factor = 1 / 60 if key == "runtime_seconds" else 1
                for row in group:
                    axes[ai].scatter(xi + offset + (row["seed"] - 1) * 0.035, row[key] * factor, s=16, marker="o" if row["mechanism"].startswith("M1") else "^", facecolor=COLORS[solver], alpha=0.48, edgecolor="none", zorder=3)
                m, sd = summary[key]["mean"] * factor, summary[key]["sd"] * factor
                axes[ai].errorbar(xi + offset, m, yerr=np.array([[min(m, sd)], [sd]]), fmt="_", markersize=11, color=COLORS[solver], capsize=3, linewidth=1.3, zorder=5)
    _base_axes(axes[0], "(a) Empty-cache runtime", "Wall-clock time (min)")
    _base_axes(axes[1], "(b) Evaluated profiles", "Distinct training profiles")
    _base_axes(axes[2], "(c) Independent audit", "Nominal relative gap (%)")
    axes[2].set_yscale("symlog", linthresh=1, linscale=0.7)
    axes[2].set_yticks([0, 1, 5, 10, 50, 100])
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    axes[2].set_ylim(-0.02, 100)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{int(value):,}"))
    for ax in axes:
        ax.set_xticks([0, 1], ["8", "10"])
        ax.set_xlabel("Manufacturers $N$ ($J=8$)", labelpad=4)
        ax.set_xlim(-0.38, 1.38)
    legend = [Line2D([0], [0], color=COLORS[solver], linewidth=1.7, label=NAMES[solver]) for solver in SPARSE_SOLVERS]
    legend.extend(Line2D([0], [0], color="#666666", marker=marker, linestyle="", markersize=4, label=label) for marker, label in (("o", "AUC1"), ("^", "AUC4")))
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=4, frameon=False, columnspacing=1.7)
    return fig


def build_benchmark(dataset, audit_report, output_dir) -> dict[str, Any]:
    """Build five table strings and two figures in an isolated new output tree."""
    output_dir = dataset.validate_output_dir(Path(output_dir))
    evidence = collect_benchmark(dataset, audit_report)
    result = {"tables": format_tables(evidence), "figures": {}, **evidence}
    if evidence["status"] != "pass":
        return result
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stems = ("solver_frontier", "scalability")
    require(not any((figure_dir / (stem + suffix)).exists() for stem in stems for suffix in (".pdf", ".png")), "Benchmark figure output already exists")
    from matplotlib import font_manager
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(name for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif") if name in available_fonts)
    style = {"font.family": font_family, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65, "lines.linewidth": 1.15, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "mathtext.fontset": "stix"}
    captions = ("Solver accuracy, independent end-to-end runtime, and profile saving across the six benchmark scales.", "Sparse-solver scalability with eight policies per manufacturer.")
    with matplotlib.rc_context(style):
        for stem, caption, factory in zip(stems, captions, (_frontier, _scalability), strict=True):
            fig = factory(evidence["matched_rows"])
            try:
                pdf, png = figure_dir / (stem + ".pdf"), figure_dir / (stem + ".png")
                # Exclusive opens ensure these reporting artifacts never overwrite.
                with pdf.open("xb") as handle:
                    fig.savefig(handle, format="pdf", bbox_inches="tight", metadata={"Title": stem.replace("_", " "), "Author": "PolicyCCE reproducibility package", "CreationDate": None, "ModDate": None})
                with png.open("xb") as handle:
                    fig.savefig(handle, format="png", bbox_inches="tight", dpi=300)
            finally:
                plt.close(fig)
            result["figures"]["fig:" + stem] = {"caption": caption, "pdf_path": str(pdf), "png_path": str(png), "font_family": font_family, "rows": evidence["benchmark_cells"] if stem == "solver_frontier" else evidence["selected_sparse_cells"], "units": ["percent", "seconds", "percent"] if stem == "solver_frontier" else ["minutes", "profiles", "percent"], "error_bars": "sample standard deviation; lower ends truncated at zero", "gap_axis": "symlog, linear through 1 percent; no omitted interval"}
    return result
