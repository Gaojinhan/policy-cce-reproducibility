"""CNC paper tables and figures, assembled from independently replayed statistics.

This module performs presentation transformations only. It never reads reference
table cells, runs a simulator, changes a policy distribution, or fits statistics.
The evidence returned beside every artifact retains unrounded values and units.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .offline import require


MECHANISMS = (
    "M1_price_first", "M2_price_critical", "M3_delivery_first", "M4_delivery_critical",
)
LABELS = dict(zip(MECHANISMS, ("AUC1", "AUC2", "AUC3", "AUC4"), strict=True))
AVAIL = "at_least_three_route_capacity_feasible_manufacturers_rate"
RELATIVE_METRICS = (
    ("payment_per_assignment", "Payment per assigned order"),
    ("manufacturer_discounted_profit", "Manufacturer profit"),
    ("assignment_rate", "Assignment rate"),
    ("normalized_winner_hhi", "Allocation concentration (HHI)"),
    ("conditional_bid_rate", "Valid-bid rate"),
    ("average_markup", "Submitted markup"),
)
FACTOR_METRICS = (
    ("assignment_rate", "Assigned orders: loss", -100., 2),
    (AVAIL, r"Orders with $\geq3$ suppliers: loss", -100., 2),
    (AVAIL + "_F4", r"F4 orders with $\geq3$ suppliers: loss", -100., 2),
    ("orders_with_scalar_route_mismatch_rate", "Route mismatch: increase", 100., 2),
    ("route_capacity_feasible_manufacturers_per_order", "Suppliers per order: loss", -1., 3),
    *(("machine_group_remaining_capacity_" + group, group + " capacity: loss", -100., 2)
      for group in ("T", "M3", "M5", "EDM", "G")),
)
MECHANISM_PANELS = (
    ("payment_per_assignment", "Payment per assigned order", "Normalized monetary units", 1., 0., 1),
    ("manufacturer_discounted_profit", "Discounted manufacturer profit", "Normalized monetary units", 1., 0., 0),
    ("conditional_bid_rate", "Valid-bid rate", "Percent", 100., 0., 1),
    ("assignment_rate", "Assignment rate", "Percent", 100., 0., 1),
    ("normalized_winner_hhi", "Normalized HHI", "Index (0-1)", 1., 0., 3),
    ("submitted_lead_time_multiplier", "Submitted lead time", "Difference from requested due time (%)", 100., -100., 2),
)


def _number(value: Any) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Non-finite or non-numeric paper value")
    return float(value)


def _indexed(rows, key, expected):
    result = {}
    for row in rows:
        require(row[key] not in result, "Duplicate paper group: " + str(row[key]))
        result[row[key]] = row
    require(set(result) == set(expected), "Incomplete or unexpected paper groups: " + key)
    return result


def _interval(stats: Mapping[str, Any], scale=1.) -> dict[str, float]:
    """Apply units/sign exactly once, reversing endpoints for loss measures."""
    lower, upper = _number(stats["ci_lower"]), _number(stats["ci_upper"])
    require(lower <= upper, "Reversed input confidence interval")
    standard_error = _number(stats["standard_error"])
    require(standard_error >= 0, "Negative standard error")
    low, high = sorted((scale * lower, scale * upper))
    return {"estimate": scale * _number(stats["estimate"]), "ci_lower": low,
            "ci_upper": high, "standard_error": abs(scale) * standard_error}


def _signed(value, decimals=2):
    # Avoid displaying negative zero after rounding.
    rounded = round(_number(value), decimals)
    return f"{0. if rounded == 0 else rounded:+.{decimals}f}"


def _cells(stats, decimals=2):
    return (f"${_signed(stats['estimate'], decimals)}$",
            f"$[{_signed(stats['ci_lower'], decimals)},\\,{_signed(stats['ci_upper'], decimals)}]$")


def _table(caption, label, columns, body, notes="", wide=False):
    # Full floats are portable to the repository preview and the manuscript.
    environment = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    return "\n".join((
        r"\begin{" + environment + r"}[!htbp]", r"\centering", r"\caption{" + caption + "}",
        r"\label{" + label + "}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{" + width + r"}{@{\extracolsep{\fill}}" + columns + "@{}}",
        r"\toprule", body, r"\bottomrule", r"\end{tabular*}",
        r"\par\smallskip\begin{minipage}{" + width + r"}\footnotesize " + notes + r"\end{minipage}",
        r"\end{" + environment + "}", "",
    ))


def _artifact(caption, label, columns, body, rows, notes="", wide=False, **extra):
    return {"caption": caption, "latex": _table(caption, label, columns, body, notes, wide),
            "rows": rows, "mismatches": [], **extra}


def _audit_rows(records):
    rows = []
    groups = [*[(mechanism, [r for r in records if r["mechanism"] == mechanism])
                for mechanism in MECHANISMS], ("All", records)]
    for mechanism, cases in groups:
        require(bool(cases), "Empty CNC audit group")
        gaps = [_number(r["statistics"]["relative_nominal_gap_percent"]) for r in cases]
        errors = [_number(r["statistics"]["max_replacement_gain_standard_error"]) for r in cases]
        bounds = [_number(r["statistics"]["max_t_relative_gap_ucb95_percent"]) for r in cases]
        largest = max(range(len(cases)), key=lambda index: gaps[index])
        rows.append({"mechanism": mechanism, "games": len(cases), "mean_gap_percent": mean(gaps),
                     "max_gap_percent": max(gaps), "gain_se_min": min(errors), "gain_se_max": max(errors),
                     "gain_se_mean": mean(errors), "upper_bound_min_percent": min(bounds),
                     "upper_bound_max_percent": max(bounds), "largest_gap_job_id": cases[largest]["job_id"],
                     "largest_gap_upper_bound_percent": bounds[largest],
                     "source_job_ids": [r["job_id"] for r in cases]})
    return rows


def _cnc_audit_table(records):
    rows = _audit_rows(records)
    lines = [r"Mechanism & Games & \shortstack{Mean gap\\(\%)} & \shortstack{Max gap\\(\%)} & \shortstack{Max gain\\SE} & \shortstack{Upper\\bound (\%)} \\", r"\midrule"]
    for row in rows:
        if row["mechanism"] == "All":
            lines.append(r"\midrule")
        lines.append(f"{LABELS.get(row['mechanism'], row['mechanism'])} & {row['games']} & "
                     f"{row['mean_gap_percent']:.2f} & {row['max_gap_percent']:.2f} & "
                     f"{row['gain_se_min']:.2f}--{row['gain_se_max']:.2f} & "
                     f"{row['upper_bound_min_percent']:.2f}--{row['upper_bound_max_percent']:.2f} " + r"\\")
    all_cases = rows[-1]
    notes = (r"The last two columns give ranges across cases: each case's largest replacement-gain standard error "
             r"and its one-sided 95\% gap upper bound. Standard errors use normalized payoff units; gaps and bounds use percentages. "
             f"Mean per-game maximum gain SE: {all_cases['gain_se_mean']:.2f}; "
             f"upper bound for the largest-gap case: {all_cases['largest_gap_upper_bound_percent']:.2f}" + r"\%.")
    return _artifact("Independent gaps in the CNC case study.", "tab:cnc_audit", "lrrrrr", "\n".join(lines), rows, notes)


def _transplant_table(records):
    rows = []
    directions = ((MECHANISMS[0], MECHANISMS[1]), (MECHANISMS[1], MECHANISMS[0]),
                  (MECHANISMS[2], MECHANISMS[3]), (MECHANISMS[3], MECHANISMS[2]))
    require(set((r["source_mechanism"], r["target_mechanism"]) for r in records) == set(directions),
            "Unexpected policy-transfer direction")
    for source, target in directions:
        cases = [r for r in records if (r["source_mechanism"], r["target_mechanism"]) == (source, target)]
        require(len(cases) == 24, "Expected 24 matched cases per transfer direction")
        summaries = [r["statistics"] for r in cases]
        arm_means = {arm: mean(_number(s["arm_relative_nominal_gap_percent"][arm]) for s in summaries)
                     for arm in ("same_control", "cross_transplant", "target_recomputed")}
        increases = [_number(s["effects"]["cross_minus_same_control_percent"]) for s in summaries]
        for s, increase in zip(summaries, increases, strict=True):
            arms = s["arm_relative_nominal_gap_percent"]
            require(math.isclose(increase, arms["cross_transplant"] - arms["same_control"], abs_tol=1e-10),
                    "Transplant effect disagrees with its arms")
        lower_bounds = [_number(s["effect_intervals"]["cross_minus_same_control_percent"]["lower"]) for s in summaries]
        rows.append({"source": source, "target": target, "case_count": len(cases), **arm_means,
                     "increase_pp": mean(increases), "individual_ci_positive_count": sum(v > 0 for v in lower_bounds),
                     "minimum_individual_ci_lower_pp": min(lower_bounds),
                     "source_job_ids": [r["job_id"] for r in cases]})
    lines = [r"Source $\rightarrow$ target & \shortstack{Control\\(\%)} & \shortstack{Transferred\\(\%)} & \shortstack{Recomputed\\(\%)} & \shortstack{Increase\\(pp)} \\", r"\midrule"]
    for row in rows:
        values = " & ".join(f"{row[key]:.2f}" for key in ("same_control", "cross_transplant", "target_recomputed", "increase_pp"))
        lines.append(LABELS[row["source"]] + r" $\rightarrow$ " + LABELS[row["target"]] + " & " + values + r" \\")
    notes = ("Entries are mean independent relative gaps over 24 matched cases per direction. Control evaluates the source "
             "distribution in its own mechanism; Transferred evaluates it in the target mechanism; Recomputed uses the target "
             "mechanism's own distribution. Increase is Transferred minus Control, in percentage points (pp).")
    return _artifact("Policy-transplant results across payment rules.", "tab:transplant", "lrrrr", "\n".join(lines), rows, notes,
                     individual_ci_scope="Each interval applies to one matched case; no aggregate interval is inferred.")


def _relative_table(report):
    contrasts = _indexed(report["relative_mechanism_effects"], "contrast",
                         ("AUC2-AUC1", "AUC4-AUC3", "AUC3-AUC1", "AUC4-AUC2"))
    sections = (("A. Payment rule: own-price to threshold", "AUC2-AUC1", "AUC4-AUC3",
                 "Price only (AUC2/AUC1)", "Price + delivery (AUC4/AUC3)"),
                ("B. Bid format: price only to price + delivery", "AUC3-AUC1", "AUC4-AUC2",
                 "Own-price (AUC3/AUC1)", "Threshold (AUC4/AUC2)"))
    lines, rows = [], []
    for section, left, right, left_label, right_label in sections:
        if lines:
            lines.append(r"\midrule")
        lines.extend((r"\multicolumn{5}{@{}l}{\textbf{" + section + r"}}\\[3pt]",
                      r"& \multicolumn{2}{c}{" + left_label + r"} & \multicolumn{2}{c}{" + right_label + r"}\\",
                      r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
                      r"Outcome & Change (\%) & 95\% CI & Change (\%) & 95\% CI\\", r"\midrule"))
        metrics = RELATIVE_METRICS + (("submitted_lead_time_multiplier", "Quoted lead time"),) if section.startswith("A.") else RELATIVE_METRICS
        for metric, label in metrics:
            displayed = []
            evidence = {"section": section[0], "metric": metric, "unit": "percent", "contrasts": {}}
            for name in (left, right):
                if metric == "submitted_lead_time_multiplier" and name == "AUC2-AUC1":
                    displayed.extend(("---", "---"))
                    evidence["contrasts"][name] = None
                else:
                    interval = _interval(contrasts[name]["metrics"][metric])
                    evidence["contrasts"][name] = interval
                    displayed.extend(_cells(interval))
            lines.append(label + " & " + " & ".join(displayed) + r"\\")
            rows.append(evidence)
    notes = (r"Changes and their pointwise 95\% paired bootstrap confidence intervals are in percent. "
             r"Each change is \(100(\bar y_{\mathrm{target}}/\bar y_{\mathrm{reference}}-1)\), with equal weight for each of "
             "24 matched cases. Both means and their ratio are recalculated in every bootstrap sample. The intervals describe "
             "simulation uncertainty for the fixed cases and computed distributions. Quoted-lead-time changes use the submitted "
             "lead-time multiplier and apply only to AUC4/AUC3.")
    return _artifact("Relative effects of payment rules and bid formats.", "tab:cnc_mechanism_relative", "lrrrr", "\n".join(lines), rows, notes, wide=True)


def _capacity_table(report):
    factors = _indexed(report["operating_factor_effects"], "factor", ("platform_load", "route_mix", "outside_load"))
    titles = ("A. Higher platform workload", "B. M5/EDM-intensive order mix", "C. Higher outside workload")
    lines, rows = [], []
    for factor, title in zip(("platform_load", "route_mix", "outside_load"), titles, strict=True):
        if lines:
            lines.append(r"\midrule")
        lines.extend((r"\multicolumn{3}{@{}l}{\textbf{" + title + r"}}\\[2pt]",
                      r"Outcome & Change & 95\% CI\\", r"\midrule"))
        for metric, label, scale, decimals in FACTOR_METRICS:
            interval = _interval(factors[factor]["metrics"][metric], scale)
            rows.append({"factor": factor, "metric": metric, "display_label": label,
                         "display_scale": scale, "unit": "count" if scale == -1 else "percentage_points", **interval})
            lines.append(label + " & " + " & ".join(_cells(interval, decimals)) + r"\\")
    notes = ("Each comparison averages 48 matched case pairs, holding the other factors fixed. Positive values indicate losses "
             "of assignment, supplier availability, or capacity, or increased route mismatch; negative values indicate the reverse. "
             r"Units are percentage points, except for suppliers per order, which is a count. F4 availability uses F4 orders only. "
             r"Intervals are pointwise 95\% paired bootstrap intervals for the fixed cases and computed distributions. "
             "Case ratios are recalculated within each bootstrap sample before averaging.")
    return _artifact("Operating-factor effects and confidence intervals.", "tab:cnc_capacity_paired", "lrr", "\n".join(lines), rows, notes, wide=True)


def _family_table(report):
    families = _indexed(report["family_mismatch"], "family", ("F1", "F2", "F3", "F4"))
    rows, lines = [], [r"Family & Difference (pp) & 95\% CI (pp) \\", r"\midrule"]
    for family in ("F1", "F2", "F3", "F4"):
        require(families[family]["unit"] == "percentage_points", "Family mismatch is not in percentage points")
        interval = _interval(families[family])
        rows.append({"family": family, "unit": "percentage_points", **interval})
        lines.append(family + " & " + " & ".join(_cells(interval, 3)) + r"\\")
    notes = ("Among manufacturer--order pairs accepted by the total-capacity check, entries give the share rejected by the "
             "machine-group check in each family. They compare the M5/EDM-intensive mix with the balanced mix over 48 matched "
             "case pairs, in percentage points, with pointwise paired intervals.")
    return _artifact("Order-mix effects on feasibility errors by family.", "tab:cnc_family_mismatch", "lrr", "\n".join(lines), rows, notes)


def _endpoint_table(report):
    endpoints = _indexed(report["endpoint_means"], "label", ("Reference", "Combined pressure"))
    metrics = ("assignment_rate", AVAIL, *(AVAIL + "_F" + str(i) for i in range(1, 5)))
    rows, lines = [], [r"Condition & \shortstack{Assignment\\(\%)} & \multicolumn{5}{c}{\shortstack{Three-manufacturer\\availability (\%)}} \\",
                       r"\cmidrule{3-7}", r" & & \shortstack{All\\orders} & F1 & F2 & F3 & F4 \\", r"\midrule"]
    for label in ("Reference", "Combined pressure"):
        endpoint = endpoints[label]
        require(endpoint["n_cases"] == 12, "Expected 12 endpoint cases")
        values = {metric: 100 * _number(endpoint["metrics"][metric]) for metric in metrics}
        require(all(0 <= value <= 100 for value in values.values()), "Invalid endpoint percentage")
        rows.append({"label": label, "condition": endpoint["condition"], "case_count": endpoint["n_cases"], "unit": "percent", "metrics": values})
        display = r"\shortstack[l]{Combined\\pressure}" if label == "Combined pressure" else label
        lines.append(display + " & " + " & ".join(f"{values[metric]:.2f}" for metric in metrics) + r"\\")
    notes = ("Reference uses normal platform and outside workloads with a balanced mix; Combined pressure applies all three "
             "pressures together. Entries average four mechanisms and three matched replications. Ratios are calculated "
             "within each case before averaging; family rates use orders in that family, and the overall rate reflects each condition's order mix.")
    return _artifact("Assignment and supplier availability under combined pressure.", "tab:cnc_capacity_endpoints", "lrrrrrr", "\n".join(lines), rows, notes)


def _mechanism_figure_data(report):
    levels = _indexed(report["mechanism_levels"], "mechanism", MECHANISMS)
    panels = []
    for metric, title, unit, scale, offset, decimals in MECHANISM_PANELS:
        points = []
        for index, mechanism in enumerate(MECHANISMS):
            require(levels[mechanism]["n_cases"] == 24, "Expected 24 cases per mechanism")
            if metric == "submitted_lead_time_multiplier" and index < 2:
                continue
            stats = levels[mechanism]["metrics"][metric]
            standard_deviation = _number(stats["sample_sd"])
            require(standard_deviation >= 0, "Negative between-case standard deviation")
            points.append({"mechanism": mechanism, "bid_format": index // 2, "payment": index % 2,
                           "mean": scale * _number(stats["mean"]) + offset,
                           "sample_sd": abs(scale) * standard_deviation})
        panels.append({"metric": metric, "title": title, "unit": unit, "decimals": decimals, "points": points,
                       "error_bar": "between-case sample standard deviation", "scale": scale, "offset": offset})
    return panels


def _policy_figure_data(report):
    groups = _indexed(report["policy_composition"], "mechanism", MECHANISMS)
    rows = []
    for mechanism in MECHANISMS:
        values = {}
        for key in ("composition", "winner_shares"):
            shares = [_number(x) for x in groups[mechanism][key]]
            require(len(shares) == 3 and all(0 <= x <= 1 for x in shares)
                    and math.isclose(sum(shares), 1., abs_tol=1e-10), "Invalid policy-group shares")
            values[key] = [100 * share for share in shares]
        rows.append({"mechanism": mechanism, "unit": "percent", **values})
    return rows


def _plot_style():
    return {"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
            "axes.titleweight": "bold", "axes.titlecolor": "#193650", "axes.labelsize": 11,
            "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.spines.top": False,
            "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200}


def _plot_mechanisms(panels, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors, ink, muted = ("#3c83ac", "#df9828"), "#193650", "#5c6d7d"
    with plt.rc_context(_plot_style()):
        figure, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
        for index, (axis, panel) in enumerate(zip(axes.flat, panels, strict=True)):
            lower, upper = [], []
            for point in sorted(panel["points"], key=lambda point: (point["payment"], point["bid_format"])):
                x = point["bid_format"] + (point["payment"] - .5) * .32
                y, error = point["mean"], point["sample_sd"]
                axis.bar(x, y, .32, color=colors[point["payment"]], alpha=.92, zorder=3,
                         label=("Own-price payment", "Threshold payment")[point["payment"]]
                         if point["bid_format"] == 0 else None)
                axis.errorbar(x, y, yerr=error, fmt="none", color=muted, capsize=3, linewidth=1, zorder=4)
                lower.append(y - error)
                upper.append(y + error)
            ymin, ymax = min(0., min(lower)), max(upper)
            span = max(ymax - ymin, .02)
            for point in panel["points"]:
                x = point["bid_format"] + (point["payment"] - .5) * .32
                axis.text(x, point["mean"] + point["sample_sd"] + .035 * span,
                          f"{point['mean']:.{panel['decimals']}f}", ha="center", va="bottom",
                          color=muted, weight="bold", fontsize=10)
            axis.set_ylim(ymin - (.12 * span if ymin < 0 else 0), ymax + .23 * span)
            if index == 5:
                axis.text(0, .45 * (ymax + .23 * span), "N/A", ha="center", color=muted, weight="bold")
                axis.axhline(0, color=".6", lw=.7)
            axis.set_title(f"({chr(97 + index)}) {panel['title']}", pad=12)
            axis.set_ylabel(panel["unit"])
            axis.set_xticks([0, 1], ["Price only", "Price + delivery"])
            axis.set_xlim(-.52, 1.52)
            axis.grid(axis="y", color="#d9e0e5", lw=.7)
            axis.set_axisbelow(True)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                      bbox_to_anchor=(.5, 1.02), labelcolor=ink)
        figure.subplots_adjust(left=.07, right=.985, bottom=.08, top=.90, hspace=.53, wspace=.39)
        try:
            for suffix in (".pdf", ".png"):
                with destination.with_suffix(suffix).open("xb") as stream:
                    figure.savefig(stream, format=suffix[1:], bbox_inches="tight", pad_inches=.10)
        finally:
            plt.close(figure)


def _plot_policies(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors, muted, ink = ("#238c82", "#706398", "#d8dfe6"), "#5c6d7d", "#193650"
    labels = ("A1: frequent low-margin", "A2: regular cost-plus", "A3 to A6: other policies")
    row_labels = ("Price only\nOwn-price payment", "Price only\nThreshold payment",
                  "Price + delivery\nOwn-price payment", "Price + delivery\nThreshold payment")
    with plt.rc_context(_plot_style()):
        figure, axes = plt.subplots(1, 2, sharey=True, figsize=(11.3, 4.7))
        for axis, key, title in zip(axes, ("composition", "winner_shares"),
                                    ("(a) Policies in selected distributions", "(b) Policies among winners"), strict=True):
            values = np.array([row[key] for row in rows])
            left = np.zeros(4)
            for group in range(3):
                axis.barh(np.arange(4), values[:, group], left=left, height=.56, color=colors[group],
                          edgecolor="white", linewidth=.8, label=labels[group])
                for row in range(4):
                    if values[row, group] >= 4:
                        axis.text(left[row] + values[row, group] / 2, row, f"{values[row, group]:.1f}",
                                  ha="center", va="center", color="white" if group < 2 else muted, fontsize=10)
                left += values[:, group]
            axis.set_xlim(0, 100)
            axis.set_xticks(range(0, 101, 20))
            axis.set_xlabel("Policy share (%)", color=ink)
            axis.set_title(title, pad=18)
            axis.axhline(1.5, color="#d9e0e5", lw=1)
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", length=0)
        axes[0].set_yticks(range(4), row_labels)
        axes[0].invert_yaxis()
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.53, 1.04),
                      ncol=3, frameon=False, labelcolor=ink, fontsize=10)
        figure.subplots_adjust(left=.21, right=.99, bottom=.13, top=.84, wspace=.15)
        try:
            for suffix in (".pdf", ".png"):
                with destination.with_suffix(suffix).open("xb") as stream:
                    figure.savefig(stream, format=suffix[1:], bbox_inches="tight", pad_inches=.10)
        finally:
            plt.close(figure)


def build_cnc(dataset, audit_report, outcomes_report, output_dir) -> dict:
    """Build six current CNC tables and two plots after all input replays pass."""
    dataset.validate_output_dir(output_dir)
    destination = Path(output_dir) / "figures"
    for stem in ("cnc_mechanism_levels", "cnc_policy_composition_original_style"):
        for suffix in (".pdf", ".png"):
            candidate = destination / (stem + suffix)
            if candidate.exists() or candidate.is_symlink():
                raise FileExistsError("Refusing to overwrite existing CNC artifact: " + str(candidate))
    for name, report in (("audit", audit_report), ("outcomes", outcomes_report)):
        require(report.get("status") == "pass", name + " replay did not pass")
        require(not report.get("mismatches") and not report.get("failures"), name + " replay contains mismatches or failures")
    require(outcomes_report.get("case_count") == 96, "Expected 96 CNC outcome cases")
    require(not outcomes_report.get("mismatch_count") and not audit_report.get("mismatch_field_count"), "Unresolved replay mismatches")
    expected = {family: {jid for jid, spec in dataset.jobs.items() if spec["family"] == family}
                for family in ("cnc_main", "policy_transplant")}
    require(all(len(ids) == 96 for ids in expected.values()), "Expected 96 CNC and 96 policy-transplant jobs")
    selected = {family: [] for family in expected}
    seen = set()
    for report in audit_report["jobs"]:
        family = report["family"]
        if family not in selected:
            continue
        jid = report["job_id"]
        require(jid not in seen, "Duplicate replayed job: " + jid)
        seen.add(jid)
        require(jid in expected[family] and report["status"] == "pass" and not report["mismatches"], "Invalid replayed CNC job: " + jid)
        result = dataset.result(jid)
        if family == "cnc_main":
            mechanism = result["mechanism"]
            require(mechanism in MECHANISMS, "Unknown CNC mechanism")
            statistics = report["recomputed"]["distribution_audits"]["platform_operating_score"]["statistics"]
            selected[family].append({"job_id": jid, "mechanism": mechanism, "statistics": statistics})
        else:
            selected[family].append({"job_id": jid, "source_mechanism": result["source_mechanism"],
                                     "target_mechanism": result["target_mechanism"],
                                     "statistics": report["recomputed"]["three_arm_summary"]})
    for family, records in selected.items():
        require({r["job_id"] for r in records} == expected[family], "Incomplete replayed paper scope: " + family)
    require(all(sum(r["mechanism"] == m for r in selected["cnc_main"]) == 24 for m in MECHANISMS), "Expected 24 CNC cases per mechanism")
    tables = {
        "tab:cnc_audit": _cnc_audit_table(selected["cnc_main"]),
        "tab:transplant": _transplant_table(selected["policy_transplant"]),
        "tab:cnc_mechanism_relative": _relative_table(outcomes_report),
        "tab:cnc_capacity_paired": _capacity_table(outcomes_report),
        "tab:cnc_family_mismatch": _family_table(outcomes_report),
        "tab:cnc_capacity_endpoints": _endpoint_table(outcomes_report),
    }
    panels, policies = _mechanism_figure_data(outcomes_report), _policy_figure_data(outcomes_report)
    destination.mkdir(parents=True, exist_ok=True)
    _plot_mechanisms(panels, destination / "cnc_mechanism_levels")
    _plot_policies(policies, destination / "cnc_policy_composition_original_style")
    figures = {
        "fig:cnc_mechanism_tradeoffs": {"caption": "Outcomes under the four CNC auction mechanisms.",
            "pdf": str(destination / "cnc_mechanism_levels.pdf"), "png": str(destination / "cnc_mechanism_levels.png"), "rows": panels, "mismatches": []},
        "fig:cnc_policy_composition": {"caption": "Policy shares in the computed distributions and assigned orders.",
            "pdf": str(destination / "cnc_policy_composition_original_style.pdf"), "png": str(destination / "cnc_policy_composition_original_style.png"), "rows": policies, "mismatches": []},
    }
    return {"status": "pass", "tables": tables, "figures": figures, "mismatches": []}
