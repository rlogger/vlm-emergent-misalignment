"""Render proposed paper figures directly from the preserved component-study data.

No models or judges run. ASRs and paired mean changes are checked against saved
row-level judgments; confidence limits are the archived bootstrap results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/summaries/artifacts/Step3__step3_component_resolved_complete.json"

CONDITIONS = [
    ("baseline", "Baseline", "baseline"),
    ("repair_text_only_a600", "Response text: repair", "text"),
    ("repair_vision_only_a600", "Vision shift: repair", "vision"),
    ("repair_shared_a600", "Bisector: repair", "shared"),
    ("induce_text_only_a600", "Response text: induce", "text"),
    ("induce_vision_only_a600", "Vision shift: induce", "vision"),
    ("induce_shared_a600", "Bisector: induce", "shared"),
    ("random_1_a600", "Random direction 1", "random"),
    ("random_2_a600", "Random direction 2", "random"),
    ("random_3_a600", "Random direction 3", "random"),
]
COMPARISONS = [
    ("text repair - baseline", "Text repair - baseline", "repair_text_only_a600", "baseline", "text"),
    ("text induce - baseline", "Text induce - baseline", "induce_text_only_a600", "baseline", "text"),
    ("vision repair - baseline", "Vision repair - baseline", "repair_vision_only_a600", "baseline", "vision"),
    ("vision induce - baseline", "Vision induce - baseline", "induce_vision_only_a600", "baseline", "vision"),
    ("shared repair - baseline", "Bisector repair - baseline", "repair_shared_a600", "baseline", "shared"),
    (
        "shared - text",
        "Bisector repair - text repair",
        "repair_shared_a600",
        "repair_text_only_a600",
        "shared",
    ),
    (
        "shared - vision",
        "Bisector repair - vision repair",
        "repair_shared_a600",
        "repair_vision_only_a600",
        "shared",
    ),
]
COLORS = {
    "text": "#067D81",
    "vision": "#506F9F",
    "shared": "#A66827",
    "random": "#747C85",
    "baseline": "#222C38",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/revised")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def verified_data(path):
    """Reject missing, duplicated, mismatched, or inconsistent recorded rows."""
    wrapper = json.loads(path.read_text())
    data = wrapper["data"]
    expected_ids = set(data["protocol"]["confirm_ids"])
    expected_conditions = {row[0] for row in CONDITIONS}
    rows = [row for row in data["judgments"] if row["bank"] == "confirm"]
    keys = [(row["condition"], row["row_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicated confirm judgments")
    if set(keys) != {(condition, row_id) for condition in expected_conditions for row_id in expected_ids}:
        raise ValueError("Confirm judgments do not form the complete paired ten-condition design")
    lookup = {(row["condition"], row["row_id"]): row for row in rows}
    for row in rows:
        expected = row["safety"] == "unsafe" and not row["refusal"]
        if bool(row["unsafe_compliance"]) != expected:
            raise ValueError("Stored compliance indicator disagrees with safety/refusal labels")
    sources = Counter(lookup[("baseline", row_id)]["source"] for row_id in expected_ids)
    if len(sources) != 4 or len(set(sources.values())) != 1:
        raise ValueError("Expected four equally sized confirm sources")
    for row_id in expected_ids:
        if len({lookup[(condition, row_id)]["source"] for condition in expected_conditions}) != 1:
            raise ValueError("Source labels disagree across paired conditions")
    recorded_rates = {row["condition"]: row for row in data["asr_confirm"]}
    rates = []
    for condition, label, family in CONDITIONS:
        count = sum(int(lookup[(condition, row_id)]["unsafe_compliance"]) for row_id in expected_ids)
        rate = 100 * count / len(expected_ids)
        record = recorded_rates[condition]
        if record["n"] != len(expected_ids) or not math.isclose(
            rate, record["asr_qwen_unsafe_compliance"], abs_tol=1e-10
        ):
            raise ValueError(f"Recorded ASR disagrees with judgments for {condition}")
        rates.append(
            {
                "condition": condition,
                "label": label,
                "family": family,
                "unsafe_nonrefusal_count": count,
                "n": len(expected_ids),
                "asr_percent": rate,
            }
        )
    intervals = {row["comparison"]: row for row in data["component_table"]}
    contrasts = []
    for name, label, first, second, family in COMPARISONS:
        delta = (
            100
            * sum(
                int(lookup[(first, row_id)]["unsafe_compliance"])
                - int(lookup[(second, row_id)]["unsafe_compliance"])
                for row_id in expected_ids
            )
            / len(expected_ids)
        )
        saved = intervals[name]
        if saved["n"] != len(expected_ids) or not math.isclose(delta, saved["delta_pp"], abs_tol=1e-10):
            raise ValueError(f"Recorded paired delta disagrees with judgments for {name}")
        if not saved["ci_low"] <= delta <= saved["ci_high"]:
            raise ValueError(f"Unexpected archived interval for {name}")
        contrasts.append(
            {
                "comparison": name,
                "label": label,
                "family": family,
                "first_condition": first,
                "second_condition": second,
                "n": len(expected_ids),
                "delta_pp": delta,
                "ci_low_pp": saved["ci_low"],
                "ci_high_pp": saved["ci_high"],
                "interval_source": "archived source-stratified paired bootstrap; 20000 resamples",
            }
        )
    return wrapper, rates, contrasts, rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "text.color": "#25303C",
            "axes.labelcolor": "#25303C",
            "xtick.color": "#4E5864",
            "ytick.color": "#25303C",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#B8C0C8",
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(fig, output_dir, name, dpi):
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=dpi, facecolor="white")


def plot_component_results(plt, data, rates, contrasts, output_dir, dpi):
    """Keep absolute rates and paired-difference intervals on separate axes."""
    fig = plt.figure(figsize=(13.8, 7.1))
    left = fig.add_axes([0.18, 0.17, 0.285, 0.65])
    right = fig.add_axes([0.735, 0.17, 0.235, 0.65])
    n = rates[0]["n"]
    fig.text(0.04, 0.945, "Direction effects on held-out harmful prompts", fontsize=18, weight="bold")
    fig.text(
        0.04,
        0.902,
        f"Gemma 3-4B + LoRA  |  layer {data['protocol']['layer']}  |  "
        f"alpha = {data['protocol']['alpha']:g}  |  {n} paired examples, 12 per source",
        fontsize=11,
        color="#53606D",
    )
    fig.text(0.04, 0.848, "A   Unsafe compliance", fontsize=12, weight="bold")
    fig.text(0.51, 0.848, "B   Paired changes with archived 95% intervals", fontsize=12, weight="bold")

    ys = [11.2, 9.8, 8.8, 7.8, 6.3, 5.3, 4.3, 2.8, 1.8, 0.8]
    baseline = rates[0]["asr_percent"]
    left.axvline(baseline, color=COLORS["baseline"], lw=1, ls=(0, (3, 3)), alpha=0.5, zorder=1)
    for index, (y, row) in enumerate(zip(ys, rates)):
        color = COLORS[row["family"]]
        left.hlines(y, 0, row["asr_percent"], color=color, alpha=0.17, lw=2.2, zorder=1)
        marker = "^" if "induce" in row["condition"] else "o"
        if index == 0:
            marker = "s"
        left.scatter([row["asr_percent"]], [y], color=color, s=51, marker=marker, zorder=3)
        left.text(
            row["asr_percent"] + 3.0,
            y,
            f"{row['asr_percent']:.2f}%  ({row['unsafe_nonrefusal_count']}/{row['n']})",
            va="center",
            fontsize=10,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
    left.set_yticks(ys, [row["label"] for row in rates])
    left.set_ylim(0.1, 12.0)
    left.set_xlim(-2, 113)
    left.set_xticks([0, 25, 50, 75, 100])
    left.set_xlabel("Unsafe and non-refusal responses (%)", labelpad=12)
    left.tick_params(axis="y", length=0, pad=10)
    left.grid(axis="x", color="#E7EAED", linewidth=0.7)
    left.set_axisbelow(True)
    for boundary in (10.5, 7.0, 3.5):
        left.axhline(boundary, color="#E8EBEE", lw=0.7, zorder=0)

    delta_ys = [6.7, 5.7, 4.5, 3.5, 2.3, 1.1, 0.1]
    right.axvline(0, color="#616A75", lw=1.0, ls=(0, (3, 3)), zorder=1)
    for y, row in zip(delta_ys, contrasts):
        color = COLORS[row["family"]]
        low, high, center = row["ci_low_pp"], row["ci_high_pp"], row["delta_pp"]
        right.hlines(y, low, high, color=color, lw=1.6)
        right.vlines([low, high], y - 0.07, y + 0.07, color=color, lw=1.4)
        right.scatter([center], [y], color=color, s=40, zorder=3)
        right.text(
            center,
            y + 0.22,
            f"{center:+.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
    right.set_yticks(delta_ys, [row["label"] for row in contrasts])
    right.set_ylim(-0.65, 7.4)
    right.set_xlim(-65, 42)
    right.set_xticks([-60, -40, -20, 0, 20, 40])
    right.set_xlabel("Change in unsafe compliance\n(percentage points)", labelpad=9, fontsize=10)
    right.tick_params(axis="y", length=0, pad=10, labelsize=10)
    right.grid(axis="x", color="#E7EAED", linewidth=0.7)
    right.set_axisbelow(True)
    for boundary in (5.05, 2.9, 1.7):
        right.axhline(boundary, color="#E8EBEE", lw=0.7)
    fig.text(
        0.04,
        0.070,
        "All steering arms edit the final prefill text state and every cached decode state; "
        '"vision" names the direction\'s origin.',
        fontsize=10,
        color="#4D5865",
    )
    fig.text(
        0.04,
        0.037,
        "Panel A: empirical rates, no error bars.  Panel B: paired-difference CIs; "
        "intervals spanning zero do not establish equivalence.",
        fontsize=10,
        color="#4D5865",
    )
    save_figure(fig, output_dir, "component_effects", dpi)
    plt.close(fig)


def plot_direction_schematic(plt, data, output_dir, dpi):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    geometry = {row["cosine"]: row["value"] for row in data["geometry"]}
    actual_cos = geometry["response text vs aligned vision FT shift"]
    shift_cos = geometry["paired FT-minus-base text/vision"]
    projected_text = data["protocol"]["alpha"] * geometry["bisector vs response text"]
    fig = plt.figure(figsize=(12.4, 6.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, "Direction construction and the common intervention site", fontsize=18, weight="bold")
    ax.text(
        0.04, 0.89, "Definitions from the preserved layer-13 component study", fontsize=11, color="#53606D"
    )

    def box(x, y, width, height, facecolor, edgecolor):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.010",
            linewidth=1,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(patch)

    box(0.05, 0.655, 0.435, 0.17, "#EEF7F6", "#93C2BE")
    ax.text(0.065, 0.790, "Response-text direction  $c_R$", fontsize=12, weight="bold", color=COLORS["text"])
    ax.text(0.065, 0.752, "Faces: harmful vs. safe teacher-forced responses", fontsize=10.5)
    ax.text(
        0.065,
        0.704,
        r"$c_R = \mathrm{unit}\!\left(\overline{h}_{\mathrm{ft,\ harmful\ response}} - "
        r"\overline{h}_{\mathrm{ft,\ safe\ response}}\right)$",
        fontsize=12,
    )

    box(0.05, 0.403, 0.435, 0.18, "#F0F3F8", "#ACBCD2")
    ax.text(
        0.065,
        0.547,
        "Vision model-shift direction  $c_V$",
        fontsize=12,
        weight="bold",
        color=COLORS["vision"],
    )
    ax.text(0.065, 0.509, "VLGuard: fine-tuned vs. base; same image inputs", fontsize=10.5)
    ax.text(
        0.065,
        0.460,
        r"$c_V = \mathrm{unit}\!\left(\overline{h}_{\mathrm{ft,\ image\ tokens}} - "
        r"\overline{h}_{\mathrm{base,\ image\ tokens}}\right)$",
        fontsize=12,
    )
    ax.text(
        0.065, 0.426, "Image-token pooling is inside the language decoder.", fontsize=9.5, color="#526277"
    )
    ax.text(
        0.067,
        0.612,
        f"Sign-aligned cosine of the component directions: {actual_cos:.3f}",
        fontsize=10.5,
        weight="bold",
    )

    box(0.05, 0.15, 0.435, 0.175, "#FAF4EC", "#D4B997")
    ax.text(0.065, 0.287, "Unit bisector  $c_B$", fontsize=12, weight="bold", color=COLORS["shared"])
    ax.text(0.065, 0.244, r"$c_B = \mathrm{unit}(c_R + s\,c_V)$", fontsize=13)
    ax.text(0.065, 0.207, "s aligns the vision shift's sign with the response direction.", fontsize=9.5)
    ax.text(
        0.065, 0.171, f"At alpha = 600: projection onto $c_R$ is {projected_text:.0f}, not 600.", fontsize=10
    )

    box(0.60, 0.403, 0.34, 0.422, "#FAFBFC", "#C5CDD5")
    ax.text(0.62, 0.787, "Same site for all steering arms", fontsize=12, weight="bold")
    ax.text(0.62, 0.739, r"$h' = h \pm 600\,c$", fontsize=22)
    ax.text(0.62, 0.689, r"Repair: $-600c$     Induction: $+600c$", fontsize=11)
    ax.text(0.62, 0.660, "Layer 13  |  Baseline: no residual edit", fontsize=9, color="#607080")
    ax.text(0.62, 0.637, "PREFILL", fontsize=9, weight="bold", color="#607080")
    for i in range(3):
        ax.add_patch(
            Rectangle((0.624 + i * 0.043, 0.579), 0.033, 0.035, facecolor="#DCE5F0", edgecolor="none")
        )
    ax.text(0.767, 0.591, "...", fontsize=13)
    ax.add_patch(
        Rectangle((0.801, 0.579), 0.080, 0.035, facecolor="#BBD8C7", edgecolor="#4D8060", linewidth=1)
    )
    ax.text(0.836, 0.594, r"$\Delta h$", ha="center", va="center", fontsize=10)
    ax.text(0.623, 0.552, "image tokens", fontsize=9, color="#607080")
    ax.text(0.787, 0.552, "final text state", fontsize=9, color="#41634F")
    ax.text(0.62, 0.505, "DECODE", fontsize=9, weight="bold", color="#607080")
    for i in range(4):
        x = 0.622 + i * 0.069
        ax.add_patch(
            Rectangle((x, 0.447), 0.047, 0.033, facecolor="#BBD8C7", edgecolor="#4D8060", linewidth=0.6)
        )
        ax.text(x + 0.0235, 0.463, r"$\Delta h$", ha="center", va="center", fontsize=10)
        if i < 3:
            ax.annotate(
                "",
                (x + 0.068, 0.463),
                (x + 0.048, 0.463),
                arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#6B7B71"},
            )
    ax.text(0.622, 0.420, "Every cached decode state is edited.", fontsize=9.5)
    for y in (0.74, 0.49, 0.24):
        ax.add_patch(
            FancyArrowPatch(
                (0.499, y),
                (0.588, 0.692),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1,
                color="#7F8A95",
                connectionstyle="arc3,rad=0.0",
            )
        )

    box(0.60, 0.15, 0.34, 0.175, "#FFFFFF", "#C9CFD6")
    ax.text(0.617, 0.286, "Separate model-shift geometry", fontsize=11, weight="bold")
    ax.text(0.617, 0.246, f"Cosine = {shift_cos:.3f}", fontsize=15, weight="bold")
    ax.text(0.617, 0.210, "FT-minus-base text shift vs. FT-minus-base vision shift.", fontsize=9)
    ax.text(0.617, 0.176, "This pair does not use the response-text direction $c_R$.", fontsize=9)
    ax.text(
        0.05,
        0.075,
        "Equal alpha matches total intervention norm. Comparing bisector and text arms changes "
        "both the vision component and the text projection.",
        fontsize=10,
        color="#52606D",
    )
    save_figure(fig, output_dir, "direction_and_site", dpi)
    plt.close(fig)


def write_captions(output_dir, data):
    text = """# Proposed replacement figures

These figures replot preserved upstream results. They are proposed replacements
for the submitted manuscript's Figure 1 and Table 2, not additional experiments.
PNG previews and editable SVGs share the same content; CSVs expose every number.

## component_effects

Component interventions on 48 paired held-out VLGuard examples, with 12 examples
from each of four sources. Unsafe compliance is Qwen3Guard classification as
unsafe and not a refusal. (A) All ten recorded confirm conditions, with observed
counts and rates. Repair subtracts a unit direction scaled by alpha=600; induction
adds it. Three random controls are orthogonal to the plane of the response-text
and vision-shift directions. These are empirical rates without error bars.
(B) Archived source-stratified 95% bootstrap intervals over 20,000 resamples for
seven paired differences. These intervals belong to differences, not individual
ASRs. Rates and mean differences were checked against all 480 saved confirm
judgments. Intervals spanning zero do not establish equivalence.

Every steering arm in this artifact edits the final attended prefill text state
and each cached decode state at layer 13; baseline has no edit. The vision label describes a direction estimated
from image-token model shifts, not an intervention on native image-token states.
The manuscript's additional native-image row is not present in this component
artifact and is deliberately not merged into this paired plot. The bisector and
text arms have equal total norm but different text projections; their comparison
does not isolate an independently controlled vision contribution. The preserved
paired bisector-minus-text difference is +14.58 pp [6.25, 25.00].

## direction_and_site

Direction definitions and the shared application site for the preserved component
study. The response-text direction contrasts harmful and safe teacher-forced
Faces completions under the fine-tuned model. The vision-shift direction contrasts
fine-tuned and base activations at image-placeholder positions on the same
VLGuard inputs. Their sign-aligned cosine is 0.407. The unit bisector has cosine
0.839 with each; at alpha=600 its projection on the response-text direction is
approximately 503. The separate 0.802 geometric result compares FT-minus-base
text and vision shifts. It is not the cosine between the two component directions
used for the response-text/vision-shift bisector. The schematic does not infer
that the vision pathway is causally inactive.

## Reproduction and provenance

Run `python -m experiments.analysis.paper_figures --output-dir figures/revised`
from the repository root with matplotlib installed. The script uses the extracted
JSON for `step3_component_resolved_complete.pt`, recomputes all ten rates and seven
paired means, checks them against stored summaries, and plots the archived CIs
without changing their computation. `source_manifest.json` records the upstream
artifact identity, input JSON hash, and verification scope. No model, judge, or
bootstrap rerun is needed. Confidence limits for shared induction or individual
random-control differences were not stored in this component table and are not
invented here.
"""
    (output_dir / "CAPTIONS.md").write_text(text)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    wrapper, rates, contrasts, judgments = verified_data(args.input)
    data = wrapper["data"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "confirm_rates.csv", rates)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(args.output_dir / "geometry.csv", data["geometry"])
    indicator_rows = [
        {key: row[key] for key in ("condition", "row_id", "source", "safety", "refusal", "unsafe_compliance")}
        for row in judgments
    ]
    write_csv(args.output_dir / "confirm_indicators.csv", indicator_rows)
    provenance = {
        "source_artifact": wrapper["source"],
        "input_json_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "status": "archived results replotted; no new experiments",
        "verification": {
            "confirm_judgments": len(judgments),
            "conditions": len(rates),
            "paired_deltas_checked": len(contrasts),
            "ci_method": "archived intervals, not recomputed",
        },
        "protocol": data["protocol"],
    }
    (args.output_dir / "source_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt = configure_matplotlib()
    plot_component_results(plt, data, rates, contrasts, args.output_dir, args.dpi)
    plot_direction_schematic(plt, data, args.output_dir, args.dpi)
    write_captions(args.output_dir, data)
    print(
        f"Verified {len(judgments)} confirm judgments; saved two PNG/SVG figures and four data CSVs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
