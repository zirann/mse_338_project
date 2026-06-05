#!/usr/bin/env python3
"""Generate the report's data figures and a reproducible stats artifact.

All numbers are recomputed from a results bundle (per-arm/seed `metrics.json`
and `eval_responses.jsonl`); nothing is hardcoded. The script is standalone:
the uncertainty / hedge lexicons and density functions are vendored below so it
runs from anywhere without installing the project package.

Outputs written to --out_dir:
  fig_uncertainty_bars.png     uncertainty score, mean +/- SE across arms
  fig_calibration_bars.png     overconfident% + wrong-but-hedged% across arms
  report_stats.json            per-arm table, calibration cross-tab, paired
                               p-values (decoupled vs others), example excerpts

The two conceptual diagrams (fig_pipeline_schematic.png,
fig_support_coupling.png) are not derived from results. They are treated as
static assets: the script regenerates a plain fallback only if the file is
absent, so hand-edited versions are never overwritten.

Usage (from the repo root, or from submission_ready/):
  python make_report_figures.py --bundle results --out_dir paper/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

# --- vendored metrics (kept identical to src/complexity_theater) -------------
HEDGE_PHRASES = [
    "it's worth noting", "it is worth noting", "generally speaking", "in some cases",
    "in most cases", "however", "one might argue", "arguably", "to some extent",
    "broadly speaking",
    "however, it should be acknowledged", "that said", "with that said",
    "having said that", "on the flip side", "it should be noted",
    "with the caveat that", "all that said",
    "on one hand", "on the other hand", "this depends on", "it varies",
    "context matters", "the answer is nuanced", "it's complicated",
]
EXPLICIT_UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure", "not certain",
    "not sure", "it's unclear", "it is unclear", "unclear", "uncertain",
    "hard to say", "i'm unsure", "possibly", "perhaps", "may ", "might ", "could be",
]
UNCERTAINTY_PHRASES = HEDGE_PHRASES + EXPLICIT_UNCERTAINTY_PHRASES


def _tokens(text):
    return text.split() if text else []


def length(text):
    return len(_tokens(text))


def _count(text, phrases):
    if not text:
        return 0
    low = text.lower()
    return sum(low.count(p.lower()) for p in phrases if p)


def density(text, phrases):
    n = length(text)
    return 100.0 * _count(text, phrases) / n if n else 0.0


def uncertainty_score(text):
    return density(text, UNCERTAINTY_PHRASES)


def hedge_density(text):
    return density(text, HEDGE_PHRASES)


# --- arms --------------------------------------------------------------------
ARMS = ["baseline", "vanilla_dpo", "sampo_dpo", "dpop", "sampo_dpop", "sampo_dpop_decoupled"]
ARM_LABEL = {
    "baseline": "baseline", "vanilla_dpo": "vanilla\nDPO", "sampo_dpo": "SamPO",
    "dpop": "DPOP", "sampo_dpop": "naive\nSamPO+DPOP",
    "sampo_dpop_decoupled": "decoupled\nSamPO+DPOP",
}
EXPLICIT = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure", "not sure",
    "no comment", "unclear", "i'm unsure", "hard to say", "can't be sure",
    "cannot be sure", "not certain",
]


def seed_dirs(bundle, arm):
    if arm == "baseline":
        d = bundle / "baseline"
        return [d] if d.exists() else []
    return [Path(p) for p in sorted(glob.glob(str(bundle / arm / "seed*")))]


def _mean_se(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), 0.0
    m = sum(xs) / len(xs)
    se = (st.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return m, se


def _fac_label(f):
    f = 0.0 if f is None else float(f)
    return "CORRECT" if f >= 1 else ("INCORRECT" if f <= 0 else "PARTIAL")


def _conf_label(unc, resp):
    t = resp.lower()
    if unc >= 1.21 or any(p in t for p in EXPLICIT):
        return "LOW"
    if unc < 0.2:
        return "HIGH"
    return "MEDIUM"


def collect(bundle):
    per_arm = {}
    for arm in ARMS:
        dirs = seed_dirs(bundle, arm)
        seed_metrics = [json.loads((d / "metrics.json").read_text())
                        for d in dirs if (d / "metrics.json").exists()]
        agg = {"n_seeds": len(seed_metrics)}
        for k in ("uncertainty_score", "hedge_density", "confidence_marker_density",
                  "factuality", "judge_win_rate_vs_round_0", "length"):
            m, se = _mean_se([x.get(k) for x in seed_metrics])
            agg[f"{k}_mean"], agg[f"{k}_se"] = m, se
        labels = []
        for d in dirs:
            ep = d / "eval_responses.jsonl"
            if not ep.exists():
                continue
            for line in ep.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                resp = r.get("response", "")
                labels.append((_fac_label(r.get("factuality")),
                               _conf_label(uncertainty_score(resp), resp)))
        n = len(labels)
        over = sum(1 for fl, cl in labels if fl == "INCORRECT" and cl == "HIGH")
        inc = [(fl, cl) for fl, cl in labels if fl == "INCORRECT"]
        inc_low = sum(1 for fl, cl in inc if cl == "LOW")
        agg["n_eval_pooled"] = n
        agg["overconfident_pct"] = 100.0 * over / n if n else float("nan")
        agg["wrong_but_hedged_pct"] = 100.0 * inc_low / len(inc) if inc else float("nan")
        per_arm[arm] = agg
    return per_arm


def _by_prompt(bundle, arm, fn):
    acc = {}
    for d in seed_dirs(bundle, arm):
        ep = d / "eval_responses.jsonl"
        if not ep.exists():
            continue
        for line in ep.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            acc.setdefault(r["prompt_id"], []).append(fn(r))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _normal_p(z):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def paired(bundle, a1, a2, fn):
    A, B = _by_prompt(bundle, a1, fn), _by_prompt(bundle, a2, fn)
    ids = [i for i in A if i in B]
    d = [A[i] - B[i] for i in ids]
    if len(d) < 2:
        return None
    md, sd = sum(d) / len(d), st.stdev(d)
    se = sd / math.sqrt(len(d))
    t = md / se if se > 0 else 0.0
    return {"n": len(d), "mean": md, "se": se, "t": t, "p": _normal_p(t),
            "cohen_d": md / sd if sd > 0 else 0.0}


def example_excerpts(bundle, prompt_id):
    out = {}
    for arm in ("baseline", "vanilla_dpo", "sampo_dpo", "sampo_dpop_decoupled"):
        d = bundle / "baseline" if arm == "baseline" else bundle / arm / "seed0"
        ep = d / "eval_responses.jsonl"
        if not ep.exists():
            continue
        for line in ep.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt_id") == prompt_id:
                txt = " ".join(r["response"].replace("<think>", " ").replace("</think>", " ").split())
                out[arm] = {
                    "excerpt": txt[:300],
                    "uncertainty_score": round(uncertainty_score(r["response"]), 3),
                    "hedge_density": round(hedge_density(r["response"]), 3),
                    "factuality": r.get("factuality"),
                    "correct_reference": r.get("correct_reference"),
                    "question": r.get("question"),
                }
                break
    return out


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                         "savefig.bbox": "tight", "font.size": 10})
    return plt


def fig_uncertainty(plt, per_arm, out):
    arms = [a for a in ARMS if not math.isnan(per_arm[a]["uncertainty_score_mean"])]
    means = [per_arm[a]["uncertainty_score_mean"] for a in arms]
    ses = [per_arm[a]["uncertainty_score_se"] for a in arms]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = range(len(arms))
    ax.bar(xs, means, yerr=ses, capsize=4, color="#4C72B0", alpha=0.9)
    ax.axhline(per_arm["baseline"]["uncertainty_score_mean"], ls="--", color="grey",
               lw=1, label="baseline")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=8)
    ax.set_ylabel("uncertainty score (markers / 100 tok)")
    ax.set_title("Uncertainty signaling across arms (mean +/- SE over 3 seeds)")
    ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_calibration(plt, per_arm, out):
    import numpy as np
    over = [per_arm[a]["overconfident_pct"] for a in ARMS]
    wbh = [per_arm[a]["wrong_but_hedged_pct"] for a in ARMS]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x, w = np.arange(len(ARMS)), 0.38
    ax.bar(x - w / 2, over, w, label="overconfident error %", color="#C44E52")
    ax.bar(x + w / 2, wbh, w, label="wrong-but-hedged %", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=8)
    ax.set_ylabel("% of responses")
    ax.set_title("Calibration on incorrect answers across arms")
    ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fallback_pipeline(plt, out):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAEAF2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec="black", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    box(0.02, 0.55, 0.18, 0.3, "Vanilla DPO")
    box(0.27, 0.55, 0.22, 0.3, "uncertainty\nsuppression")
    box(0.56, 0.55, 0.22, 0.3, "overconfident\nerrors")
    ax.annotate("", xy=(0.27, 0.70), xytext=(0.20, 0.70), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.56, 0.70), xytext=(0.49, 0.70), arrowprops=dict(arrowstyle="->"))
    for i, m in enumerate(["SamPO", "DPOP", "naive SamPO+DPOP", "decoupled (ours)"]):
        box(0.02 + i * 0.245, 0.08, 0.22, 0.3, m, fc="#DCE6F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out)
    plt.close(fig)


def fallback_support(plt, out):
    from matplotlib.patches import FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))

    def panel(ax, title, preserve_full):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        for i in range(10):
            fc = "#4C72B0" if i in (1, 3, 4, 7) else "#CCCCCC"
            ax.add_patch(FancyBboxPatch((0.05 + i * 0.088, 0.62), 0.075, 0.12,
                                        boxstyle="round,pad=0.005", fc=fc, ec="black", lw=0.5))
        ax.text(0.5, 0.80, "chosen tokens (blue = subset S)", ha="center", fontsize=7.5)
        if preserve_full:
            ax.add_patch(FancyBboxPatch((0.04, 0.30), 0.92, 0.12, boxstyle="round,pad=0.005",
                                        fc="none", ec="#C44E52", lw=1.5))
            ax.text(0.5, 0.20, "DPOP anchor on full C", ha="center", fontsize=8, color="#C44E52")
        else:
            for i in (1, 3, 4, 7):
                ax.add_patch(FancyBboxPatch((0.05 + i * 0.088, 0.30), 0.075, 0.12,
                                            boxstyle="round,pad=0.005", fc="none",
                                            ec="#C44E52", lw=1.5))
            ax.text(0.5, 0.20, "DPOP anchor on subset S", ha="center", fontsize=8, color="#C44E52")

    panel(axes[0], "Naive: shared subset S", False)
    panel(axes[1], "Decoupled (ours): full C", True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default=str(here.parent / "mse338_results_bundle" / "outputs"),
                   help="results bundle with baseline/ and <arm>/seed*/ subdirs")
    p.add_argument("--out_dir", default=str(here.parent / "mse338_report_template_update"),
                   help="where figures + report_stats.json are written")
    args = p.parse_args()
    bundle, out_dir = Path(args.bundle), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_arm = collect(bundle)
    fns = {"uncertainty_score": lambda r: uncertainty_score(r["response"]),
           "hedge_density": lambda r: hedge_density(r["response"]),
           "length": lambda r: length(r["response"])}
    paired_stats = {ref: {m: paired(bundle, "sampo_dpop_decoupled", ref, fn)
                          for m, fn in fns.items()}
                    for ref in ("sampo_dpop", "dpop", "sampo_dpo", "vanilla_dpo", "baseline")}
    stats = {
        "source": str(bundle),
        "arms": per_arm,
        "paired_vs_decoupled": paired_stats,
        "medical_example": example_excerpts(bundle, "Health_86"),
        "calibration_rules": {
            "overconfident": "factual_label==INCORRECT and confidence_label==HIGH",
            "confidence_HIGH": "uncertainty_score<0.2",
            "confidence_LOW": "uncertainty_score>=1.21 or explicit-uncertainty phrase",
            "factual_label": "factuality 1.0->CORRECT, 0.5->PARTIAL, 0.0->INCORRECT",
        },
    }
    (out_dir / "report_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[report] wrote {out_dir / 'report_stats.json'}")

    try:
        plt = _setup_mpl()
    except Exception as e:
        print(f"[report] matplotlib unavailable ({e}); figures skipped")
        return
    fig_uncertainty(plt, per_arm, out_dir / "fig_uncertainty_bars.png")
    fig_calibration(plt, per_arm, out_dir / "fig_calibration_bars.png")
    print(f"[report] wrote fig_uncertainty_bars.png, fig_calibration_bars.png to {out_dir}")
    for name, gen in (("fig_pipeline_schematic.png", fallback_pipeline),
                      ("fig_support_coupling.png", fallback_support)):
        target = out_dir / name
        if target.exists():
            print(f"[report] kept existing {name} (conceptual diagram preserved)")
        else:
            gen(plt, target)
            print(f"[report] wrote fallback {name}")

    print("\n[report] per-arm summary:")
    for a in ARMS:
        d = per_arm[a]
        print(f"  {a:22s} unc={d['uncertainty_score_mean']:.3f}+-{d['uncertainty_score_se']:.3f} "
              f"overconf={d['overconfident_pct']:.1f}% wrong_hedged={d['wrong_but_hedged_pct']:.1f}% "
              f"fact={d['factuality_mean']:.3f} win={d['judge_win_rate_vs_round_0_mean']:.3f}")


if __name__ == "__main__":
    main()
