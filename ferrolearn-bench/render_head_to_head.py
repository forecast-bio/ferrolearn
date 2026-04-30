#!/usr/bin/env python3
"""Render the head_to_head.json output as a Markdown comparison table."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def fmt_time(us):
    if us is None:
        return "—"
    if us < 1:
        return f"{us * 1000:.0f} ns"
    if us < 1000:
        return f"{us:.1f} us"
    if us < 1_000_000:
        return f"{us / 1000:.1f} ms"
    return f"{us / 1e6:.2f} s"


def fmt_speedup(sk, fl):
    if sk is None or fl is None or fl <= 0:
        return "—"
    s = sk / fl
    return f"**{s:.1f}x**" if s >= 2 else f"{s:.2f}x"


def fmt_score(s, metric):
    if s is None:
        return "—"
    if metric == "accuracy":
        return f"{s * 100:.2f}%"
    if metric in ("recon_rel", "rel_diff_vs_sklearn"):
        return f"{s:.3e}"
    return f"{s:.4f}"


def fmt_score_diff(fl, sk, metric):
    if fl is None or sk is None:
        return "—"
    if metric in ("r2", "accuracy", "ari"):
        d = fl - sk
        sign = "+" if d >= 0 else ""
        if metric == "accuracy":
            return f"{sign}{d * 100:.2f}pp"
        return f"{sign}{d:.4f}"
    if metric == "recon_rel":
        if sk <= 0:
            return "—"
        return f"{fl / sk:.2f}x"
    return "—"


def render_family(family: str, rows: list[dict]) -> str:
    out = [f"### {family.title()} — {len(rows)} comparisons\n"]
    out.append(
        "| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup "
        "| sklearn predict | ferrolearn predict | predict speedup "
        "| metric | sklearn | ferrolearn | Δ |"
    )
    out.append("|" + "|".join(["---"] * 12) + "|")

    for r in sorted(rows, key=lambda r: (r["algorithm"], r["dataset"])):
        sk = r["sklearn"]
        fl = r["ferrolearn"]
        m = r["metric"]
        out.append(
            f"| {r['algorithm']} | {r['dataset']} | "
            f"{fmt_time(sk['fit_us'])} | {fmt_time(fl['fit_us'])} | "
            f"{fmt_speedup(sk['fit_us'], fl['fit_us'])} | "
            f"{fmt_time(sk['predict_us'])} | {fmt_time(fl['predict_us'])} | "
            f"{fmt_speedup(sk['predict_us'], fl['predict_us'])} | "
            f"{m} | {fmt_score(sk['score'], m)} | {fmt_score(fl['score'], m)} | "
            f"{fmt_score_diff(fl['score'], sk['score'], m)} |"
        )
    out.append("")
    return "\n".join(out)


def render_summary(records: list[dict]) -> str:
    by_family = defaultdict(list)
    for r in records:
        by_family[r["family"]].append(r)

    out = ["## Summary — geometric mean speedup\n"]
    out.append("| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |")
    out.append("|---|---:|---:|---:|---:|")

    for family in sorted(by_family):
        rows = by_family[family]
        fit_ratios = []
        pred_ratios = []
        deltas = []
        for r in rows:
            sk, fl = r["sklearn"], r["ferrolearn"]
            if sk["fit_us"] and fl["fit_us"] and fl["fit_us"] > 0:
                fit_ratios.append(sk["fit_us"] / fl["fit_us"])
            if sk["predict_us"] and fl["predict_us"] and fl["predict_us"] > 0:
                pred_ratios.append(sk["predict_us"] / fl["predict_us"])
            if (r["metric"] in ("r2", "accuracy", "ari")
                    and fl["score"] is not None and sk["score"] is not None):
                deltas.append(fl["score"] - sk["score"])

        def gm(xs):
            return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None

        fg = gm(fit_ratios)
        pg = gm(pred_ratios)
        md = sum(deltas) / len(deltas) if deltas else None

        out.append(
            f"| {family} | {len(rows)} | "
            f"{fg:.2f}x | " if fg is not None else f"| {family} | {len(rows)} | — | "
        )
        # Recompose the row cleanly
        cells = [
            f"| {family}", str(len(rows)),
            (f"{fg:.2f}x" if fg is not None else "—"),
            (f"{pg:.2f}x" if pg is not None else "—"),
            (f"{md:+.4f}" if md is not None else "—"),
        ]
        out[-1] = " | ".join(cells) + " |"
    return "\n".join(out)


def main():
    if len(sys.argv) != 2:
        print("Usage: render_head_to_head.py <head_to_head.json>", file=sys.stderr)
        sys.exit(1)
    records = json.loads(Path(sys.argv[1]).read_text())

    by_family = defaultdict(list)
    for r in records:
        by_family[r["family"]].append(r)

    print("# ferrolearn vs scikit-learn — head-to-head report\n")
    print("Each row is a single (algorithm, dataset) head-to-head: same canonical")
    print("dataset (sklearn `make_*`), same train/test split, same hyperparameters,")
    print("same quality metric, both libraries fit + predict in the same Python")
    print("process. Δ is `ferrolearn − sklearn` for the quality metric (positive")
    print("means ferrolearn is more accurate; for `recon_rel`, lower is better and")
    print("the cell shows `ferrolearn / sklearn` ratio).\n")

    for family in sorted(by_family):
        print(render_family(family, by_family[family]))

    print(render_summary(records))
    print()


if __name__ == "__main__":
    main()
