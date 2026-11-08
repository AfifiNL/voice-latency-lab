"""voice-lab: language policy, latency simulation and comparison, TTFT recording."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from voice_lab import harness, ttft
from voice_lab.language import LanguagePolicy, PolicyConfig, detect


def _detect(args: argparse.Namespace) -> int:
    for text in args.text:
        d = detect(text)
        print(f"{d.language or '-':>3}  conf={d.confidence:.2f}  {d.reason}  | {text}")
    return 0


def _converse(args: argparse.Namespace) -> int:
    policy = LanguagePolicy(PolicyConfig(switch_after=args.switch_after))
    lines = Path(args.file).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(filter(None, lines), 1):
        d = policy.observe(line)
        flag = "SWITCH" if d.switched else "      "
        print(f"{i:>2} reply in {d.reply_language} {flag} {d.reason:<48} | {line}")
    return 0


def _sim_config(args: argparse.Namespace) -> harness.SimConfig:
    cfg = harness.SimConfig(speculation=not args.no_speculation)
    if args.ttft_from:
        rows = ttft.read_rows(Path(args.ttft_from))
        if args.ttft_config:
            rows = [r for r in rows if r["config"] == args.ttft_config]
        # Same rows the TTFT percentiles use: no failed calls, no cold (first) call.
        rows = [r for r in rows if r.get("ttft_ms") is not None and not r.get("cold")]
        if not rows:
            raise SystemExit("no TTFT samples matched")
        cfg = replace(cfg, ttft_samples_ms=tuple(float(r["ttft_ms"]) for r in rows))
    return cfg


def _simulate(args: argparse.Namespace) -> int:
    records = harness.simulate(_sim_config(args), args.turns, args.seed)
    harness.write_jsonl(Path(args.out), records)
    print(json.dumps(harness.summarize(records), indent=2))
    return 0


def _summary(args: argparse.Namespace) -> int:
    print(json.dumps(harness.summarize(harness.read_jsonl(Path(args.file))), indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    base = harness.summarize(harness.read_jsonl(Path(args.baseline)))
    cur = harness.summarize(harness.read_jsonl(Path(args.current)))
    t = harness.Thresholds(
        p50_ms=args.p50_ms,
        p95_ms=args.p95_ms,
        pct=args.max_pct / 100,
        wasted_rate=args.max_wasted_rate,
    )
    note = "" if base["turns"] == cur["turns"] else "  (different turn counts: rates compared)"
    print(f"{'turns':>16}  {base['turns']} -> {cur['turns']}{note}")
    print(f"{'wasted per turn':>16}  {base['wasted_rate']:.3f} -> {cur['wasted_rate']:.3f}")
    for m in (*t.metrics, "turn_end_ms"):
        print(
            f"{m:>16}  p50 {base[m]['p50']:>8.1f} -> {cur[m]['p50']:>8.1f}   "
            f"p95 {base[m]['p95']:>8.1f} -> {cur[m]['p95']:>8.1f}"
        )
    problems = harness.compare(base, cur, t)
    if problems:
        print("REGRESSION:\n  " + "\n  ".join(problems))
        return 1
    print("no regression")
    return 0


def _ttft_report(args: argparse.Namespace) -> int:
    review = ttft.load_review(Path(args.review)) if args.review else None
    summary = ttft.summarize(ttft.read_rows(Path(args.file)), review)
    print(
        f"{'config/language':<44} {'calls':>5} {'ttft p50':>9} {'ttft p95':>9} "
        f"{'total p50':>10} {'right lang':>10}" + (f" {'flagged':>11}" if review else "")
    )
    for name, s in summary.items():
        line = (
            f"{name:<44} {s['calls']:>5} {s['ttft_p50_ms']:>9.1f} {s['ttft_p95_ms']:>9.1f} "
            f"{s['total_p50_ms']:>10.1f} {s['reply_in_requested_language']:>7}/{s['calls']:<2}"
        )
        if review:
            line += f" {s['flagged_in_review']:>8}/{s['calls']}"
        print(line)
    if args.check:
        expected = json.loads(Path(args.check).read_text(encoding="utf-8"))
        if expected != summary:
            print(
                f"MISMATCH: {args.check} does not match the recording; regenerate it with --write"
            )
            return 1
        print(f"matches {args.check}")
    if args.write:
        Path(args.write).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


def _record_ttft(args: argparse.Namespace) -> int:  # pragma: no cover - live calls
    from voice_lab.llm import GeminiStreamingLLM

    llm = GeminiStreamingLLM(
        os.environ.get("VOICE_LAB_GEMINI_MODEL", "gemini-3.6-flash"),
        max_calls=int(os.environ.get("VOICE_LAB_MAX_LIVE_CALLS", "0") or 0),
        vertex=os.environ.get("VOICE_LAB_GEMINI_VERTEX", "").lower() in {"1", "true", "yes"},
        location=os.environ.get("VOICE_LAB_GEMINI_LOCATION") or None,
        thinking_level=args.thinking or None,
    )
    utterances = ttft.load_utterances(Path(args.utterances))[: args.limit or None]
    rows = asyncio.run(ttft.record(llm, utterances, llm.name))
    out = Path(args.out)
    previous = ttft.read_rows(out) if out.exists() else []
    kept = [r for r in previous if r["config"] != llm.name]
    ttft.write_rows(out, kept + rows)
    print(f"recorded {len(rows)} calls ({llm.budget.used} used) as {llm.name} to {out}")
    return 0


def _demo(_: argparse.Namespace) -> int:
    from voice_lab.demo import run_demo

    return run_demo()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voice-lab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("detect", help="detect the language of each text")
    s.add_argument("text", nargs="+")
    s.set_defaults(fn=_detect)
    s = sub.add_parser("converse", help="run the language policy over a file of user replies")
    s.add_argument("file")
    s.add_argument("--switch-after", type=int, default=2)
    s.set_defaults(fn=_converse)
    s = sub.add_parser("simulate", help="seeded latency simulation -> JSONL turn records")
    s.add_argument("--turns", type=int, default=300)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--no-speculation", action="store_true")
    s.add_argument("--ttft-from", help="bootstrap LLM TTFT from a recorded TTFT file")
    s.add_argument("--ttft-config", help="only samples recorded with this model config")
    s.add_argument("--out", default="reports/sim.jsonl")
    s.set_defaults(fn=_simulate)
    s = sub.add_parser("summary", help="p50/p95 of a turn-record file")
    s.add_argument("file")
    s.set_defaults(fn=_summary)
    s = sub.add_parser("compare", help="exit 1 if CURRENT is slower than BASELINE")
    s.add_argument("baseline")
    s.add_argument("current")
    d = harness.Thresholds()
    s.add_argument("--p50-ms", type=float, default=d.p50_ms, help="allowed p50 growth (ms)")
    s.add_argument("--p95-ms", type=float, default=d.p95_ms, help="allowed p95 growth (ms)")
    s.add_argument("--max-pct", type=float, default=d.pct * 100, help="or this %% of baseline")
    s.add_argument(
        "--max-wasted-rate",
        type=float,
        default=d.wasted_rate,
        help="allowed growth in wasted calls per turn",
    )
    s.set_defaults(fn=_compare)
    s = sub.add_parser("ttft-report", help="summarise a recorded TTFT file")
    s.add_argument("file")
    s.add_argument("--review", help="hand-labelled review of the replies (JSON)")
    s.add_argument("--check", help="fail unless the summary equals this JSON file")
    s.add_argument("--write", help="write the summary to this JSON file")
    s.set_defaults(fn=_ttft_report)
    s = sub.add_parser("record-ttft", help="LIVE: stream real model calls and record TTFT")
    s.add_argument("--utterances", default="data/utterances.jsonl")
    s.add_argument("--out", default="fixtures/live-ttft.jsonl")
    s.add_argument("--thinking", default="")
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=_record_ttft)
    s = sub.add_parser("demo", help="offline demo: policy, speculation, latency comparison")
    s.set_defaults(fn=_demo)
    args = p.parse_args(argv)
    code: int = args.fn(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
