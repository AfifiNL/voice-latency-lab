"""Measure the language detector and the request patterns on labelled corpora, and gate them.

Detector corpus (`data/detector_corpus.jsonl`): each item has the expected language, or
`null` when the right answer is "undetermined" (other languages, balanced mixes, fillers,
numbers, names). Per expected class: correct, undetermined, wrong. For a language, *wrong* (a
different language) is the costly error; *undetermined* only keeps the current reply language.

Request corpus (`data/requests.jsonl`): explicit requests with their target language, and near
misses that must not be read as requests (expected `null`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voice_lab.language import detect
from voice_lab.switch_requests import find_request

CLASSES = ("en", "nl", "ar", "none")


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def detector_scores(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cls in CLASSES:
        items = [r for r in rows if (r["expected"] or "none") == cls]
        got = [detect(r["text"]).language for r in items]
        correct = sum(1 for r, g in zip(items, got, strict=True) if g == r["expected"])
        undetermined = sum(1 for g in got if g is None)
        if cls == "none":
            correct = undetermined
        wrong = len(items) - correct - (0 if cls == "none" else undetermined)
        n = len(items)
        out[cls] = {
            "items": n,
            "correct": correct,
            "undetermined": undetermined,
            "wrong": wrong,
            "correct_rate": round(correct / n, 3) if n else None,
            "wrong_rate": round(wrong / n, 3) if n else None,
            "misses": [
                {"id": r["id"], "text": r["text"], "got": g}
                for r, g in zip(items, got, strict=True)
                if g != r["expected"]
            ],
        }
    return out


def request_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    requests = [r for r in rows if r["expected"]]
    near = [r for r in rows if not r["expected"]]
    found: dict[str, str | None] = {}
    for r in rows:
        hit = find_request(r["text"])
        found[r["id"]] = hit.language if hit else None
    recalled = sum(1 for r in requests if found[r["id"]] == r["expected"])
    wrong_target = sum(
        1 for r in requests if found[r["id"]] is not None and found[r["id"]] != r["expected"]
    )
    false_pos = sum(1 for r in near if found[r["id"]] is not None)
    return {
        "requests": len(requests),
        "recalled": recalled,
        "wrong_target": wrong_target,
        "near_misses": len(near),
        "false_positives": false_pos,
        "recall": round(recalled / len(requests), 3) if requests else None,
        "false_positive_rate": round(false_pos / len(near), 3) if near else None,
        "misses": [
            {"id": r["id"], "text": r["text"], "expected": r["expected"], "got": found[r["id"]]}
            for r in rows
            if found[r["id"]] != r["expected"]
        ],
    }


def check(det: dict[str, Any], req: dict[str, Any], floors: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for lang, f in floors["languages"].items():
        s = det[lang]
        if s["correct_rate"] < f["min_correct"]:
            problems.append(f"{lang}: correct {s['correct_rate']:.3f} < {f['min_correct']}")
        if s["wrong_rate"] > f["max_wrong"]:
            problems.append(f"{lang}: wrong {s['wrong_rate']:.3f} > {f['max_wrong']}")
    need = floors["undetermined_expected"]["min_undetermined"]
    if det["none"]["correct_rate"] < need:
        problems.append(f"none: undetermined {det['none']['correct_rate']:.3f} < {need}")
    rf = floors["requests"]
    if req["recall"] < rf["min_recall"]:
        problems.append(f"requests: recall {req['recall']:.3f} < {rf['min_recall']}")
    if req["false_positive_rate"] > rf["max_false_positive_rate"]:
        problems.append(
            f"requests: false positives {req['false_positive_rate']:.3f} > "
            f"{rf['max_false_positive_rate']}"
        )
    if req["wrong_target"] > rf["max_wrong_target"]:
        problems.append(f"requests: {req['wrong_target']} with the wrong target language")
    return problems


def table(det: dict[str, Any], req: dict[str, Any]) -> str:
    lines = ["| expected | items | correct | undetermined | wrong |", "|---|---|---|---|---|"]
    for cls in CLASSES:
        s = det[cls]
        und = "-" if cls == "none" else s["undetermined"]
        lines.append(
            f"| {cls} | {s['items']} | {s['correct']} ({s['correct_rate']:.3f}) | {und} "
            f"| {s['wrong']} ({s['wrong_rate']:.3f}) |"
        )
    lines += [
        "",
        f"requests: {req['recalled']}/{req['requests']} recognised "
        f"({req['wrong_target']} wrong target); near misses read as requests: "
        f"{req['false_positives']}/{req['near_misses']}",
    ]
    return "\n".join(lines)
