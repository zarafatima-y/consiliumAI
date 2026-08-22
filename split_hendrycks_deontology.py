#!/usr/bin/env python3
"""Split Hendrycks ETHICS Deontology CSV into request and duty/role inputs.

Routing is deliberately label-blind: only the scenario and response (normally
named ``excuse``) are used to determine the case type. Gold labels are written
to a separate private mapping and never copied into the public skill-generation
files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROUTER_VERSION = "hendrycks_deontology_text_router_v1"

REQUEST_SCENARIO_PREFIXES = (
    "aren't you supposed to",
    "are you supposed to",
    "weren't you supposed to",
    "were you supposed to",
    "could you",
    "can you",
    "would you",
    "will you",
    "couldn't you",
    "can't you",
    "don't you need to",
    "do you need to",
    "go ",
)

REQUEST_RESPONSE_PATTERNS = (
    re.compile(r"^no\b", re.IGNORECASE),
    re.compile(r"^but\b", re.IGNORECASE),
)

DUTY_ROLE_RESPONSE_PATTERNS = (
    re.compile(r"^so\s+i\s+should\b", re.IGNORECASE),
    re.compile(r"^i\s+should\b", re.IGNORECASE),
    re.compile(r"^so\s+i\s+ought\s+to\b", re.IGNORECASE),
    re.compile(r"^therefore\s+i\s+should\b", re.IGNORECASE),
)

DUTY_ROLE_SCENARIO_PATTERNS = (
    re.compile(r"^i\s+am\b", re.IGNORECASE),
    re.compile(r"^i['’]m\b", re.IGNORECASE),
    re.compile(r"^i\s+work\b", re.IGNORECASE),
    re.compile(r"^i\s+serve\b", re.IGNORECASE),
    re.compile(r"^as\s+(?:an?|the)\b", re.IGNORECASE),
    re.compile(r"^my\s+(?:job|role|position|profession)\b", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--scenario-column", default="scenario")
    parser.add_argument("--response-column", default="excuse")
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=0,
        help="Optional exact row-count assertion; 0 disables it.",
    )
    parser.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help="Complete successfully while keeping ambiguous rows in a separate file.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def starts_with_pattern(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def route_case(scenario: str, response: str) -> tuple[str, str]:
    """Return (input_format, routing_reason) without consulting the label."""

    scenario_lower = scenario.lower()
    request_response = starts_with_pattern(response, REQUEST_RESPONSE_PATTERNS)
    duty_role_response = starts_with_pattern(response, DUTY_ROLE_RESPONSE_PATTERNS)
    request_scenario = scenario.endswith("?") or scenario_lower.startswith(
        REQUEST_SCENARIO_PREFIXES
    )
    duty_role_scenario = starts_with_pattern(scenario, DUTY_ROLE_SCENARIO_PATTERNS)

    # Response form is the strongest signal in the released dataset.
    if duty_role_response and not request_response:
        if request_scenario and not duty_role_scenario:
            return "ambiguous", "duty_role_response_conflicts_with_request_scenario"
        return "duty_role", "explicit_so_i_should_response"

    if request_response and not duty_role_response:
        if duty_role_scenario and not request_scenario:
            # Some request/excuse cases have declarative scenarios. The explicit
            # No/But answer still identifies the request-style task.
            return "request", "explicit_no_or_but_response_with_declarative_scenario"
        return "request", "explicit_no_or_but_response"

    if request_scenario and not duty_role_scenario:
        return "request", "request_shaped_scenario"

    if duty_role_scenario and not request_scenario:
        return "duty_role", "duty_or_role_shaped_scenario"

    if request_scenario and duty_role_scenario:
        return "ambiguous", "scenario_has_both_request_and_duty_role_signals"

    return "ambiguous", "no_supported_textual_routing_rule"


def parse_label(raw: object, row_number: int) -> int:
    value = normalize_text(raw)
    if value not in {"0", "1"}:
        raise ValueError(
            f"CSV data row {row_number} has non-binary label {value!r}; expected 0 or 1"
        )
    return int(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.expected_rows < 0:
        raise ValueError("--expected-rows cannot be negative")
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "request": args.output_dir / "hendrycks_test_request.jsonl",
        "duty_role": args.output_dir / "hendrycks_test_duty_role.jsonl",
        "combined": args.output_dir / "hendrycks_test_input_only.jsonl",
        "private": args.output_dir / "hendrycks_test_private_labels.jsonl",
        "routing": args.output_dir / "hendrycks_test_routing_audit.jsonl",
        "ambiguous": args.output_dir / "hendrycks_test_ambiguous.jsonl",
        "summary": args.output_dir / "hendrycks_test_split_summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; use a new --output-dir or pass --overwrite: "
            + ", ".join(map(str, existing))
        )

    with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        required = {args.label_column, args.scenario_column, args.response_column}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Input CSV is missing columns {missing}; found {reader.fieldnames}"
            )
        source_rows = list(reader)

    if args.expected_rows and len(source_rows) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} CSV rows; found {len(source_rows)}"
        )
    if not source_rows:
        raise ValueError("Input CSV contains no data rows")

    public_by_type: dict[str, list[dict]] = {"request": [], "duty_role": []}
    combined: list[dict] = []
    private_labels: list[dict] = []
    routing_audit: list[dict] = []
    ambiguous: list[dict] = []
    label_counts: dict[str, Counter[int]] = {
        "request": Counter(),
        "duty_role": Counter(),
        "ambiguous": Counter(),
    }
    scenario_counts: Counter[str] = Counter()
    exact_input_counts: Counter[tuple[str, str]] = Counter()

    width = max(6, len(str(len(source_rows) - 1)))
    for source_idx, row in enumerate(source_rows):
        row_number = source_idx + 1
        scenario = normalize_text(row.get(args.scenario_column))
        response = normalize_text(row.get(args.response_column))
        if not scenario or not response:
            raise ValueError(
                f"CSV data row {row_number} has an empty scenario or response"
            )
        gold_verdict = parse_label(row.get(args.label_column), row_number)
        input_format, routing_reason = route_case(scenario, response)
        target_id = f"H{source_idx:0{width}d}"

        scenario_counts[scenario] += 1
        exact_input_counts[(scenario, response)] += 1
        label_counts[input_format][gold_verdict] += 1

        public = {
            "target_id": target_id,
            "source_idx": source_idx,
            "input_format": input_format,
            "scenario": scenario,
            "excuse": response,
        }
        private_labels.append(
            {
                "target_id": target_id,
                "source_idx": source_idx,
                "input_format": input_format,
                "gold_verdict": gold_verdict,
            }
        )
        routing_audit.append(
            {
                **public,
                "routing_reason": routing_reason,
            }
        )

        if input_format == "ambiguous":
            ambiguous.append({**public, "routing_reason": routing_reason})
            continue

        public_by_type[input_format].append(public)
        combined.append(public)

    # Preserve original CSV order after recombining the two routed groups.
    combined.sort(key=lambda row: row["source_idx"])
    private_labels.sort(key=lambda row: row["source_idx"])
    routing_audit.sort(key=lambda row: row["source_idx"])

    write_jsonl(paths["request"], public_by_type["request"])
    write_jsonl(paths["duty_role"], public_by_type["duty_role"])
    write_jsonl(paths["combined"], combined)
    write_jsonl(paths["private"], private_labels)
    write_jsonl(paths["routing"], routing_audit)
    write_jsonl(paths["ambiguous"], ambiguous)

    duplicated_inputs = sum(count - 1 for count in exact_input_counts.values() if count > 1)
    repeated_scenario_rows = sum(count for count in scenario_counts.values() if count > 1)
    summary = {
        "status": "needs_ambiguous_review" if ambiguous else "complete",
        "router_version": ROUTER_VERSION,
        "routing_is_label_blind": True,
        "input_csv": str(args.input_csv.resolve()),
        "input_sha256": sha256(args.input_csv),
        "columns": {
            "label": args.label_column,
            "scenario": args.scenario_column,
            "response": args.response_column,
        },
        "counts": {
            "total": len(source_rows),
            "request": len(public_by_type["request"]),
            "duty_role": len(public_by_type["duty_role"]),
            "ambiguous": len(ambiguous),
            "combined_routable": len(combined),
            "unique_scenarios": len(scenario_counts),
            "rows_in_repeated_scenario_groups": repeated_scenario_rows,
            "duplicated_exact_inputs_beyond_first": duplicated_inputs,
        },
        "private_label_counts_for_audit_only": {
            name: {str(label): count for label, count in sorted(counts.items())}
            for name, counts in label_counts.items()
        },
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        "next_step": (
            "Review and resolve every ambiguous record before skill generation."
            if ambiguous
            else "The two public files are ready for type-specific skill generation."
        ),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if ambiguous and not args.allow_ambiguous:
        raise SystemExit(
            f"[ABORT] {len(ambiguous)} ambiguous rows require review; see {paths['ambiguous']}"
        )


if __name__ == "__main__":
    main()
