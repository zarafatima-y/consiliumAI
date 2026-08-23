#!/usr/bin/env python3
"""Freeze label-blind Hendrycks queries and a complete MoralReason retrieval bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hendrycks-skills", type=Path, required=True)
    parser.add_argument("--hendrycks-private-labels", type=Path, required=True)
    parser.add_argument("--moralreason-skills", type=Path, required=True)
    parser.add_argument("--moralreason-source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-hendrycks", type=int, default=3595)
    parser.add_argument("--expected-moralreason", type=int, default=664)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str:
    return str(value or "").strip()


def source_reference(row: dict) -> dict:
    reference = row.get("source_reference")
    return reference if isinstance(reference, dict) else {}


def source_idx(row: dict) -> int:
    value = row.get("source_idx", source_reference(row).get("source_idx"))
    if value in (None, ""):
        raise ValueError(f"Missing source_idx: {row}")
    return int(value)


def target_id(row: dict) -> str:
    value = row.get("target_id", source_reference(row).get("target_id"))
    if value in (None, ""):
        raise ValueError(f"Missing target_id: {row}")
    return clean(value)


def scenario_id(row: dict) -> str:
    value = row.get("scenario_id", source_reference(row).get("scenario_id"))
    return clean(value)


def index_unique(rows: list[dict], key_fn, name: str) -> dict:
    indexed = {}
    for row in rows:
        key = key_fn(row)
        if key in indexed:
            raise ValueError(f"Duplicate {name} key: {key!r}")
        indexed[key] = row
    return indexed


def normalize_gold(row: dict) -> int:
    for key in ("gold_verdict", "label", "gold", "gold_label"):
        if key in row and row[key] not in (None, ""):
            value = int(row[key])
            if value not in (0, 1):
                raise ValueError(f"Non-binary Hendrycks label: {row[key]!r}")
            return value
    raise ValueError(f"Private label row has no recognized label field: {row}")


def normalize_hendrycks(skills: list[dict], labels: list[dict]) -> tuple[list[dict], list[dict]]:
    label_by_idx = index_unique(labels, source_idx, "Hendrycks private-label")
    queries = []
    private = []
    seen = set()
    for row in sorted(skills, key=source_idx):
        idx = source_idx(row)
        if idx in seen:
            raise ValueError(f"Duplicate Hendrycks skill source_idx={idx}")
        seen.add(idx)
        label = label_by_idx.get(idx)
        if label is None:
            raise ValueError(f"No private label for Hendrycks source_idx={idx}")
        skill_target = target_id(row)
        label_target = clean(label.get("target_id", source_reference(label).get("target_id")))
        if label_target and label_target != skill_target:
            raise ValueError(
                f"Hendrycks target mismatch at source_idx={idx}: "
                f"{skill_target!r} != {label_target!r}"
            )
        value = row.get("input") if isinstance(row.get("input"), dict) else row
        scenario = clean(value.get("scenario"))
        excuse = clean(value.get("excuse"))
        skill = clean(row.get("skill_description"))
        input_format = clean(row.get("input_format"))
        if not scenario or not excuse or not skill or input_format not in {"request", "duty_role"}:
            raise ValueError(f"Incomplete Hendrycks skill row for source_idx={idx}")
        forbidden = {"label", "gold", "gold_label", "gold_verdict"}.intersection(row)
        forbidden |= {"label", "gold", "gold_label", "gold_verdict"}.intersection(value)
        if forbidden:
            raise ValueError(f"Hendrycks skill file leaks labels at {idx}: {sorted(forbidden)}")
        queries.append(
            {
                "target_id": skill_target,
                "source_idx": idx,
                "input_format": input_format,
                "scenario": scenario,
                "excuse": excuse,
                "skill_description": skill,
            }
        )
        private.append(
            {
                "target_id": skill_target,
                "source_idx": idx,
                "input_format": input_format,
                "gold_verdict": normalize_gold(label),
            }
        )
    extra_labels = sorted(set(label_by_idx).difference(seen))
    if extra_labels:
        raise ValueError(f"Private labels without skills: {extra_labels[:10]}")
    return queries, private


def csv_scenario_id(row: dict) -> str:
    return clean(row.get("scenario_id") or row.get("id"))


def resolve_chosen_action(row: dict, action1: str, action2: str) -> str:
    chosen = clean(row.get("chosen_action"))
    if chosen:
        return chosen
    choice = clean(row.get("choice")).lower()
    if choice in {"1", "a", "action1", "action_1", "option1", "option_1"}:
        return action1
    if choice in {"2", "b", "action2", "action_2", "option2", "option_2"}:
        return action2
    raise ValueError(
        f"Cannot resolve chosen action for scenario_id={csv_scenario_id(row)!r}; "
        f"choice={row.get('choice')!r}"
    )


def normalize_moralreason(skills: list[dict], source: list[dict]) -> list[dict]:
    source_by_id = index_unique(source, csv_scenario_id, "MoralReason source scenario_id")
    bank = []
    seen = set()
    for skill_row in sorted(skills, key=source_idx):
        sid = scenario_id(skill_row)
        if not sid:
            raise ValueError(f"MoralReason skill row lacks scenario_id: {skill_row}")
        if sid in seen:
            raise ValueError(f"Duplicate MoralReason skill scenario_id={sid}")
        seen.add(sid)
        source_row = source_by_id.get(sid)
        if source_row is None:
            raise ValueError(f"No MoralReason source row for scenario_id={sid}")
        value = skill_row.get("input") if isinstance(skill_row.get("input"), dict) else skill_row
        context = clean(source_row.get("context"))
        action1 = clean(source_row.get("action1") or source_row.get("action_1"))
        action2 = clean(source_row.get("action2") or source_row.get("action_2"))
        thought = clean(source_row.get("thought_process"))
        skill = clean(skill_row.get("skill_description"))
        if not all((context, action1, action2, thought, skill)):
            missing = [
                name
                for name, item in {
                    "context": context,
                    "action1": action1,
                    "action2": action2,
                    "thought_process": thought,
                    "skill_description": skill,
                }.items()
                if not item
            ]
            raise ValueError(f"MoralReason {sid} missing fields: {missing}")
        # Protect against joining a skill to the wrong source record.
        skill_context = clean(value.get("context"))
        if skill_context and skill_context != context:
            raise ValueError(f"Context mismatch for MoralReason scenario_id={sid}")
        bank.append(
            {
                "exemplar_id": sid,
                "scenario_id": sid,
                "source_idx": source_idx(skill_row),
                "framework": clean(source_row.get("framework") or skill_row.get("framework")),
                "moral_categories": skill_row.get("moral_categories", []),
                "skill_description": skill,
                "context": context,
                "action1": action1,
                "action2": action2,
                "chosen_action": resolve_chosen_action(source_row, action1, action2),
                "thought_process": thought,
            }
        )
    return bank


def main() -> None:
    args = parse_args()
    for path in (
        args.hendrycks_skills,
        args.hendrycks_private_labels,
        args.moralreason_skills,
        args.moralreason_source_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    hendrycks_queries, private_labels = normalize_hendrycks(
        read_jsonl(args.hendrycks_skills),
        read_jsonl(args.hendrycks_private_labels),
    )
    moralreason_bank = normalize_moralreason(
        read_jsonl(args.moralreason_skills),
        read_csv(args.moralreason_source_csv),
    )
    if len(hendrycks_queries) != args.expected_hendrycks:
        raise ValueError(
            f"Expected {args.expected_hendrycks} Hendrycks rows; "
            f"found {len(hendrycks_queries)}"
        )
    if len(moralreason_bank) != args.expected_moralreason:
        raise ValueError(
            f"Expected {args.expected_moralreason} MoralReason rows; "
            f"found {len(moralreason_bank)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_path = args.output_dir / "hendrycks_queries_label_blind.jsonl"
    labels_path = args.output_dir / "hendrycks_private_labels.jsonl"
    bank_path = args.output_dir / "moralreason_retrieval_bank.jsonl"
    write_jsonl(query_path, hendrycks_queries)
    write_jsonl(labels_path, private_labels)
    write_jsonl(bank_path, moralreason_bank)
    manifest = {
        "method": "frozen full-test retrieval input preparation v1",
        "counts": {
            "hendrycks_queries": len(hendrycks_queries),
            "hendrycks_by_input_format": dict(
                Counter(row["input_format"] for row in hendrycks_queries)
            ),
            "moralreason_retrieval_bank": len(moralreason_bank),
        },
        "label_blinding": {
            "query_file_contains_gold": False,
            "private_labels_used_only_after_generation": True,
        },
        "inputs": {
            "hendrycks_skills": str(args.hendrycks_skills.resolve()),
            "hendrycks_private_labels": str(args.hendrycks_private_labels.resolve()),
            "moralreason_skills": str(args.moralreason_skills.resolve()),
            "moralreason_source_csv": str(args.moralreason_source_csv.resolve()),
        },
        "outputs": {
            "hendrycks_queries": str(query_path.resolve()),
            "hendrycks_private_labels": str(labels_path.resolve()),
            "moralreason_retrieval_bank": str(bank_path.resolve()),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["sha256"] = {
        "hendrycks_queries": sha256(query_path),
        "hendrycks_private_labels": sha256(labels_path),
        "moralreason_retrieval_bank": sha256(bank_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[complete] hendrycks_queries={len(hendrycks_queries)} path={query_path}")
    print(
        "[complete] hendrycks_by_input_format="
        f"{dict(Counter(row['input_format'] for row in hendrycks_queries))}"
    )
    print(f"[complete] private_labels={len(private_labels)} path={labels_path}")
    print(f"[complete] moralreason_bank={len(moralreason_bank)} path={bank_path}")
    print(f"[complete] manifest={manifest_path}")


if __name__ == "__main__":
    main()
