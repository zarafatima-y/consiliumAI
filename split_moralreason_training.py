#!/usr/bin/env python3
"""Categorize MoralReason training cases and select skill-writing seeds.

Category assignment is deliberately answer-blind. By default, the router uses
the complete public decision input (context, action1, action2), but it never
consults thought_process, choice, or chosen_action. Cases may receive multiple
neutral moral categories. Cases with no supported keyword match are written to
an ambiguous file for human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROUTER_VERSION = "moralreason_neutral_moral_category_keyword_router_v1"

CATEGORY_DESCRIPTIONS = {
    "life_and_lethal_harm": "Questions involving death, killing, preservation of life, or lethal force.",
    "pain_and_suffering": "Questions involving physical or emotional pain, suffering, injury, or abuse.",
    "ability_and_impairment": "Questions involving disability, bodily or mental impairment, or loss of capacity.",
    "freedom_and_coercion": "Questions involving autonomy, consent, coercion, confinement, force, or restricted choice.",
    "pleasure_and_welfare": "Questions involving welfare, well-being, happiness, pleasure, or deprivation of benefit.",
    "truth_and_deception": "Questions involving honesty, deception, secrecy, disclosure, or misleading others.",
    "fairness_and_cheating": "Questions involving fairness, cheating, theft, fraud, favoritism, or exploitation.",
    "promises_and_commitments": "Questions involving promises, agreements, vows, contracts, or commitments.",
    "law_and_legality": "Questions involving laws, crimes, legal institutions, punishment, or lawful authority.",
    "role_duty_and_obligation": "Questions involving duties, responsibilities, professional roles, orders, or obligations.",
}


def compile_terms(*terms: str) -> re.Pattern[str]:
    alternatives = "|".join(terms)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


# Patterns intentionally emphasize reasonably specific words and phrases. The
# audit file retains every match so false positives can be corrected explicitly.
CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "life_and_lethal_harm": compile_terms(
        r"kill(?:s|ed|ing)?", r"murder(?:s|ed|ing)?", r"death", r"dead",
        r"die", r"dies", r"dying", r"life", r"lives", r"lethal",
        r"fatal", r"suicid(?:e|al)", r"execut(?:e|es|ed|ion)",
        r"assassinat(?:e|es|ed|ion)", r"shoot(?:s|ing)?", r"shot",
        r"stab(?:s|bed|bing)?", r"grenade", r"weapon of mass destruction",
        r"end (?:his|her|their|a|the) life", r"preserv(?:e|es|ed|ing) life",
    ),
    "pain_and_suffering": compile_terms(
        r"pain", r"painful", r"suffer(?:s|ed|ing)?", r"agony", r"tortur(?:e|es|ed|ing)",
        r"injur(?:y|ies|ed)", r"wound(?:s|ed)?", r"harm(?:s|ed|ful|ing)?",
        r"abuse(?:r|d)?", r"assault(?:s|ed|ing)?", r"rape(?:s|d)?",
        r"attack(?:s|ed|ing)?", r"beat(?:s|en|ing)?", r"fight(?:s|ing)?",
        r"distress(?:ed)?", r"trauma(?:tic)?", r"violence", r"violent",
    ),
    "ability_and_impairment": compile_terms(
        r"disabl(?:e|ed|ing|ity)", r"impair(?:s|ed|ment|ing)?", r"incapacitat(?:e|ed|ion)",
        r"amputat(?:e|ed|ion)", r"paraly(?:ze|zed|sis)", r"unconscious",
        r"organ transplant(?:ation)?s?", r"replacement organ", r"bodily capacity",
        r"mental capacity", r"loss of ability",
    ),
    "freedom_and_coercion": compile_terms(
        r"freedom", r"liberty", r"autonomy", r"consent(?:s|ed|ing)?",
        r"without consent", r"coerc(?:e|es|ed|ion|ive)",
        r"forc(?:e|es|ed|ing)", r"kidnap(?:s|ped|ping)?", r"hostage",
        r"imprison(?:s|ed|ment|ing)?", r"confin(?:e|ed|ement|ing)",
        r"restrict(?:s|ed|ion|ing)?", r"threaten(?:s|ed|ing)?", r"demand(?:s|ed|ing)?",
    ),
    "pleasure_and_welfare": compile_terms(
        r"pleasure", r"happiness", r"well[- ]being", r"welfare",
        r"quality of life", r"depriv(?:e|es|ed|ing) (?:of )?(?:pleasure|benefit)",
        r"compassion", r"mercy", r"reliev(?:e|es|ed|ing) suffering",
    ),
    "truth_and_deception": compile_terms(
        r"lie", r"lies", r"lied", r"lying", r"deceiv(?:e|es|ed|ing)",
        r"deception", r"dishonest(?:y)?", r"honest(?:y)?", r"truth(?:ful)?",
        r"mislead(?:s|ing)?", r"secret(?:s)?", r"conceal(?:s|ed|ing|ment)?",
        r"disclos(?:e|es|ed|ing|ure)", r"confess(?:es|ed|ing|ion)?",
        r"withhold(?:s|ing)? information", r"do not tell", r"cover up",
    ),
    "fairness_and_cheating": compile_terms(
        r"cheat(?:s|ed|ing)?", r"fair(?:ness|ly)?", r"unfair(?:ly|ness)?",
        r"steal(?:s|ing)?", r"stole", r"stolen", r"theft", r"fraud(?:ulent)?",
        r"plagiari(?:ze|zed|sm)", r"exploit(?:s|ed|ation|ing)?",
        r"favoritism", r"discriminat(?:e|es|ed|ion|ing)", r"equal(?:ity|ly)?",
        r"affair", r"infidelity", r"unfaithful", r"sexual(?:ly)? unfaithful",
        r"credit card", r"take credit", r"undeserved advantage",
    ),
    "promises_and_commitments": compile_terms(
        r"promis(?:e|es|ed|ing)", r"commitment(?:s)?", r"committed to",
        r"pledge(?:s|d)?", r"vow(?:s|ed)?", r"agreement(?:s)?", r"contract(?:s|ed)?",
        r"gave (?:his|her|their|my|your) word", r"keep (?:his|her|their|my|your) word",
    ),
    "law_and_legality": compile_terms(
        r"law", r"laws", r"legal(?:ly|ity)?", r"illegal(?:ly|ity)?", r"unlawful(?:ly)?",
        r"crime(?:s)?", r"criminal(?:s)?", r"police", r"jury", r"juror(?:s)?",
        r"court", r"judge", r"verdict", r"guilty", r"innocent", r"arrest(?:s|ed|ing)?",
        r"prison(?:er|ers)?", r"punish(?:es|ed|ment|ing)?", r"constitution(?:al)?",
        r"government", r"regulation(?:s)?", r"authorities", r"death penalty",
    ),
    "role_duty_and_obligation": compile_terms(
        r"duty", r"duties", r"obligation(?:s)?", r"responsib(?:le|ility|ilities)",
        r"ordered", r"order(?:s|ed) you", r"command(?:s|ed|ing)?", r"professional",
        r"soldier", r"police officer", r"police offer", r"doctor", r"nurse",
        r"surgeon", r"specialist", r"teacher", r"attorney", r"lawyer", r"judge",
        r"juror", r"jury", r"editor", r"ceo", r"manager", r"parent", r"guardian",
        r"counselor", r"officer", r"employee", r"employer", r"scientist",
        r"paramedic", r"ambulance driver", r"military commander", r"undercover agent",
        r"part of (?:a|an|the) (?:group|team|mission)", r"on duty",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--routing-text",
        choices=("context", "full_input"),
        default="full_input",
        help="full_input uses context plus both candidate actions; neither mode uses answers.",
    )
    parser.add_argument(
        "--overrides-jsonl",
        type=Path,
        help="Human corrections with scenario_id and moral_categories list.",
    )
    parser.add_argument("--allow-ambiguous", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader.fieldnames), list(reader)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def keyword_matches(text: str) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for category, pattern in CATEGORY_PATTERNS.items():
        values = sorted({match.group(0).casefold() for match in pattern.finditer(text)})
        if values:
            matches[category] = values
    return matches


def load_overrides(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    overrides: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            scenario_id = normalize(row.get("scenario_id"))
            categories = row.get("moral_categories")
            if not scenario_id or not isinstance(categories, list) or not categories:
                raise ValueError(
                    f"Invalid override at {path}:{line_number}; moral_categories "
                    "must be a non-empty JSON list"
                )
            categories = sorted({normalize(value) for value in categories})
            unknown = sorted(set(categories).difference(CATEGORY_DESCRIPTIONS))
            if unknown:
                raise ValueError(f"Unknown categories at {path}:{line_number}: {unknown}")
            if scenario_id in overrides:
                raise ValueError(f"Duplicate override for {scenario_id}")
            overrides[scenario_id] = categories
    return overrides


def choose_stratified_seeds(
    records: list[dict], count: int, random_seed: int
) -> list[dict]:
    """Select unique cases with broad category coverage without using answers."""

    if count < 1:
        raise ValueError("--seed-count must be positive")
    if len(records) < count:
        raise ValueError(
            f"Need {count} categorized records for seeds; found only {len(records)}"
        )

    rng = random.Random(random_seed)
    tie_break = {record["scenario_id"]: rng.random() for record in records}
    category_frequency = Counter(
        category for record in records for category in record["moral_categories"]
    )
    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_category_counts: Counter[str] = Counter()

    def candidate_key(record: dict, focus: str | None = None) -> tuple:
        categories = record["moral_categories"]
        uncovered = sum(selected_category_counts[c] == 0 for c in categories)
        rarity = sum(1 / category_frequency[c] for c in categories)
        focus_count = selected_category_counts[focus] if focus else 0
        return (
            focus_count,
            -uncovered,
            -rarity,
            len(categories),
            tie_break[record["scenario_id"]],
            record["scenario_id"],
        )

    # First guarantee one example for every category represented in the data.
    for category in CATEGORY_DESCRIPTIONS:
        if selected_category_counts[category] > 0:
            continue
        candidates = [
            record for record in records
            if record["scenario_id"] not in selected_ids
            and category in record["moral_categories"]
        ]
        if not candidates or len(selected) >= count:
            continue
        chosen = min(candidates, key=lambda record: candidate_key(record, category))
        selected.append(chosen)
        selected_ids.add(chosen["scenario_id"])
        selected_category_counts.update(chosen["moral_categories"])

    # Fill remaining slots by improving coverage of underrepresented categories.
    while len(selected) < count:
        candidates = [r for r in records if r["scenario_id"] not in selected_ids]
        if not candidates:
            break
        chosen = min(candidates, key=candidate_key)
        selected.append(chosen)
        selected_ids.add(chosen["scenario_id"])
        selected_category_counts.update(chosen["moral_categories"])

    return selected


def main() -> None:
    args = parse_args()
    fields, source_rows = read_csv(args.input_csv)
    required = {
        "scenario_id", "framework", "context", "action1", "action2",
        "thought_process", "choice", "chosen_action",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"Input CSV is missing columns {missing}; found {fields}")
    if args.expected_rows and len(source_rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows; found {len(source_rows)}")
    if not source_rows:
        raise ValueError("Input CSV contains no rows")

    scenario_ids = [normalize(row["scenario_id"]) for row in source_rows]
    duplicates = sorted(k for k, n in Counter(scenario_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate scenario IDs: {duplicates[:10]}")
    overrides = load_overrides(args.overrides_jsonl)
    unknown_overrides = sorted(set(overrides).difference(scenario_ids))
    if unknown_overrides:
        raise ValueError(f"Overrides contain unknown scenario IDs: {unknown_overrides[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "categorized": args.output_dir / "moralreason_train_categorized.jsonl",
        "ambiguous": args.output_dir / "moralreason_train_ambiguous.jsonl",
        "audit": args.output_dir / "moralreason_train_category_audit.jsonl",
        "override_template": args.output_dir / "moralreason_train_category_override_template.jsonl",
        "seeds": args.output_dir / f"moralreason_train_seed_bank_{args.seed_count}.jsonl",
        "payload": args.output_dir / "moralreason_train_exemplar_payload.jsonl",
        "summary": args.output_dir / "moralreason_train_category_summary.json",
    }
    category_paths = {
        category: args.output_dir / f"moralreason_train_category_{category}.jsonl"
        for category in CATEGORY_DESCRIPTIONS
    }
    existing = [p for p in [*paths.values(), *category_paths.values()] if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; choose another --output-dir or pass --overwrite: "
            + ", ".join(map(str, existing))
        )

    categorized: list[dict] = []
    ambiguous: list[dict] = []
    audits: list[dict] = []
    payloads: list[dict] = []
    by_category: dict[str, list[dict]] = defaultdict(list)

    for source_idx, row in enumerate(source_rows):
        scenario_id = normalize(row["scenario_id"])
        framework = normalize(row["framework"])
        context = normalize(row["context"])
        action1 = normalize(row["action1"])
        action2 = normalize(row["action2"])
        if not all((scenario_id, framework, context, action1, action2)):
            raise ValueError(f"Row {source_idx} has an empty required public field")

        routing_text = context
        if args.routing_text == "full_input":
            routing_text = " ".join((context, action1, action2))
        matches = keyword_matches(routing_text)
        inferred_categories = sorted(matches)
        categories = overrides.get(scenario_id, inferred_categories)
        source = "human_override" if scenario_id in overrides else "keyword_router"

        public = {
            "scenario_id": scenario_id,
            "source_idx": source_idx,
            "source_dataset": "moralreason",
            "source_split": "train",
            "framework": framework,
            "moral_categories": categories,
            "category_source": source,
            "context": context,
            "action1": action1,
            "action2": action2,
        }
        payloads.append(
            {
                **public,
                "thought_process": normalize(row["thought_process"]),
                "choice": normalize(row["choice"]),
                "chosen_action": normalize(row["chosen_action"]),
            }
        )
        audits.append(
            {
                **public,
                "routing_text_mode": args.routing_text,
                "inferred_categories": inferred_categories,
                "matched_keywords": matches,
                "human_override_applied": scenario_id in overrides,
                "needs_human_review": not categories,
            }
        )
        if not categories:
            ambiguous.append(public)
            continue
        categorized.append(public)
        for category in categories:
            by_category[category].append(public)

    seed_records = choose_stratified_seeds(categorized, args.seed_count, args.seed)
    seeds = [
        {
            "seed_id": f"moralreason_seed_{position:02d}",
            "source_dataset": "moralreason",
            "source_split": "train",
            "source_reference": {
                "scenario_id": record["scenario_id"],
                "source_idx": record["source_idx"],
            },
            "framework": record["framework"],
            "moral_categories": record["moral_categories"],
            "category_descriptions": {
                category: CATEGORY_DESCRIPTIONS[category]
                for category in record["moral_categories"]
            },
            "input": {
                "context": record["context"],
                "action1": record["action1"],
                "action2": record["action2"],
            },
            "skill_description": "",
        }
        for position, record in enumerate(seed_records, 1)
    ]

    write_jsonl(paths["categorized"], categorized)
    write_jsonl(paths["ambiguous"], ambiguous)
    write_jsonl(paths["audit"], audits)
    write_jsonl(
        paths["override_template"],
        (
            {
                "scenario_id": row["scenario_id"],
                "moral_categories": [],
                "audit_note": "",
            }
            for row in ambiguous
        ),
    )
    write_jsonl(paths["seeds"], seeds)
    write_jsonl(paths["payload"], payloads)
    for category, path in category_paths.items():
        write_jsonl(path, by_category[category])

    seed_coverage = Counter(
        category for record in seeds for category in record["moral_categories"]
    )
    summary = {
        "status": "needs_ambiguous_review" if ambiguous else "complete",
        "router_version": ROUTER_VERSION,
        "routing_is_answer_blind": True,
        "routing_text": args.routing_text,
        "input_csv": str(args.input_csv.resolve()),
        "input_sha256": sha256(args.input_csv),
        "counts": {
            "total": len(source_rows),
            "categorized": len(categorized),
            "ambiguous": len(ambiguous),
            "seed_examples": len(seeds),
            "human_overrides": len(overrides),
        },
        "category_membership_counts": {
            category: len(by_category[category]) for category in CATEGORY_DESCRIPTIONS
        },
        "seed_category_coverage": dict(sorted(seed_coverage.items())),
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "outputs": {
            **{name: str(path.resolve()) for name, path in paths.items()},
            "category_files": {
                name: str(path.resolve()) for name, path in category_paths.items()
            },
        },
        "notes": [
            "Category files overlap because moral dilemmas may have multiple categories.",
            "Seed selection uses no thought process, choice, or chosen action.",
            "Audit ambiguous cases and false-positive keyword matches before skill generation.",
            "Keep the full exemplar payload out of embedding and skill-writing inputs.",
        ],
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if ambiguous and not args.allow_ambiguous:
        raise SystemExit(
            f"[ABORT] {len(ambiguous)} records require review; see {paths['ambiguous']}"
        )


if __name__ == "__main__":
    main()
