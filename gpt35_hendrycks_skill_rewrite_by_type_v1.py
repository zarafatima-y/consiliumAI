#!/usr/bin/env python3
"""Generate Hendrycks test skills with type-matched 16-shot exemplars.

Request cases are rewritten only from the request exemplar bank. Duty/role
cases are rewritten only from the duty/role exemplar bank. The program uses
the OpenAI Batch API, is resumable, and never sends labels to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path


PROMPT_VERSION = "skill_knn_hendrycks_by_type_v1_gpt35"
STREAMS = ("request", "duty_role")

INSTRUCTION = """### Task
Study the 16 demonstrations to learn how a complete ethical case is rewritten as
one neutral description of the reasoning skill needed to evaluate it. Apply the
same transformation to the new case.

### What the description must capture
- The duty, request, role, constraint, conflict, alignment, or feasibility check
  that must be examined.
- For a request or role case, the relationship between the stated expectation and
  the proposed excuse, duty claim, or responsibility.
- For an ethical dilemma, the decision-relevant conflict between the two candidate
  actions without selecting either action.

### Abstraction rule
Preserve decision-relevant relationships such as agent, object, owner, time,
resource, prior commitment, safety constraint, and role scope. Abstract away names,
objects, occupations, and other surface story details when they are not needed to
identify the reasoning mechanism. Do not merely restate the case, but do not make
the description so general that its decisive reasoning check is lost.

### Restrictions
Do not predict a verdict, choose an action, state that a duty is released or
binding, use a gold label, copy a rationale, or introduce unsupported facts. Do
not explain your process or mention the demonstrations.

### Output
Output exactly one concise sentence beginning with "Assess". Output only that
sentence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-seed-bank", type=Path, required=True)
    parser.add_argument("--duty-role-seed-bank", type=Path, required=True)
    parser.add_argument("--request-inputs", type=Path, required=True)
    parser.add_argument("--duty-role-inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Requests per sequential Batch API shard.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--api-root",
        default=os.environ.get("OPENAI_API_ROOT", "https://api.openai.com/v1"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit/check once and exit; rerun the same command until complete.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path} line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError(
                f"Existing prepared file has a different configuration: {path}. "
                "Use a new output directory or remove that run after auditing it."
            )
        return
    write_jsonl(path, rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_type(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace("/", "_")
    aliases = {
        "request": "request",
        "requests": "request",
        "role": "duty_role",
        "duty": "duty_role",
        "duty_role": "duty_role",
        "role_duty": "duty_role",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported input type: {value!r}")
    return aliases[text]


def validate_seed_bank(rows: list[dict], expected_type: str, path: Path) -> None:
    if len(rows) != 16:
        raise ValueError(f"Expected exactly 16 seeds in {path}; found {len(rows)}")
    ids = [row.get("seed_id") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != 16:
        raise ValueError(f"{path} requires 16 unique non-empty seed_id values")
    forbidden = {"label", "gold", "gold_label", "gold_verdict", "choice", "thought_process"}
    for row in rows:
        actual_type = canonical_type(row.get("input_format"))
        if actual_type != expected_type:
            raise ValueError(
                f"{row.get('seed_id')} in {path} is {actual_type}, expected {expected_type}"
            )
        value = row.get("input")
        if not isinstance(value, dict):
            raise ValueError(f"{row.get('seed_id')} has no input object")
        leaked = sorted(forbidden.intersection(value))
        if leaked:
            raise ValueError(f"{row.get('seed_id')} input leaks fields: {leaked}")
        if not str(value.get("scenario", "")).strip() or not str(value.get("excuse", "")).strip():
            raise ValueError(f"{row.get('seed_id')} requires scenario and excuse")
        skill = str(row.get("skill_description", "")).strip()
        if not skill.startswith("Assess"):
            raise ValueError(f"{row.get('seed_id')} skill must begin with 'Assess'")


def source_reference(row: dict, row_number: int) -> dict:
    reference = {"source_idx": row.get("source_idx", row_number)}
    if row.get("target_id") is not None:
        reference["target_id"] = row["target_id"]
    return reference


def normalize_inputs(rows: list[dict], expected_type: str, path: Path) -> list[dict]:
    normalized = []
    seen = set()
    forbidden_top_level = {"label", "gold", "gold_label", "gold_verdict"}
    for row_number, row in enumerate(rows):
        declared = row.get("input_format", row.get("task_type", expected_type))
        actual_type = canonical_type(declared)
        if actual_type != expected_type:
            raise ValueError(
                f"Wrongly routed row in {path} at record {row_number + 1}: "
                f"found {actual_type}, expected {expected_type}"
            )
        scenario = str(row.get("scenario", "")).strip()
        excuse = str(row.get("excuse", "")).strip()
        if not scenario or not excuse:
            raise ValueError(f"Missing scenario/excuse in {path} record {row_number + 1}")
        reference = source_reference(row, row_number)
        key = json.dumps(reference, sort_keys=True)
        if key in seen:
            raise ValueError(f"Duplicate source reference in {path}: {reference}")
        seen.add(key)
        normalized.append(
            {
                "source_dataset": "hendrycks_test",
                "source_reference": reference,
                "input_format": expected_type,
                "input": {"scenario": scenario, "excuse": excuse},
                "private_fields_present_in_source": sorted(forbidden_top_level.intersection(row)),
            }
        )
    return normalized


def format_case(input_format: str, value: dict) -> str:
    if input_format == "request":
        kind = "request"
        second_field = "Proposed excuse or duty claim"
    else:
        kind = "role"
        second_field = "Proposed responsibility"
    return "\n".join(
        [
            f"### Case type: {kind}",
            f"### Scenario: {value['scenario']}",
            f"### {second_field}: {value['excuse']}",
        ]
    )


def build_prompt(seed_rows: list[dict], input_format: str, value: dict) -> str:
    sections = [INSTRUCTION]
    for position, row in enumerate(seed_rows, start=1):
        sections.append(
            "\n".join(
                [
                    f"### Demonstration {position}",
                    format_case(input_format, row["input"]),
                    f"### Skills: {row['skill_description']}",
                ]
            )
        )
    sections.append(
        "\n".join(
            ["### New case", format_case(input_format, value), "### Skills:"]
        )
    )
    return "\n\n".join(sections)


def prepare_requests(
    stream: str,
    normalized: list[dict],
    seeds: list[dict],
    model: str,
    max_tokens: int,
) -> tuple[list[dict], list[dict]]:
    requests = []
    metadata = []
    demonstration_ids = [row["seed_id"] for row in seeds]
    for order, row in enumerate(normalized, start=1):
        custom_id = f"hendrycks-{stream}-{order:06d}"
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": build_prompt(seeds, stream, row["input"]),
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
            }
        )
        metadata.append(
            {
                **row,
                "custom_id": custom_id,
                "order": order,
                "demonstration_bank": stream,
                "demonstration_ids": demonstration_ids,
                "requested_model": model,
                "prompt_version": PROMPT_VERSION,
            }
        )
    return requests, metadata


def api_json(
    api_root: str,
    api_key: str,
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{api_root.rstrip('/')}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI API {method} {path} failed ({error.code}): {body}"
        ) from error


def upload_batch_file(api_root: str, api_key: str, path: Path) -> str:
    boundary = f"----skill-rewrite-{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n"
            ).encode(),
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{api_root.rstrip('/')}/files",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI file upload failed ({error.code}): {response_body}"
        ) from error
    if not result.get("id"):
        raise RuntimeError(f"Upload returned no file id: {result}")
    return result["id"]


def submit_batch(api_root: str, api_key: str, request_file: Path, job: dict) -> dict:
    file_id = upload_batch_file(api_root, api_key, request_file)
    batch = api_json(
        api_root,
        api_key,
        "POST",
        "/batches",
        {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {
                "description": (
                    f"Hendrycks {job['stream']} skill rewriting "
                    f"part {job['part_index']:04d}"
                )
            },
        },
    )
    if not batch.get("id"):
        raise RuntimeError(f"Batch submission returned no id: {batch}")
    return {
        "batch_id": batch["id"],
        "input_file_id": file_id,
        "status": batch.get("status"),
        "stream": job["stream"],
        "part_index": job["part_index"],
    }


def refresh_state(api_root: str, api_key: str, state: dict, state_path: Path) -> dict:
    for job_key, entry in state["batches"].items():
        if entry.get("status") == "completed" and entry.get("output_file_id"):
            continue
        batch = api_json(api_root, api_key, "GET", f"/batches/{entry['batch_id']}")
        entry["status"] = batch.get("status")
        entry["output_file_id"] = batch.get("output_file_id")
        entry["error_file_id"] = batch.get("error_file_id")
        entry["errors"] = batch.get("errors")
        print(f"[batch] job={job_key} status={entry['status']}", flush=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def recover_failed_batches(state: dict, state_path: Path) -> dict:
    terminal_failures = {"failed", "expired", "cancelled", "cancelling"}
    for job_key, entry in list(state.get("batches", {}).items()):
        if entry.get("status") not in terminal_failures:
            continue
        errors = (entry.get("errors") or {}).get("data") or []
        codes = {error.get("code") for error in errors}
        legacy_oversized_batch = job_key in STREAMS and codes == {"token_limit_exceeded"}
        if legacy_oversized_batch:
            state.setdefault("failed_attempts", []).append(
                {"job_key": job_key, "reason": "replaced_by_shards", **entry}
            )
            del state["batches"][job_key]
            print(
                f"[recover] retired oversized legacy batch={job_key}; "
                "continuing with sequential shards",
                flush=True,
            )
            continue
        raise RuntimeError(
            f"Batch {job_key} ended with status={entry.get('status')}: "
            f"{entry.get('errors')}"
        )
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def all_complete(state: dict, job_keys: list[str]) -> bool:
    return set(state.get("batches", {})) == set(job_keys) and all(
        entry.get("status") == "completed" and entry.get("output_file_id")
        for entry in state["batches"].values()
    )


def download_file(api_root: str, api_key: str, file_id: str, output: Path) -> None:
    request = urllib.request.Request(
        f"{api_root.rstrip('/')}/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI download failed ({error.code}): {body}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)


def clean_skill(raw: str) -> str:
    value = raw.strip()
    value = re.sub(
        r"^\s*(?:#{1,4}\s*)?(?:skill description|skills?)\s*:\s*",
        "",
        value,
        flags=re.I,
    )
    value = " ".join(value.split()).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def format_checks(skill: str) -> dict:
    word_count = len(skill.split())
    return {
        "starts_with_assess": skill.startswith("Assess"),
        "single_line": "\n" not in skill,
        "word_count": word_count,
        "word_count_in_range": 12 <= word_count <= 40,
    }


def materialize(metadata: list[dict], batch_output: Path, output: Path) -> dict:
    batch_rows = read_jsonl(batch_output)
    by_id = {row.get("custom_id"): row for row in batch_rows}
    expected = {row["custom_id"] for row in metadata}
    if len(by_id) != len(batch_rows) or set(by_id) != expected:
        raise ValueError(
            f"Batch/metadata ID mismatch: missing={sorted(expected-set(by_id))[:5]} "
            f"unexpected={sorted(set(by_id)-expected)[:5]}"
        )
    output_rows = []
    usage_total = Counter()
    invalid_ids = []
    for metadata_row in sorted(metadata, key=lambda item: item["order"]):
        batch_row = by_id[metadata_row["custom_id"]]
        if batch_row.get("error"):
            raise RuntimeError(f"{metadata_row['custom_id']} failed: {batch_row['error']}")
        response = batch_row.get("response") or {}
        if response.get("status_code") != 200:
            raise RuntimeError(f"{metadata_row['custom_id']} returned {response}")
        body = response.get("body") or {}
        choices = body.get("choices") or []
        raw = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError(f"{metadata_row['custom_id']} returned no text")
        skill = clean_skill(raw)
        checks = format_checks(skill)
        format_valid = all(
            checks[key]
            for key in ("starts_with_assess", "single_line", "word_count_in_range")
        )
        if not format_valid:
            invalid_ids.append(metadata_row["custom_id"])
        usage = body.get("usage") or {}
        usage_total.update(
            {
                key: int(usage.get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        )
        output_rows.append(
            {
                key: value
                for key, value in metadata_row.items()
                if key not in {"custom_id", "order", "private_fields_present_in_source"}
            }
            | {
                "skill_description": skill,
                "raw_generation": raw,
                "rewrite_model": body.get("model"),
                "system_fingerprint": body.get("system_fingerprint"),
                "api_usage": usage,
                "batch_custom_id": metadata_row["custom_id"],
                "format_checks": checks,
                "format_valid": format_valid,
            }
        )
    write_jsonl(output, output_rows)
    return {
        "records": len(output_rows),
        "format_failures": len(invalid_ids),
        "format_failure_ids": invalid_ids,
        "usage": dict(usage_total),
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
    }


def write_combined_and_audit(
    stream_outputs: dict[str, Path],
    combined_path: Path,
    audit_path: Path,
    audit_size: int,
    seed: int,
) -> int:
    by_stream = {stream: read_jsonl(path) for stream, path in stream_outputs.items()}
    combined = by_stream["request"] + by_stream["duty_role"]
    combined.sort(
        key=lambda row: (
            int(row["source_reference"].get("source_idx", 10**12)),
            row["input_format"],
        )
    )
    write_jsonl(combined_path, combined)

    target = min(audit_size, len(combined))
    request_target = round(target * len(by_stream["request"]) / len(combined))
    targets = {
        "request": request_target,
        "duty_role": target - request_target,
    }
    rng = random.Random(seed)
    selected = []
    for stream in STREAMS:
        pool = list(by_stream[stream])
        rng.shuffle(pool)
        selected.extend(pool[: targets[stream]])
    rng.shuffle(selected)
    audit_rows = []
    for audit_id, row in enumerate(selected, start=1):
        audit_rows.append(
            {
                "audit_id": audit_id,
                "source_reference": row["source_reference"],
                "input_format": row["input_format"],
                "input": row["input"],
                "skill_description": row["skill_description"],
                "captures_reasoning_mechanism": "",
                "abstract_and_non_specific": "",
                "neutral_no_verdict_leakage": "",
                "audit_notes": "",
            }
        )
    write_jsonl(audit_path, audit_rows)
    return len(audit_rows)


def main() -> None:
    args = parse_args()
    if min(args.max_tokens, args.batch_size, args.poll_seconds, args.audit_size) < 1:
        raise ValueError("Token, batch, polling, and audit values must be positive")

    seed_paths = {
        "request": args.request_seed_bank,
        "duty_role": args.duty_role_seed_bank,
    }
    input_paths = {
        "request": args.request_inputs,
        "duty_role": args.duty_role_inputs,
    }
    seeds = {stream: read_jsonl(seed_paths[stream]) for stream in STREAMS}
    sources = {stream: read_jsonl(input_paths[stream]) for stream in STREAMS}
    for stream in STREAMS:
        validate_seed_bank(seeds[stream], stream, seed_paths[stream])
    normalized = {
        stream: normalize_inputs(sources[stream], stream, input_paths[stream])
        for stream in STREAMS
    }

    request_keys = {
        json.dumps(row["source_reference"], sort_keys=True)
        for row in normalized["request"]
    }
    duty_keys = {
        json.dumps(row["source_reference"], sort_keys=True)
        for row in normalized["duty_role"]
    }
    overlap = request_keys.intersection(duty_keys)
    if overlap:
        raise ValueError(f"Request and duty/role inputs overlap: {sorted(overlap)[:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        stream: {
            "metadata": args.out_dir / stream / "batch_metadata.jsonl",
            "batch_output": args.out_dir / stream / "batch_output.jsonl",
            "skills": args.out_dir / stream / f"hendrycks_{stream}_skills.jsonl",
        }
        for stream in STREAMS
    }
    state_path = args.out_dir / "openai_batch_state.json"
    manifest_path = args.out_dir / "manifest.json"
    combined_path = args.out_dir / "hendrycks_test_skills_combined.jsonl"
    audit_path = args.out_dir / f"hendrycks_skill_audit_{args.audit_size}.jsonl"

    metadata = {}
    jobs = []
    for stream in STREAMS:
        requests, metadata[stream] = prepare_requests(
            stream, normalized[stream], seeds[stream], args.model, args.max_tokens
        )
        ensure_jsonl(paths[stream]["metadata"], metadata[stream])
        for offset in range(0, len(requests), args.batch_size):
            part_index = offset // args.batch_size + 1
            job_key = f"{stream}_part_{part_index:04d}"
            request_path = (
                args.out_dir / stream / "parts" / f"{job_key}_requests.jsonl"
            )
            output_path = (
                args.out_dir / stream / "parts" / f"{job_key}_output.jsonl"
            )
            ensure_jsonl(request_path, requests[offset : offset + args.batch_size])
            jobs.append(
                {
                    "job_key": job_key,
                    "stream": stream,
                    "part_index": part_index,
                    "request_path": request_path,
                    "output_path": output_path,
                }
            )
        print(
            f"[prepared] stream={stream} records={len(requests)} "
            f"parts={(len(requests) + args.batch_size - 1) // args.batch_size} "
            f"seed_bank={seed_paths[stream]}",
            flush=True,
        )
    job_by_key = {job["job_key"]: job for job in jobs}
    job_keys = [job["job_key"] for job in jobs]

    manifest = {
        "status": "prepared" if args.prepare_only else "running",
        "method": "Type-matched 16-shot Skill-KNN rewriting via OpenAI Batch API",
        "paper_basis": "An et al., EMNLP 2023",
        "prompt_version": PROMPT_VERSION,
        "instruction": INSTRUCTION,
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "batch_jobs": job_keys,
        "seed": args.seed,
        "audit_size": args.audit_size,
        "seed_banks": {
            stream: {
                "path": str(seed_paths[stream].resolve()),
                "sha256": file_sha256(seed_paths[stream]),
            }
            for stream in STREAMS
        },
        "inputs": {
            stream: {
                "path": str(input_paths[stream].resolve()),
                "sha256": file_sha256(input_paths[stream]),
                "records": len(normalized[stream]),
            }
            for stream in STREAMS
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(f"[complete] prepare-only output={args.out_dir}")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"batches": {}}
    )
    if state["batches"]:
        print(f"[resume] state={state_path}", flush=True)
        state = refresh_state(args.api_root, api_key, state, state_path)
        state = recover_failed_batches(state, state_path)

    if args.no_wait:
        active = [
            job_key
            for job_key, entry in state["batches"].items()
            if entry.get("status") != "completed" or not entry.get("output_file_id")
        ]
        if not active:
            missing = [job_key for job_key in job_keys if job_key not in state["batches"]]
            if missing:
                job_key = missing[0]
                job = job_by_key[job_key]
                state["batches"][job_key] = submit_batch(
                    args.api_root, api_key, job["request_path"], job
                )
                state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
                print(
                    f"[submitted] job={job_key} records="
                    f"{len(read_jsonl(job['request_path']))} "
                    f"batch_id={state['batches'][job_key]['batch_id']}",
                    flush=True,
                )
    else:
        while not all_complete(state, job_keys):
            active = [
                job_key
                for job_key, entry in state["batches"].items()
                if entry.get("status") != "completed" or not entry.get("output_file_id")
            ]
            if not active:
                missing = [
                    job_key for job_key in job_keys if job_key not in state["batches"]
                ]
                job_key = missing[0]
                job = job_by_key[job_key]
                state["batches"][job_key] = submit_batch(
                    args.api_root, api_key, job["request_path"], job
                )
                state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
                print(
                    f"[submitted] job={job_key} records="
                    f"{len(read_jsonl(job['request_path']))} "
                    f"batch_id={state['batches'][job_key]['batch_id']}",
                    flush=True,
                )
            time.sleep(args.poll_seconds)
            state = refresh_state(args.api_root, api_key, state, state_path)
            state = recover_failed_batches(state, state_path)

    if not all_complete(state, job_keys):
        manifest["status"] = "submitted_or_pending"
        manifest["batch_state"] = state
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[pending] rerun the identical command; state={state_path}", flush=True)
        return

    summaries = {}
    for stream in STREAMS:
        stream_batch_rows = []
        for job in (item for item in jobs if item["stream"] == stream):
            if not job["output_path"].exists():
                download_file(
                    args.api_root,
                    api_key,
                    state["batches"][job["job_key"]]["output_file_id"],
                    job["output_path"],
                )
            stream_batch_rows.extend(read_jsonl(job["output_path"]))
        write_jsonl(paths[stream]["batch_output"], stream_batch_rows)
        summaries[stream] = materialize(
            metadata[stream], paths[stream]["batch_output"], paths[stream]["skills"]
        )

    audit_count = write_combined_and_audit(
        {stream: paths[stream]["skills"] for stream in STREAMS},
        combined_path,
        audit_path,
        args.audit_size,
        args.seed,
    )
    manifest.update(
        {
            "status": "complete",
            "batch_state": state,
            "outputs": summaries,
            "combined_output": str(combined_path.resolve()),
            "combined_output_sha256": file_sha256(combined_path),
            "audit_output": str(audit_path.resolve()),
            "audit_records": audit_count,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[complete] combined={combined_path}")
    print(f"[complete] audit={audit_path}")


if __name__ == "__main__":
    main()
