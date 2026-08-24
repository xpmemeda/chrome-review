#!/usr/bin/env python3
"""Stream a Bernard log and summarize eligible KV-cache sessions."""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import time
from pathlib import Path


ALLOC_RE = re.compile(
    rb"^(\d{4}-\d\d-\d\d) (\d\d):(\d\d):(\d\d),(\d+).*"
    rb"kv_manager\.py:174 allocate kv for task ([0-9a-f-]+) "
    rb"with token_num (\d+), capacity \d+, hit_length (\d+)"
)
RELEASE_RE = re.compile(
    rb"kvcache_manager\.py:2835 task ([0-9a-f-]+) session (\S+) release"
)

IMAGE_TASK_RE = re.compile(rb"\btask ([0-9a-f-]+)\b")
IMAGE_METADATA_RE = re.compile(rb"ImageMetadata\(([^)]*)\)")
IMAGE_FIELD_RE = re.compile(rb"(\w+)=([^,]+)")
ALLOWED_IMAGE_SIZE = (632, 1400)


def summarize_images(images):
    # 同一 image_id 在历史请求或多个 worker 中重复出现，只计一次。
    counts = Counter(image[3] for image in images if image[3] is not None)
    count = sum(counts.values())
    return {
        "width": 632,
        "height": 1400,
        "unique_images": len(images),
        "images_with_token_count": count,
        "images_missing_token_count": len(images) - count,
        "tokens_per_image": next(iter(counts)) if len(counts) == 1 else None,
        "token_distribution": [
            {"tokens": tokens, "images": n} for tokens, n in sorted(counts.items())
        ],
        "average_tokens_per_image": (
            sum(tokens * n for tokens, n in counts.items()) / count if count else None
        ),
    }


def summarize(rows):
    count = len(rows)
    prompt_sum = sum(row[1] for row in rows)
    hit_sum = sum(row[2] for row in rows)
    return {
        "requests": count,
        "average_prompt_tokens": prompt_sum / count if count else 0,
        "average_hit_tokens": hit_sum / count if count else 0,
        "hit_rate": hit_sum / prompt_sum if prompt_sum else 0,
        "average_compute_tokens": (prompt_sum - hit_sum) / count if count else 0,
    }


def summarize_growth(rows):
    count = len(rows)
    new_sum = sum(row[0] for row in rows)
    rewrite_sum = sum(row[1] for row in rows)
    return {
        "requests": count,
        "average_new_tokens": new_sum / count if count else 0,
        "average_rewrite_tokens": rewrite_sum / count if count else 0,
    }


def analyze(log_path: Path, start: time, min_turns: int, max_turns: int):
    allocations = {}
    task_sessions = {}
    sequence = 0
    task_images = defaultdict(set)
    unmapped_image_records = 0

    with log_path.open("rb", buffering=16 * 1024 * 1024) as stream:
        for line in stream:
            # 在请求正文之前寻找 task 标识，不输出或执行日志中的请求正文。
            if b"ImageMetadata(" in line:
                prefix = line.split(b"ImageMetadata(", 1)[0]
                image_task = IMAGE_TASK_RE.search(prefix)
                if image_task:
                    task_id = image_task.group(1).decode()
                    for metadata in IMAGE_METADATA_RE.finditer(line):
                        fields = dict(IMAGE_FIELD_RE.findall(metadata.group(1)))
                        def number(key):
                            value = fields.get(key, b"").strip()
                            return int(value) if value.isdigit() else None
                        identity = fields.get(b"image_id", b"None").strip()
                        if identity in (b"None", b""):
                            identity = fields.get(b"hash", b"None").strip()
                        if identity in (b"None", b""):
                            identity = task_id.encode() + b":" + str(metadata.start()).encode()
                        task_images[task_id].add((
                            identity, number(b"width"), number(b"height"),
                            number(b"num_tokens"),
                        ))
                else:
                    unmapped_image_records += 1
            match = ALLOC_RE.search(line)
            if match:
                task_id = match.group(6).decode()
                if task_id not in allocations:
                    observed_time = time(
                        int(match.group(2)),
                        int(match.group(3)),
                        int(match.group(4)),
                        int(match.group(5)) * 1000,
                    )
                    allocations[task_id] = (
                        sequence,
                        int(match.group(7)),
                        int(match.group(8)),
                        observed_time,
                    )
                    sequence += 1
                continue

            match = RELEASE_RE.search(line)
            if match:
                task_sessions[match.group(1).decode()] = match.group(2).decode()

    sessions = defaultdict(list)
    for task_id, session_id in task_sessions.items():
        if task_id in allocations:
            sessions[session_id].append(allocations[task_id])

    session_images = defaultdict(set)
    for task_id, images in task_images.items():
        if task_id in task_sessions:
            session_images[task_sessions[task_id]].update(images)

    turn_count_eligible = {}
    excluded_before_start = 0
    excluded_turn_count = 0
    excluded_image_size = 0
    for session_id, rows in sessions.items():
        rows.sort(key=lambda row: row[0])
        if rows[0][3] < start:
            excluded_before_start += 1
            continue
        if not min_turns <= len(rows) <= max_turns:
            excluded_turn_count += 1
            continue
        if any(image[1:3] != ALLOWED_IMAGE_SIZE for image in session_images[session_id]):
            excluded_image_size += 1
            continue
        turn_count_eligible[session_id] = rows

    eligible = turn_count_eligible

    first_rows = [rows[0] for rows in eligible.values()]
    later_rows = [row for rows in eligible.values() for row in rows[1:]]
    turns = [len(rows) for rows in eligible.values()]

    growth_by_turn = {turn: [] for turn in range(15, 26)}
    for rows in eligible.values():
        for turn in range(15, min(25, len(rows)) + 1):
            previous = rows[turn - 2]
            current = rows[turn - 1]
            new_tokens = current[1] - previous[1]
            compute_tokens = current[1] - current[2]
            rewrite_tokens = compute_tokens - new_tokens
            growth_by_turn[turn].append((new_tokens, rewrite_tokens))

    growth_rows = [
        {"turn": turn, **summarize_growth(growth_by_turn[turn])}
        for turn in range(15, 26)
    ]
    all_growth = [row for rows in growth_by_turn.values() for row in rows]
    growth_rows.append({"turn": "average", **summarize_growth(all_growth)})

    eligible_images = set()
    for session_id in eligible:
        eligible_images.update(session_images[session_id])

    return {
        "log_path": str(log_path),
        "cohort": {
            "first_observed_at_or_after": start.isoformat(),
            "minimum_turns": min_turns,
            "maximum_turns": max_turns,
            "allowed_image_size": list(ALLOWED_IMAGE_SIZE),
            "sessions_without_observed_images_allowed": True,
        },
        "eligible_sessions": len(eligible),
        "eligible_requests": sum(turns),
        "average_turns_per_session": sum(turns) / len(turns) if turns else 0,
        "first_turn": summarize(first_rows),
        "later_turns": summarize(later_rows),
        "turns_15_to_25": growth_rows,
        "images_632x1400": summarize_images(eligible_images),
        "diagnostics": {
            "mapped_sessions": len(sessions),
            "excluded_seen_before_start": excluded_before_start,
            "excluded_outside_turn_range": excluded_turn_count,
            "excluded_image_size_or_unknown_dimensions": excluded_image_size,
            "eligible_sessions_with_observed_images": sum(
                bool(session_images[sid]) for sid in eligible
            ),
            "image_metadata_lines_without_task": unmapped_image_records,
            "image_tasks_without_session": sum(
                task_id not in task_sessions for task_id in task_images
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--start-hour", type=int, default=6)
    parser.add_argument("--min-turns", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=50)
    args = parser.parse_args()

    if not 0 <= args.start_hour <= 23:
        parser.error("--start-hour must be between 0 and 23")
    if args.min_turns < 1 or args.max_turns < args.min_turns:
        parser.error("invalid turn range")
    if not args.log_path.is_file():
        parser.error(f"log file not found: {args.log_path}")

    result = analyze(
        args.log_path,
        start=time(args.start_hour),
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
