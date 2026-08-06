#!/usr/bin/env python3
import argparse
import datetime as dt
import re
from collections import Counter, defaultdict
from pathlib import Path

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
UUID_RE = r"[0-9a-f-]{36}"


def ts_ms(line):
    m = TS_RE.match(line)
    if not m:
        return None
    return int(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1000)


def scan_log(path):
    data = {
        "call": set(),
        "done": set(),
        "msg": set(),
        "hi": set(),
        "story": set(),
        "prompt_len": {},
        "cache_hit": {},
        "events": defaultdict(dict),
    }
    with Path(path).open(errors="ignore") as f:
        for line in f:
            t = ts_ms(line)
            m = re.search(r"(?:stream )?call req .* task_id (" + UUID_RE + r")", line)
            if m:
                data["call"].add(m.group(1))
            m = re.search(r"finish task (" + UUID_RE + r") status TaskStatus\.DONE", line)
            if m:
                data["done"].add(m.group(1))
            if "vlm_processor.py:710" in line:
                m = re.search(r"task (" + UUID_RE + r")", line)
                if m:
                    tid = m.group(1)
                    data["msg"].add(tid)
                    if "hi hi" in line:
                        data["hi"].add(tid)
                    if "Tell me a long story" in line:
                        data["story"].add(tid)
            m = re.search(r"call add task (" + UUID_RE + r"), prompt_token_len: (\d+)", line)
            if m:
                data["prompt_len"][m.group(1)] = int(m.group(2))
            m = re.search(r"finish return resp task (" + UUID_RE + r").*cache_hit_len=(\d+)", line)
            if m:
                data["cache_hit"][m.group(1)] = int(m.group(2))
            if t is not None:
                ids = re.findall(UUID_RE, line)
                if ids:
                    tid = ids[0]
                    ev = data["events"][tid]
                    if "base_handler.py:135 call req" in line:
                        ev["call"] = t
                    if "DriverProxy send tasks" in line:
                        ev["driver"] = t
                    if "allocate kv for task" in line and "hit_length" in line:
                        ev["kv_alloc"] = t
                    m2 = re.search(r"hit_length (\d+)", line)
                    if m2 and "allocate kv for task" in line:
                        ev["p_hit_len"] = int(m2.group(1))
                    m2 = re.search(r"kv_cache\.sp_d2d\.copy time cost: ([\d.]+) ms", line)
                    if m2:
                        ev["sp_d2d"] = float(m2.group(1))
                        ev["sp_d2d_end"] = t
                    if "to input_embed_shm, is_finish: True" in line:
                        ev["embed_done"] = t
                    if "post_allocate_kv" in line and "hit_length" in line:
                        ev["post_alloc"] = t
                    if "finish task" in line and "from signal_src SignalSource.COMPUTE_DRIVER" in line:
                        ev.setdefault("compute_done", t)
                    if "check transfer result RetCode.SUCCESS" in line:
                        ev["transfer_done"] = t
                    if "send transfer result" in line:
                        ev["send_transfer_result"] = t
                    m2 = re.search(r"prefill tasks \[([^\]]+)\] execute time: ([\d.]+) s", line)
                    if m2:
                        elapsed_ms = float(m2.group(2)) * 1000
                        for quoted_tid in re.findall(UUID_RE, m2.group(1)):
                            step_ev = data["events"][quoted_tid]
                            step_ev.setdefault("infer_exec_ms", 0.0)
                            step_ev.setdefault("infer_exec_count", 0)
                            step_ev["infer_exec_ms"] += elapsed_ms
                            step_ev["infer_exec_count"] += 1
                    m2 = re.search(r"kv_cache\.d2h\.copy time cost: ([\d.]+) ms", line)
                    if m2:
                        ev["d2h_post"] = float(m2.group(1))
                        ev["d2h_end"] = t
    return data


def complete(data):
    return data["call"] & data["done"]


def cmd_summary(args):
    p, d = scan_log(args.p_log), scan_log(args.d_log)
    pc, dc = complete(p), complete(d)
    both = pc & dc
    print(f"p.log call={len(p['call'])} done={len(p['done'])} complete={len(pc)}")
    print(f"d.log call={len(d['call'])} done={len(d['done'])} complete={len(dc)}")
    print(f"both_logs_complete={len(both)}")
    print(f"p_complete_hi={len(pc & p['hi'])} p_complete_story={len(pc & p['story'])} p_complete_non_hi={len(pc - p['hi'])}")
    print(f"d_complete_hi={len(dc & d['hi'])} d_complete_story={len(dc & d['story'])} d_complete_non_hi={len(dc - d['hi'])}")
    print(f"both_complete_hi={len(both & (p['hi'] | d['hi']))} both_complete_story={len(both & (p['story'] | d['story']))}")


def cmd_story_hits(args):
    p, d = scan_log(args.p_log), scan_log(args.d_log)
    ids = sorted((p["story"] | d["story"]) & complete(p) & complete(d))
    print("task\tprompt_token_len\tcache_hit_len")
    plen_dist, hit_dist = Counter(), Counter()
    for tid in ids:
        plen = p["prompt_len"].get(tid, d["prompt_len"].get(tid, "NA"))
        hit = d["cache_hit"].get(tid, "NA")
        print(f"{tid}\t{plen}\t{hit}")
        plen_dist[plen] += 1
        hit_dist[hit] += 1
    print("prompt_token_len_distribution", dict(plen_dist))
    print("cache_hit_len_distribution", dict(hit_dist))


def diff(a, b):
    return "NA" if a is None or b is None else a - b


def fmt(v):
    if v == "NA" or v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def cmd_p_breakdown(args):
    p, d = scan_log(args.p_log), scan_log(args.d_log)
    ids = complete(p) & complete(d)
    if args.story:
        ids &= p["story"] | d["story"]
    if args.cache_hit is not None:
        ids = {tid for tid in ids if d["cache_hit"].get(tid) == args.cache_hit}
    print("task\ttotal\tfrontend\tdispatch_to_kv\tkv_prepare\tsp_d2d\tembed_wait\tcompute_wall\tinfer_exec_sum\tinfer_exec_count\ttransfer\td2h_post")
    for tid in sorted(ids):
        ev = p["events"].get(tid, {})
        end = ev.get("send_transfer_result") or ev.get("transfer_done")
        row = [
            tid,
            diff(end, ev.get("call")),
            diff(ev.get("driver"), ev.get("call")),
            diff(ev.get("kv_alloc"), ev.get("driver")),
            diff(ev.get("post_alloc"), ev.get("kv_alloc")),
            ev.get("sp_d2d", "NA"),
            diff(ev.get("embed_done"), ev.get("kv_alloc")),
            diff(ev.get("compute_done"), ev.get("post_alloc")),
            ev.get("infer_exec_ms", "NA"),
            ev.get("infer_exec_count", "NA"),
            diff(end, ev.get("compute_done")),
            ev.get("d2h_post", "NA"),
        ]
        print("\t".join(fmt(x) for x in row))


def main():
    ap = argparse.ArgumentParser(description="Analyze llmserver p.log/d.log PD latency breakdowns")
    ap.add_argument("--p-log", required=True)
    ap.add_argument("--d-log", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    sub.add_parser("story-hits")
    br = sub.add_parser("p-breakdown")
    br.add_argument("--story", action="store_true")
    br.add_argument("--cache-hit", type=int)
    args = ap.parse_args()
    {"summary": cmd_summary, "story-hits": cmd_story_hits, "p-breakdown": cmd_p_breakdown}[args.cmd](args)


if __name__ == "__main__":
    main()
