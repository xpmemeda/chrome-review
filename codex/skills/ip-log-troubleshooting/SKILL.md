---
name: ip-log-troubleshooting
description: Fetch and analyze service logs from a user-provided IP address to troubleshoot llmserver, kvcache, or related inference service issues. Use when the user asks Codex to "look at/check/debug/investigate logs" for a specific IP, machine, host, IPv6 address, or online service instance.
---

# Ip Log Troubleshooting

## Overview

Use this skill to start IP-based incident investigation from the remote Bernard stdout service log,
then inspect local service source only when the log evidence requires code context.

## Quick Workflow

1. Identify the target IP address from the user's request. If the user provides multiple IPs, handle them one at a time and label findings by IP.
2. Determine the log date from the user's request. If no date is given, use the current local date in `YYYYMMDD` format.
3. Fetch the service log first with `wget --no-proxy`, saving it under a clear local filename before analysis.
4. Scan the downloaded log for explicit failures, stack traces, `ERROR`, `WARNING`, request IDs, task IDs, timeout signals, OOM/resource messages, retry loops, and abnormal latency.
5. Correlate nearby lines by timestamp, request ID, task ID, rank, worker, or component name before drawing conclusions.
6. Inspect source code only after the log points to a component, function, or behavior that needs interpretation.
7. Ask the user for additional code paths, logs, metrics, or a different timestamp only when the current log and known paths are insufficient.

## Fetching Logs

The Bernard stdout log URL pattern is:

```bash
wget --no-proxy 'http://[<ipv6>]:9320/opt/tiger/toutiao/log/run/bernard_stdout_log.<yyyymmdd>-0000'
```

If the user gives a bracketed IPv6 address or full URL, preserve that form. If the user gives a raw IPv6 literal with colons, wrap it in `[` and `]` inside the URL. If the user gives an IPv4 address or hostname, omit the brackets:

```bash
wget --no-proxy 'http://<ipv4-or-hostname>:9320/opt/tiger/toutiao/log/run/bernard_stdout_log.<yyyymmdd>-0000'
```

Use the current date instead of hard-coding examples. For example, on 2026-08-14 the suffix is `20260814-0000`.

Prefer saving the file with the IP and date in the name, such as `logs/<ip>_bernard_stdout_log.<yyyymmdd>-0000`, when a project-local log directory is appropriate. Do not use a proxy for this fetch.

## Known Source Paths

Use these paths when log interpretation needs code context:

- llmserver: `~/workspace/byted/seed/llmserver`
- kvcache: `~/workspace/byted/data/kvcache`

If investigation requires other repositories, generated configs, deployment metadata, or non-Bernard logs, ask the user for the path or artifact.

## Analysis Guidance

Start from concrete log evidence. Quote short relevant snippets or summarize line patterns, then state what each finding implies. Prefer hypotheses with confidence labels when the evidence is partial.

Useful searches include:

- `rg -n "ERROR|Error|Exception|Traceback|WARNING|timeout|Timeout|OOM|Killed|failed|Failed|retry|Retry" <log>`
- `rg -n "<request_id>|<task_id>|<component>|<timestamp>" <log>`
- `rg -n "<function_or_log_message>" ~/workspace/byted/seed/llmserver ~/workspace/byted/data/kvcache`

When reporting back:

- Lead with the most likely root cause or the current strongest finding.
- Include the fetched log path and date used.
- Mention the exact IDs or timestamps that support the conclusion.
- Separate confirmed facts from likely explanations and next data needed.
