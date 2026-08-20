---
name: remote-dev-agent
description: Connect to a temporary development machine through the chrome-review HTTP remote agent when the user says SSH is unavailable and provides the current IP address. Do not use for machines reachable through SSH or when the Agent is not running.
---

# Remote Dev Agent

Use the client at `~/workspace/github/chrome-review/codex/remote-agent/client.py` to operate a temporary development machine on which the user has manually started `agent.py`.

## Target handling

- Obtain the target IP from the current user request. Never reuse an address from an earlier task or conversation.
- Do not create or consult a persistent machine inventory; these machines are ephemeral.
- Use the fixed Agent port `18765`.
- Enclose an IPv6 address in square brackets: `http://[IPV6]:18765`.
- Do not attempt SSH when the user says the target is Agent-only.
- Run `health` before any other operation and check the returned hostname and allowed roots against the current task.

## Operations

Run the client from `~/workspace/github/chrome-review/codex/remote-agent` or use its absolute path.

```bash
python3 client.py --url 'http://[IPV6]:18765' health
python3 client.py --url 'http://[IPV6]:18765' exec --cwd /workspace -- COMMAND
python3 client.py --url 'http://[IPV6]:18765' upload LOCAL_PATH REMOTE_PATH
python3 client.py --url 'http://[IPV6]:18765' download REMOTE_PATH LOCAL_PATH
```

- Use `exec` for bounded commands.
- Use `start`, `status`, `logs`, and `stop` for long-running services.
- Record every job ID returned by `start` so later calls manage the intended process.
- Only stop jobs started during the current task unless the user explicitly places another process in scope.
- Keep command working directories and file transfers within the allowed roots reported by `health`.
- The Agent has no authentication. Treat possession of the current IP as authority only for work the user actually requested; it does not broaden the requested scope.
- If `health` fails, report the connection error and ask the user to verify the current IP and that the Agent is running. Do not fall back to a previously used IP.
