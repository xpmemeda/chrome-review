# remote_fleet

One MCP server, three ways to reach a target machine. Pick by network:

| backend | when | discovery | who connects |
|---|---|---|---|
| `ssh` | work machine can SSH to target | `hosts.toml` alias | work → target |
| `agent` | target reachable, work machine has no public IP | `hosts.toml` url | work → `direct_agent.py` |
| `reverse` | work machine has a public IP, target is not reachable | self-registration | `reverse_agent.py` → work |

`direct_agent.py` and `reverse_agent.py` are standalone single files (stdlib
only, Python 3.8+) that serve the same agent API; `client.py` is a CLI for a
direct agent.

```
reverse_agent.py <name> ── long poll ──▶ rendezvous [::]:18765 (public) ┐
direct_agent.py :18765  ◀── HTTP ──────────────────────────────────────┤ mcp_server.py
ssh target              ◀── ssh ───────────────────────────────────────┘ 127.0.0.1:18766/mcp ◀── Codex / Claude
```

## 1. Run the MCP server on the work machine

```bash
python3 -m pip install "mcp<2"       # Python 3.11+; v2 renamed FastMCP
python3 mcp_server.py               # MCP on 127.0.0.1:18766, rendezvous on [::]:18765
```

Options: `--host/--port` for the MCP endpoint, `--rendezvous-host/--rendezvous-port`
for the reverse-agent listener (`::` accepts both IPv6 and IPv4 clients;
`--rendezvous-port 0` disables reverse agents, e.g. on a machine without a
public IP). Open only the rendezvous port in the firewall; keep the MCP port
local.

A systemd unit keeps it running:

```ini
# /etc/systemd/system/remote-fleet.service
[Unit]
Description=remote_fleet MCP server
After=network-online.target

[Service]
WorkingDirectory=/path/to/chrome-review/codex/mcps/remote-fleet
ExecStart=/usr/bin/python3 mcp_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Register it with Codex:

```toml
[mcp_servers.remote_fleet]
url = "http://127.0.0.1:18766/mcp"
startup_timeout_sec = 10
tool_timeout_sec = 300
```

or Claude Code: `claude mcp add --transport http remote_fleet http://127.0.0.1:18766/mcp`.

## 2. Reverse agent (`reverse_agent.py`)

On the target machine:

```bash
python3 reverse_agent.py my-box        # --env merlin is the default
```

- The name (`[A-Za-z0-9][A-Za-z0-9_.-]*`, up to 64 chars) is the `host` used in
  MCP tools. The rendezvous defaults to `http://74.211.111.53:18765`;
  override it with `--server` or `REMOTE_FLEET_SERVER`.
- `--root` is repeatable and defaults to `/`. Command working directories and
  file transfers are confined to the roots; the first root is the host's
  default directory.
- `--env` is one of `merlin` (default), `volcano` or `null` (no env), naming an
  `[envs.<name>]` table in the VPS's
  `hosts.toml`; it is injected when a tool is called with `network=true`.
- The agent honours `HTTP(S)_PROXY`/`NO_PROXY` when dialing the VPS.

The agent long-polls `POST /v1/agents/poll`; each request it receives runs in
its own thread and its response is returned with `POST /v1/agents/result`, so
any number of commands, jobs and transfers can run concurrently. When the
connection drops it reconnects with exponential backoff (max 30 s).

Liveness and naming:

- A device is offline once it has no poll in flight for 15 s. Its queued and
  in-flight requests then fail immediately instead of waiting for timeouts.
- A second agent cannot take a name while the first is online (it keeps
  retrying and logs who holds the name). Names that collide with an ssh host
  or direct agent in `hosts.toml` are rejected the same way.
- Offline devices stay in `list_hosts` (as `backend: reverse`) for 24 hours.
- If the MCP server restarts, agents re-register automatically. Background
  jobs live in the agent process, so `job_ref`s survive an MCP restart but not
  an agent restart. Jobs themselves run in their own session and outlive the
  agent.

There is no authentication: anyone who can reach the rendezvous port can
register a name and receive the requests sent to it. They cannot send
commands, because the MCP endpoint only listens on localhost.

Job logs and an audit trail are written under `~/.local/state/dev-agent/`.

## 3. Direct agent (`direct_agent.py`)

On the target machine:

```bash
python3 direct_agent.py                 # listens on [::]:18765, roots default to /
```

Options: `--host`, `--port`, repeatable `--root`, `--max-upload-gib`,
`--tls-cert/--tls-key`. There is no authentication; restrict the port with a
firewall. Then register it on the work machine:

```toml
[hosts.dev_agent]
backend = "agent"
url = "http://[2408:xxxx:xxxx::1234]:18765"
root = "/"           # must be one of the agent's roots or below one
env = "merlin"
# proxy = "host:port"   # HTTP proxy to reach the agent; direct by default
# insecure = true       # skip TLS verification for a self-signed cert
```

`client.py` talks to a direct agent without MCP:

```bash
export DEV_AGENT_URL='http://[2408:xxxx:xxxx::1234]:18765'
python3 client.py health
python3 client.py exec -- nvidia-smi
python3 client.py start -- 'python serve.py'; python3 client.py logs --follow JOB_ID
python3 client.py upload ./a.py /root/a.py; python3 client.py download /root/out.json ./out.json
```

## 4. ssh hosts and envs

```toml
[envs.merlin]
HTTPS_PROXY = "http://proxy.example:8118"

[hosts.dev_ssh]
backend = "ssh"
ssh_alias = "dev-box"      # defaults to the table name
root = "/"
env = "merlin"             # optional; env_overrides replaces single keys
```

`env` (ssh/agent hosts) and `--env` (reverse agents) select an `[envs.<name>]`
table that is injected when a tool runs with `network=true`. Changes to
`hosts.toml` are picked up before the next tool call.

Both agents print each exec, job and file transfer to the console and keep an
audit trail and job logs under `~/.local/state/dev-agent/`.

## Tools

`list_hosts`, `health`, `inspect_host`, `exec`, `start_job`, `job_status`,
`job_logs`, `stop_job`, `download_file`, `upload_file`, `transfer_status`,
`stop_transfer`, `resume_transfer`.

File transfers are asynchronous and resumable. `download_file`/`upload_file`
return a `transfer_ref`; poll it with `transfer_status`, cancel with
`stop_transfer`, and continue after cancellation, failure or an MCP restart
with `resume_transfer`. Data moves in 8 MiB chunks into
`<destination>.partial`, which is verified against the source prefix before
resuming. On completion size and SHA-256 are checked before an atomic rename.
Existing destinations are only replaced with `overwrite=true`; an optional
`expected_sha256` guards against the wrong source.
