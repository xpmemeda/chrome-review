# Remote development agent

This directory contains a small HTTP agent for development
machines that are reachable from a Mac but cannot be accessed with SSH. It uses
only the Python standard library.

## 1. Start the agent on the development machine

On the development machine:

```bash
python3 agent.py \
  --root ~/workspace \
  --root ~/workspace/models
```

The agent listens on all IPv6 interfaces (`::`) using the fixed port `18765`.
To bind one specific address, pass `--host '<ipv6-address>'`. To use IPv4
instead, pass `--host 0.0.0.0`.

`--root` is repeatable. Commands may only use a working directory below an
allowed root, and file uploads/downloads are confined to those roots. Relative
paths are resolved below the first root.

Find the machine IP with `hostname -I` or the company's machine information
page. Ensure the host firewall only permits the Mac or trusted office network.

The agent intentionally has no authentication. Anyone who can reach port
`18765` can execute commands with the permissions of the agent process. Use it
only on a disposable debugging machine and restrict inbound access with the
host firewall. TLS can encrypt traffic but does not add authentication here.

## 2. Use it from the Mac

```bash
export DEV_AGENT_URL='http://[2408:xxxx:xxxx::1234]:18765'

python3 client.py health
python3 client.py exec --cwd /home/user/workspace -- nvidia-smi
python3 client.py upload ./server.py /home/user/workspace/server.py
python3 client.py download /home/user/workspace/result.json ./result.json
```

Multiple command arguments are shell-quoted by the client before being sent to
the agent, so argument boundaries are preserved. For commands that intentionally
use shell syntax such as pipes or redirections, pass the complete command as one
quoted argument.

Start and manage a long-running service:

```bash
python3 client.py start --cwd /home/user/workspace -- \
  'CUDA_VISIBLE_DEVICES=0 python flask-diffsynth.py --model /models/FLUX.2'

python3 client.py status JOB_ID
python3 client.py logs --follow JOB_ID
python3 client.py stop JOB_ID
```

The agent writes job logs and an audit trail under
`~/.local/state/dev-agent/`. Background jobs live only in the agent process's
in-memory registry, so keep the agent running while managing them.

## 3. Unified MCP server

`mcp_server.py` presents SSH and HTTP-agent machines through one local MCP
server. Each machine is declared in a TOML registry. The preferred `ssh`
backend uses an explicit hostname, user, port, local identity file, and
known-hosts file. The legacy `ssh_archive` backend reads an SSH config and
private key from a zip archive, expands it into a process-private temporary
directory, and removes it when the MCP process exits. Both SSH backends can
connect through an HTTP CONNECT proxy.

To register a machine running `agent.py`, add an `agent` entry. Its `root`
must be one of the roots allowed by the agent (or a directory below one):

```toml
[hosts.dev_agent]
backend = "agent"
url = "http://[2408:xxxx:xxxx::1234]:18765"
root = "/home/user/workspace"
```

For HTTPS with a development certificate, `insecure = true` disables
certificate verification. Do not use it for a trusted production endpoint.
Agent connections are direct by default; set `proxy = "host:port"` when an
HTTP proxy is required.
The agent backend supports the same health, command, job, log, stop, upload,
and download MCP tools as the SSH backends. Agent jobs remain addressable only
while the same `agent.py` process is running.

MCP file transfers are asynchronous. `download_file` and `upload_file` return a
`transfer_ref` immediately; poll it with `transfer_status`, or request
cancellation with `stop_transfer`. Use `resume_transfer` after cancellation,
failure, or an MCP process restart; it creates a new transfer reference and
continues from the preserved partial file. Transfers have no overall timeout,
but each chunk request has a bounded timeout so a stalled connection can be
stopped or retried. Transfers use 8 MiB chunks instead of loading the whole file into MCP memory. Data is written
to `<destination>.partial`. A later call with the same source and destination
verifies that partial file against the source prefix and resumes at its current
size. Once complete, remote/local size and SHA-256 are verified before an
atomic rename to the requested destination. Existing destinations are not
overwritten unless `overwrite=true` is passed explicitly. An optional
`expected_sha256` rejects the transfer before finalization when the selected
source artifact is not the expected one.
Only one active transfer may write a given destination in an MCP process. The
agent also serializes writes to each partial path. The agent has no upload-size
limit by default; set `--max-upload-gib` to a positive value to configure one.
The standalone `client.py upload` and `client.py download` commands use the
same chunk and partial-file protocol.

```bash
python3 -m pip install -r requirements-mcp.txt
python3 mcp_server.py --check-host xzb
```

Register it with Codex:

```toml
[mcp_servers.remote_fleet]
command = "/usr/bin/python3"
args = [
  "/absolute/path/to/mcp_server.py",
  "--config",
  "/absolute/path/to/hosts.toml",
]
cwd = "/absolute/path/to/remote-agent"
startup_timeout_sec = 10
tool_timeout_sec = 300
enabled = true
required = false
```

The MCP tools cover discovery, health checks, short commands, persistent jobs,
incremental logs, stopping jobs, and resumable asynchronous file transfer. Set `network=true`
for a command or job when the host's configured proxy environment is needed.
The host's `root` is both the default command working directory and the file
transfer boundary. File transfers have no configured size or suffix limit.
Changes to `hosts.toml` are detected before the next tool call. The server
reloads the registry and clears cached SSH sessions automatically, so host
changes do not require an MCP restart.

## Security boundaries

- Run the agent as an unprivileged user; never run it with `sudo`.
- Restrict `--root` to the smallest useful directories.
- Restrict inbound access with the host firewall.
- The agent intentionally supports arbitrary shell commands for automation.
  Anyone who can connect to its port has the same effective permissions as the
  agent user.
