# Remote Development Environment

When working on a remote development machine, detect the machine type and set
the corresponding proxy environment variables before running commands that
access the network.

Most network connections require the proxy, so use the configured proxy by
default. Some addresses may be unreachable through the proxy. If a network
command fails and the proxy may be the cause, retry that command once without
the proxy by clearing `HTTP_PROXY`, `http_proxy`, `HTTPS_PROXY`, `https_proxy`,
`ALL_PROXY`, and `all_proxy` only for that command. Do not permanently unset the
proxy variables for the rest of the remote shell.

## MERLIN

Treat the remote machine as a MERLIN machine when this command succeeds:

```bash
env | grep -q '^[^=]*MERLIN[^=]*='
```

Set the following variables in the same remote shell that runs subsequent
commands:

```bash
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy="${HTTP_PROXY}"
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy="${HTTPS_PROXY}"
export NO_PROXY="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,2605::/16"
export no_proxy="${NO_PROXY}"
```

## Volcano Engine

Treat the remote machine as a Volcano Engine machine when its hostname starts
with `di-`:

```bash
[[ "$(hostname)" == di-* ]]
```

Set the following variables in the same remote shell that runs subsequent
commands:

```bash
export HTTP_PROXY="http://100.66.18.103:3128"
export http_proxy=$HTTP_PROXY
export HTTPS_PROXY="http://100.66.18.103:3128"
export https_proxy=$HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1,mirrors.ivolces.com,pypi.org,files.pythonhosted.org,pypi.python.org"
export PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple
```

# Codex Skills

When creating or updating a user-specific Codex skill:

- Store the canonical skill source under `~/workspace/github/chrome-review/codex/skills/<skill-name>`.
- Expose the skill to Codex with a symbolic link at `~/.codex/skills/<skill-name>`.
- Treat the copy in the `chrome-review` repository as the single source of truth; do not keep a separate copied version under `~/.codex/skills`.
- Before creating the symbolic link, inspect any existing destination. Do not overwrite a real directory or an unrelated symbolic link without user confirmation.
- Use an absolute path as the symbolic-link target so skill discovery does not depend on the current working directory.

# Temporary Development Machines

When the user provides an IP address for a temporary development machine and
says the HTTP remote Agent is running, use the `remote-dev-agent` skill.

- Treat the target address as ephemeral and obtain it from the current request.
- Never reuse a temporary-machine address from an earlier task or conversation.
- Do not maintain a persistent inventory of temporary machines.
- Use the Agent's fixed port `18765` and run its health check before other operations.
