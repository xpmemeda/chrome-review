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

## Security boundaries

- Run the agent as an unprivileged user; never run it with `sudo`.
- Restrict `--root` to the smallest useful directories.
- Restrict inbound access with the host firewall.
- The agent intentionally supports arbitrary shell commands for automation.
  Anyone who can connect to its port has the same effective permissions as the
  agent user.
