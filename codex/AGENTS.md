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
