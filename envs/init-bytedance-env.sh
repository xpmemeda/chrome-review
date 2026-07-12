if env | grep -q '^[^=]*MERLIN[^=]*='; then
    export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
    export http_proxy="${HTTP_PROXY}"
    export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy="${HTTPS_PROXY}"
    export NO_PROXY="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,2605::/16"
    export no_proxy="${NO_PROXY}"
fi

if [ "$(hostname)" = "di-20260508172227-p44n9" ]; then
    export HTTP_PROXY="http://100.66.18.103:3128"
    export http_proxy=$HTTP_PROXY
    export HTTPS_PROXY="http://100.66.18.103:3128"
    export https_proxy=$HTTPS_PROXY
    export NO_PROXY="localhost,127.0.0.1,mirrors.ivolces.com,pypi.org,files.pythonhosted.org,pypi.python.org"
    export PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple
    # export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    export LESSCHARSET=utf-8
fi

# vibe coding cli, raecli.yaml symlink setup
export PATH=~/.local/bin:$PATH
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_TRAECLI_YAML_SRC="$_SCRIPT_DIR/traecli.yaml"
_TRAECLI_YAML_DST="$HOME/.trae/traecli.yaml"
if [ ! -e "$_TRAECLI_YAML_DST" ] && [ ! -L "$_TRAECLI_YAML_DST" ]; then
    mkdir -p "$HOME/.trae"
    ln -s "$_TRAECLI_YAML_SRC" "$_TRAECLI_YAML_DST"
elif [ -L "$_TRAECLI_YAML_DST" ]; then
    _TRAECLI_YAML_DST_TARGET="$(readlink "$_TRAECLI_YAML_DST")"
    if [ "$_TRAECLI_YAML_DST_TARGET" != "$_TRAECLI_YAML_SRC" ]; then
        echo "[WARNING] ~/.trae/traecli.yaml already exists but points to '$_TRAECLI_YAML_DST_TARGET', not '$_TRAECLI_YAML_SRC'." >&2
    fi
else
    echo "[WARNING] ~/.trae/traecli.yaml already exists and is not a symlink to '$_TRAECLI_YAML_SRC'." >&2
fi
unset _SCRIPT_DIR _TRAECLI_YAML_SRC _TRAECLI_YAML_DST _TRAECLI_YAML_DST_TARGET
