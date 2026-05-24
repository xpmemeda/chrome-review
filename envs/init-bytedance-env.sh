# proxy
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8"

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
