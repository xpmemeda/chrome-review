if env | grep -q '^[^=]*MERLIN[^=]*='; then
    export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
    export http_proxy="${HTTP_PROXY}"
    export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy="${HTTPS_PROXY}"
    export NO_PROXY="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,2605::/16"
    export no_proxy="${NO_PROXY}"
fi

if [[ "$(hostname)" == di-* ]]; then
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
