#!/bin/bash

if [[ "$0" == "$BASH_SOURCE" ]]; then
    echo "Error: This script must be sourced, not executed."
    exit 1
fi

source ${HOME}/workspace/chrome-review/scripts/activate-dirs \
	-d ${HOME}/local/llvm-mlu-19.1.7