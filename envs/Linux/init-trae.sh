#!/bin/bash

TRAE_DIR="$HOME/.trae"

LinkFile() {
    local source_path="$1"
    local target_path="$2"

    if [ -L "$target_path" ] && [ "$(readlink "$target_path")" = "$source_path" ]; then
        return
    fi

    if [ -e "$target_path" ] || [ -L "$target_path" ]; then
        mv "$target_path" "$target_path.bak"
    fi

    ln -s "$source_path" "$target_path"
}

mkdir -p "$TRAE_DIR"
LinkFile "$SCRIPT_DIR/traecli.yaml" "$TRAE_DIR/traecli.yaml"
