#!/bin/bash

VSCODE_SETTINGS_DIR="$HOME/Library/Application Support/Code/User"

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

mkdir -p "$VSCODE_SETTINGS_DIR"
LinkFile "$SCRIPT_DIR/Darwin/darwin-vscode-setting.json" "$VSCODE_SETTINGS_DIR/settings.json"
