#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")
CODEX_DIR="$HOME/.codex"
CODEX_SKILLS_DIR="$CODEX_DIR/skills"

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

mkdir -p "$CODEX_DIR"
mkdir -p "$CODEX_SKILLS_DIR"

LinkFile "$REPO_DIR/codex/hooks.json" "$CODEX_DIR/hooks.json"

for skill_dir in "$REPO_DIR"/codex/skills/*; do
    if [ ! -f "$skill_dir/SKILL.md" ]; then
        continue
    fi

    LinkFile "$skill_dir" "$CODEX_SKILLS_DIR/$(basename "$skill_dir")"
done
