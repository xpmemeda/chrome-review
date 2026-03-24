---
name: cmake-env-build-run
description: >
  Configure, build, test, and run CMake-based projects that keep their required
  environment in a `.env` file and their CMake setup in `CMakeUserPresets.json`.
  Use when Codex needs to compile or run a project and the repository convention
  is: source `.env` first, then use `cmake --preset default` and
  `cmake --build --preset default`, with fallbacks to inspect available presets
  when `default` is missing.
---

# CMake Env Build Run

## Overview

Follow the repository convention instead of inventing a new build flow. Load the project environment from `.env`, use `CMakeUserPresets.json` as the source of truth for configure and build presets, and keep environment setup and the actual command in the same shell invocation.

Do not add extra command-line flags or options to `cmake --preset default` or `cmake --build --preset default`.
If the project needs cache variables, targets, jobs, generators, toolchains, or any other parameters, they must come from `CMakeUserPresets.json`.
Do not improvise additional parameters based on guesswork.

## Workflow

1. Confirm that `.env` and `CMakeUserPresets.json` exist in the project root or in the location implied by the user request.
2. Read `CMakeUserPresets.json` before running commands if you need to confirm preset names, binary directories, generators, cache variables, or test presets.
3. Prefer the `default` preset for configure and build unless the file or the user says otherwise.
4. Run configure, build, test, and executables in commands that begin with `source .env`.

## Command Pattern

Use one shell invocation per operation so the sourced environment is active for that operation.

Examples:

```bash
source .env && cmake --preset default
source .env && cmake --build --preset default
```

Do not assume that sourcing `.env` in one tool call persists into later tool calls.

If the shell may not support `source`, use the shell that does, rather than rewriting the environment setup.

## Configure And Build

Prefer this sequence:

```bash
source .env && cmake --preset default
source .env && cmake --build --preset default
```

Use these commands exactly as written unless `CMakeUserPresets.json` itself requires a different preset name.
Do not append flags such as `-j`, `--target`, `--config`, `-D...`, or other ad-hoc options.

If `default` does not exist:

1. Inspect `CMakeUserPresets.json`.
2. Run `cmake --list-presets` if needed.
3. Pick the most relevant preset instead of guessing.

If the repository uses separate configure and build preset names, follow the file rather than forcing both to be `default`.

## Run And Test

Use the same environment-loading pattern for test and runtime commands:

```bash
source .env && ctest --preset default
source .env && <binary-or-script>
```

When the executable path is unclear, derive it from `CMakeUserPresets.json`, the build tree, or the target the user asked for. Prefer evidence from the project configuration over assumptions.

## Failure Handling

If configure or build fails:

1. Keep the current preset and inspect the exact error first.
2. Check whether `.env` was sourced in the same command invocation.
3. Re-read `CMakeUserPresets.json` for required cache variables, toolchain files, or inherited presets.
4. Only switch presets when the configuration shows that `default` is wrong for the task.

If the user asks only for compilation, stop after a successful build. If the user asks to run the project, identify the correct test, executable, or script and run it with `source .env` in the same command.

If the user explicitly provides a command, follow it exactly instead of "improving" it with extra parameters.
