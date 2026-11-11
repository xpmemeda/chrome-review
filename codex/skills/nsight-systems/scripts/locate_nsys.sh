#!/bin/sh
#######################################
# Print the path of the Nsys CLI executable this machine should use.
#
# Checks NSYS_PATH first, then PATH, then the newest runnable installation in
# known Linux locations, with /usr/local/cuda/bin/nsys as the final fallback.
# README.md defines the shared contract; errors.txt supplies failure diagnostics.
# Globals:
#   Reads NSYS_PATH as the directory containing the Nsys CLI executable.
#   Reads PATH for executable lookup and HOME for home installation discovery.
#   Uses process-local discovery and diagnostic variables.
# Arguments:
#   None.
# Outputs:
#   On success, writes the bare executable path and a newline to standard output.
#   On failure, writes an ERROR: diagnostic to standard error.
#   Suppresses output from candidate executables' --version checks.
# Returns:
#   Exits with status 0 when a runnable executable is found; 1 on resolution failure.
# Examples:
#   sh locate_nsys.sh
#   Prints /opt/nvidia/nsight-systems/2026.3.1/target-linux-x64/nsys on success.
#######################################

set -eu

#######################################
# Read one shared diagnostic segment without evaluating file content as shell code.
# Globals:
#   Writes error_segment; uses here, cr, and line as scratch variables.
# Arguments:
#   1: Exact diagnostic segment key defined in errors.txt.
# Outputs:
#   None.
# Returns:
#   0 when found; 1 when errors.txt or the key is missing.
#######################################
read_error_segment() {
    error_segment=
    here=$(cd "$(dirname "$0")" && pwd -P)
    [ -f "$here/errors.txt" ] || return 1
    cr=$(printf '\r')
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%"$cr"}
        case $line in
            "$1="*)
                error_segment=${line#*=}
                return 0
                ;;
        esac
    done < "$here/errors.txt"
    return 1
}

#######################################
# Print a shared diagnostic with values between its segments and stop the script.
# Globals:
#   Reads diagnostics through read_error_segment.
#   Uses key, fail_out, fail_i, and fail_val as scratch variables.
# Arguments:
#   1: Trusted diagnostic key defined in errors.txt.
#   2 and following: Values inserted between consecutive diagnostic segments.
# Outputs:
#   Writes one diagnostic line to standard error.
# Returns:
#   Does not return; exits the script with status 1.
# Examples:
#   fail err_root
#   fail err_arch_unsupported mips64
#######################################
fail() {
    key=$1
    shift
    if read_error_segment "$key"; then
        fail_out=$error_segment
    else
        # The key names the failure whenever errors.txt is unavailable.
        printf 'ERROR: %s%s\n' "$key" "${1:+ $1}" >&2
        exit 1
    fi
    fail_i=0
    for fail_val in "$@"; do
        fail_i=$((fail_i + 1))
        read_error_segment "${key}_${fail_i}" || error_segment=
        fail_out=$fail_out$fail_val$error_segment
    done
    printf '%s\n' "$fail_out" >&2
    exit 1
}

#######################################
# Confirm that the Nsys CLI executable can run on this machine.
# Globals:
#   None.
# Arguments:
#   1: Path to the candidate Nsys CLI executable.
# Outputs:
#   None; the version command's output is suppressed.
# Returns:
#   0 for an executable file whose version command succeeds; non-zero otherwise.
#######################################
validate_executable() {
    # Execute permissions alone do not prove the binary can load on this machine.
    [ -f "$1" ] && [ -x "$1" ] && "$1" --version >/dev/null 2>&1
}

# 1. NSYS_PATH names the directory containing the Nsys CLI executable. It is set deliberately,
#    so it answers on its own, once it is confirmed runnable. One trailing slash is
#    dropped, which is what a shell completing a directory name leaves behind.
if [ -n "${NSYS_PATH:-}" ]; then
    named=${NSYS_PATH%/}/nsys
    [ -f "$named" ] || fail err_nsys_missing "$named"
    validate_executable "$named" || fail err_nsys_not_runnable "$named"
    printf '%s\n' "$named"
    exit 0
fi

# 2. The Nsys CLI executable already on PATH, which is what the machine is configured to
#    use. command -v prints its path, but the binary may still fail to load or run.
on_path=$(command -v nsys 2>/dev/null || true)
if [ -n "$on_path" ] && validate_executable "$on_path"; then
    printf '%s\n' "$on_path"
    exit 0
fi

# The rest builds a path of its own, so from here the target tree has to be named. The
# running machine names it, and an unknown arch is an error.
arch=$(uname -m 2>/dev/null || true)
if [ -z "$arch" ]; then
    fail err_arch
fi
case $arch in
    x86_64|amd64)  platforms=linux-x64 ;;
    aarch64|arm64) platforms=linux-sbsa-armv8 ;;
    *)             fail err_arch_unsupported "$arch" ;;
esac

#######################################
# Pad a version component's leading digits with zeros to a fixed width.
# Globals:
#   Writes pad_out with the result, keeping the rightmost digits if necessary.
# Arguments:
#   1: Desired width in digits; callers use 4 or 5.
#   2: Version component; an empty value or non-digit suffix is allowed.
# Outputs:
#   None.
# Returns:
#   0; the result is in pad_out.
# Examples:
#   pad 4 10     sets pad_out to 0010.
#   pad 4 3rc1   sets pad_out to 0003.
#   pad 4 ""     sets pad_out to 0000.
#######################################
pad() {
    # String padding avoids printf treating leading zeros as octal prefixes.
    pad_out=00000${2%%[!0-9]*}
    while [ "${#pad_out}" -gt "$1" ]; do
        pad_out=${pad_out#?}
    done
}

#######################################
# Build a fixed-width version key that preserves numeric version ordering.
# Globals:
#   Reads IFS and restores it after splitting the version.
#   Writes key and pad_out; uses saved_ifs as a scratch variable.
#   Enables pathname expansion before returning.
# Arguments:
#   1: Nsight Systems version with one to four dot-separated components.
# Outputs:
#   None.
# Returns:
#   0; the result is in key.
# Examples:
#   version_key 2026.9.2     sets key to 20260009000200000.
#   version_key 2026.10.1    sets key to 20260010000100000.
#   version_key 2026.3       sets key to 20260003000000000.
#   version_key 2024.7.1.84  sets key to 20240007000100084.
#######################################
version_key() {
    # skill_pack_paths.py version_key is the Python equivalent.
    saved_ifs=$IFS
    IFS=.
    set -f
    # shellcheck disable=SC2086
    set -- $1
    IFS=$saved_ifs
    set +f
    pad 4 "${1:-}"; key=$pad_out
    pad 4 "${2:-}"; key=$key$pad_out
    pad 4 "${3:-}"; key=$key$pad_out
    pad 5 "${4:-}"; key=$key$pad_out
}

#######################################
# Find a runnable Nsys CLI executable in an installation's target directories.
# Globals:
#   Reads platforms in preference order.
#   Writes found_platform; uses vr_platform as a scratch variable.
# Arguments:
#   1: Nsight Systems installation root directory.
# Outputs:
#   None.
# Returns:
#   0 with found_platform set to the first runnable target; 1 with it empty.
#######################################
validate_root() {
    found_platform=
    for vr_platform in $platforms; do
        if validate_executable "$1/target-$vr_platform/nsys"; then
            found_platform=$vr_platform
            return 0
        fi
    done
    return 1
}

#######################################
# Select the newest runnable installation, keeping the earlier root on a tie.
# Globals:
#   Reads platforms.
#   Writes best_key, best_root, and best_platform, clearing them if none qualify.
#   Uses dir and version as scratch variables; called helpers' global effects apply.
# Arguments:
#   1 and following: Candidate installation roots in tie-breaking preference order.
# Outputs:
#   None.
# Returns:
#   0; best_root and best_platform are empty if no installation qualifies.
# Examples:
#   select_root /opt/nvidia/nsight-systems/* /opt/nvidia/nsight-systems-cli/*
#######################################
select_root() {
    best_key= best_root= best_platform=
    for dir do
        [ -d "$dir" ] || continue
        version=${dir##*/}
        # Rank CUDA Toolkit copies by their Nsight Systems version, not CUDA's version.
        case $version in
            nsight-systems-*)     version=${version#nsight-systems-} ;;
            NsightSystems-cli-*) version=${version#NsightSystems-cli-} ;;
            NsightSystems-*)      version=${version#NsightSystems-} ;;
        esac
        case $version in [0-9]*) ;; *) continue ;; esac
        validate_root "$dir" || continue
        version_key "$version"
        if [ -z "$best_key" ] || [ "$key" -gt "$best_key" ]; then
            best_key=$key best_root=$dir best_platform=$found_platform
        fi
    done
}

# 3. The newest installation on this machine, including home and CUDA Toolkit installations.
#    At the same version, prefer the system full install, then the home full install,
#    then the standalone CLI-only install, then the CUDA Toolkit copies.
set -- /opt/nvidia/nsight-systems/*
if [ -n "${HOME:-}" ]; then
    set -- "$@" "$HOME"/nsight-systems-*
fi
select_root "$@" /opt/nvidia/nsight-systems-cli/* \
    /usr/local/cuda-*/nsight-systems-* \
    /usr/local/cuda-*/NsightSystems-[0-9]* /usr/local/cuda-*/NsightSystems-cli-*

if [ -n "$best_root" ]; then
    # validate_root already proved the Nsys CLI executable runs, so this names it.
    printf '%s\n' "$best_root/target-$best_platform/nsys"
    exit 0
fi

# 4. CUDA Toolkit may expose the Nsys CLI executable through its active bin directory
#    without a discoverable versioned installation under the standard parents.
if validate_executable /usr/local/cuda/bin/nsys; then
    printf '%s\n' /usr/local/cuda/bin/nsys
    exit 0
fi

fail err_root
