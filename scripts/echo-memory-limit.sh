#!/usr/bin/env bash

# 统一用这个变量存“当前已知最小限制（GB）”
memorylimit_gb=""

update_min() {
    local val="$1"
    # 跳过空值或非数字
    [[ -z "$val" || ! "$val" =~ ^[0-9]+$ ]] && return
    if [[ -z "$memorylimit_gb" || "$val" -lt "$memorylimit_gb" ]]; then
        memorylimit_gb="$val"
    fi
}

# ---------- cgroup v1 ----------
cgroup_v1="/sys/fs/cgroup/memory/memory.limit_in_bytes"
if [[ -r "$cgroup_v1" ]]; then
    v1_bytes=$(<"$cgroup_v1")
    # cgroup v1 的 "无限大" 通常是 9223372036854771712 之类的超大值，直接忽略
    if [[ "$v1_bytes" =~ ^[0-9]+$ ]] && (( v1_bytes < 9000000000000000000 )); then
        v1_gb=$(( v1_bytes / 1024 / 1024 / 1024 ))
        update_min "$v1_gb"
    fi
fi

# ---------- cgroup v2 ----------
cgroup_v2="/sys/fs/cgroup/memory.max"
if [[ -r "$cgroup_v2" ]]; then
    v2_raw=$(<"$cgroup_v2")
    # v2 里 “max” 表示无限制
    if [[ "$v2_raw" =~ ^[0-9]+$ ]]; then
        v2_bytes="$v2_raw"
        if (( v2_bytes < 9000000000000000000 )); then
            v2_gb=$(( v2_bytes / 1024 / 1024 / 1024 ))
            update_min "$v2_gb"
        fi
    fi
fi

# ---------- 物理内存 ----------
meminfo="/proc/meminfo"
if [[ -r "$meminfo" ]]; then
    # 这里用 MemTotal 更像“真实物理内存容量”
    hw_kb=$(awk '/^MemTotal:/ {print $2}' "$meminfo")
    if [[ "$hw_kb" =~ ^[0-9]+$ ]]; then
        hw_gb=$(( hw_kb / 1024 / 1024 ))
        update_min "$hw_gb"
    fi
fi

# ---------- 输出 ----------
if [[ -n "$memorylimit_gb" ]]; then
    echo "${memorylimit_gb} GB"
else
    echo "unlimited"
fi
