#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "用法: $0 --version <版本> [--prefix <安装目录>]" >&2
    echo "示例: $0 --version 24.21.0" >&2
}

version=""
prefix=""
while (($#)); do
    case "$1" in
        --version|-v)
            (($# >= 2)) || { usage; exit 2; }
            version="$2"
            shift 2
            ;;
        --prefix)
            (($# >= 2)) || { usage; exit 2; }
            prefix="$2"
            shift 2
            ;;
        *) usage; exit 2 ;;
    esac
done

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { usage; exit 2; }
prefix="${prefix:-$HOME/local/node-$version}"
[[ "$prefix" = /* ]] || { echo "安装目录必须是绝对路径: $prefix" >&2; exit 2; }

case "$(uname -s):$(uname -m)" in
    Linux:x86_64) platform=linux-x64 ;;
    Linux:aarch64|Linux:arm64) platform=linux-arm64 ;;
    Darwin:x86_64) platform=darwin-x64 ;;
    Darwin:arm64) platform=darwin-arm64 ;;
    *) echo "不支持的系统或架构: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac

if [[ -e "$prefix" ]]; then
    if [[ -x "$prefix/bin/node" ]] && [[ "$("$prefix/bin/node" --version)" == "v$version" ]]; then
        echo "已安装: $prefix"
        exit 0
    fi
    echo "目标目录已存在，未覆盖: $prefix" >&2
    exit 1
fi

command -v curl >/dev/null || { echo "需要 curl" >&2; exit 1; }
command -v tar >/dev/null || { echo "需要 tar" >&2; exit 1; }
if command -v sha256sum >/dev/null; then
    checksum=(sha256sum)
elif command -v shasum >/dev/null; then
    checksum=(shasum -a 256)
else
    echo "需要 sha256sum 或 shasum" >&2
    exit 1
fi

archive="node-v$version-$platform.tar.xz"
base_url="https://nodejs.org/dist/v$version"
parent="$(dirname "$prefix")"
mkdir -p "$parent"
temp_dir="$(mktemp -d "$parent/.node-$version.XXXXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT

curl -fLsS --retry 3 "$base_url/$archive" -o "$temp_dir/$archive"
curl -fLsS --retry 3 "$base_url/SHASUMS256.txt" -o "$temp_dir/SHASUMS256.txt"
expected="$(awk -v name="$archive" '$2 == name { print $1 }' "$temp_dir/SHASUMS256.txt")"
[[ -n "$expected" ]] || { echo "校验文件中未找到 $archive" >&2; exit 1; }
actual="$("${checksum[@]}" "$temp_dir/$archive" | awk '{ print $1 }')"
[[ "$actual" == "$expected" ]] || { echo "SHA-256 校验失败" >&2; exit 1; }

mkdir "$temp_dir/extracted"
tar -xJf "$temp_dir/$archive" -C "$temp_dir/extracted" --strip-components=1
"$temp_dir/extracted/bin/node" --version
mv "$temp_dir/extracted" "$prefix"

echo "安装完成: $prefix"
echo "将 $prefix/bin 加入 PATH 后即可使用 node、npm 和 npx。"
