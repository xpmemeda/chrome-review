#!/usr/bin/env bash
set -euo pipefail

HDFS_BIN_DEFAULT="/opt/tiger/yarn_deploy/hadoop/bin/hdfs"
HDFS_ROOT_DEFAULT="hdfs://haruna/home/byte_device_intelligence_model/xiongpeng.123"
MODELS_ROOT_DEFAULT="${HOME}/workspace/models"

usage() {
  echo "Usage: $0 [--model-name NAME] [--models-root PATH] [--hdfs-bin PATH] [--hdfs-root URI] REPO_ID" >&2
}

model_name=""
models_root="$MODELS_ROOT_DEFAULT"
hdfs_bin="$HDFS_BIN_DEFAULT"
hdfs_root="$HDFS_ROOT_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      model_name="${2:?missing value for --model-name}"
      shift 2
      ;;
    --models-root)
      models_root="${2:?missing value for --models-root}"
      shift 2
      ;;
    --hdfs-bin)
      hdfs_bin="${2:?missing value for --hdfs-bin}"
      shift 2
      ;;
    --hdfs-root)
      hdfs_root="${2:?missing value for --hdfs-root}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -n "${repo_id:-}" ]]; then
        echo "Only one REPO_ID may be supplied" >&2
        usage
        exit 2
      fi
      repo_id="$1"
      shift
      ;;
  esac
done

if [[ -z "${repo_id:-}" ]]; then
  usage
  exit 2
fi

if [[ -z "$model_name" ]]; then
  model_name="${repo_id##*/}"
fi
if [[ -z "$model_name" || "$model_name" == "." || "$model_name" == ".." || "$model_name" == */* ]]; then
  echo "Invalid model name: $model_name" >&2
  exit 2
fi

models_root="${models_root/#\~/$HOME}"
destination="${models_root}/${model_name}"
hdfs_uri="${hdfs_root%/}/${model_name}"
partial="${destination}.partial.$$"
locks_dir="${models_root}/.locks"
lock_dir="${locks_dir}/${model_name}.lock"

validate_model_dir() {
  local directory="$1"
  [[ -d "$directory" ]] || return 1
  find "$directory" -type f -print -quit | grep -q . || return 1
  if find "$directory" -type f -size -1024c -exec grep -Il '^version https://git-lfs.github.com/spec/v1' {} + | grep -q .; then
    echo "Unresolved Git LFS pointer found under $directory" >&2
    return 1
  fi
}

mkdir -p "$models_root" "$locks_dir"

if validate_model_dir "$destination"; then
  echo "MODEL_SOURCE=local"
  echo "MODEL_PATH=$destination"
  exit 0
fi
if [[ -e "$destination" ]]; then
  echo "Destination exists but is incomplete; refusing to overwrite: $destination" >&2
  exit 1
fi
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Another download may be active for $model_name: $lock_dir" >&2
  exit 1
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

if [[ -e "$partial" ]]; then
  echo "Partial path already exists: $partial" >&2
  exit 1
fi

if [[ -x "$hdfs_bin" ]] && "$hdfs_bin" dfs -test -e "$hdfs_uri"; then
  mkdir "$partial"
  echo "MODEL_SOURCE=hdfs"
  echo "HDFS_URI=$hdfs_uri"
  if ! "$hdfs_bin" dfs -get "$hdfs_uri" "$partial/"; then
    echo "HDFS model exists but download failed; partial data preserved at $partial" >&2
    exit 1
  fi
  downloaded="${partial}/${model_name}"
  if ! validate_model_dir "$downloaded"; then
    echo "HDFS download failed validation; partial data preserved at $partial" >&2
    exit 1
  fi
  mv "$downloaded" "$destination"
  rmdir "$partial"
else
  command -v git >/dev/null || {
    echo "git is required for Hugging Face fallback" >&2
    exit 1
  }
  hf_url="https://huggingface.co/${repo_id}"
  echo "MODEL_SOURCE=huggingface"
  echo "HUGGINGFACE_URL=$hf_url"
  if ! git clone "$hf_url" "$partial"; then
    echo "Hugging Face clone failed; partial data preserved at $partial" >&2
    exit 1
  fi
  if ! validate_model_dir "$partial"; then
    echo "Hugging Face clone failed validation; partial data preserved at $partial" >&2
    exit 1
  fi
  mv "$partial" "$destination"
fi

echo "MODEL_PATH=$destination"
