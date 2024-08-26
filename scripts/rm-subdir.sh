#!/bin/bash

# 检查是否提供了目录参数
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# 获取要处理的目录
TARGET_DIR="$1"

# 使用 find 命令查找并删除所有 __pycache__ 目录
find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -rf {} +

echo "All __pycache__ directories have been removed from $TARGET_DIR."