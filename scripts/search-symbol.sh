#!/bin/bash

# T 该符号位于代码段（.text section），表示它是一个全局函数或已初始化的静态函数。
# U 该符号在当前文件中未定义（需要在其他目标文件或库中解析）。
# W 弱符号。允许同名符号存在，且不会导致链接错误；最终会优先选择非弱符号。
# B	符号位于未初始化数据段（.bss section），通常是未初始化的全局变量或静态变量。
# C	公共符号（Common symbol），类似 B，但可能未被分配具体地址（需链接器处理合并）。
# D	符号位于已初始化数据段（.data section），通常是已初始化的全局变量或静态变量。
# R	符号位于只读数据段（.rodata section），例如字符串常量或 const 修饰的全局变量。
# S	符号位于特殊未初始化段（Small common symbols），优化后的小型公共符号。
# V	弱对象符号（Weak object），类似于 W，但针对变量而非函数。
# t	小写的 t 表示局部函数（static function）的代码段符号。

SYMBOL=""
LIB_DIR=""
LIB_PATH=""

while getopts "s:d:f:" opt; do
  case $opt in
    s)
      SYMBOL=$OPTARG
      ;;
    d)
      LIB_DIR=$OPTARG
      ;;
    f)
      LIB_PATH=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ $SYMBOL == "" ]]; then
    echo "Usage: $0 -s <symbol> [ -d <search dir> ] [ -f <search file> ]"
    exit 1
fi
if [[ $LIB_DIR == "" && $LIB_PATH == "" ]]; then
    echo "Usage: $0 -s <symbol> [ -d <search dir> ] [ -f <search file> ]"
    exit 1
fi

# -A Print name of the input file before every symbol
# -g Display only external symbols
# -D Display dynamic symbols instead of normal symbols

if [[ $LIB_DIR != "" ]]; then
  for lib in "$LIB_DIR"/*; do
    if nm -AgD "$lib" 2>/dev/null | grep -q "$SYMBOL"; then
      r=$(nm -AgD "$lib" | grep --color=always "$SYMBOL")
      echo "$r"
    fi
    if nm -Ag "$lib" 2>/dev/null | grep -q "$SYMBOL"; then
      r=$(nm -Ag "$lib" | grep --color=always "$SYMBOL")
      echo "$r"
    fi
  done
fi

if [[ $LIB_PATH != "" ]]; then
  if nm -AgD "$LIB_PATH" 2>/dev/null | grep -q "$SYMBOL"; then
    r=$(nm -AgD "$LIB_PATH" | grep --color=always "$SYMBOL")
    echo "$r"
  fi
  if nm -Ag "$LIB_PATH" 2>/dev/null | grep -q "$SYMBOL"; then
    r=$(nm -Ag "$LIB_PATH" | grep --color=always "$SYMBOL")
    echo "$r"
  fi
fi
