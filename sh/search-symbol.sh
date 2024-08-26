#!/bin/bash

SYMBOL=""
LIB_DIR=""

while getopts "s:d:" opt; do
  case $opt in
    s)
      SYMBOL=$OPTARG
      ;;
    d)
      LIB_DIR=$OPTARG
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
    echo "Usage: $0 -s <symbol> -d <search dir>"
    exit 1
fi
if [[ $LIB_DIR == "" ]]; then
    echo "Usage: $0 -s <symbol> -d <search dir>"
    exit 1
fi

for lib in "$LIB_DIR"/*; do
    if nm -g "$lib" 2>/dev/null | grep -q "$SYMBOL"; then
        echo "Symbol $SYMBOL found in $lib"
    fi
done
