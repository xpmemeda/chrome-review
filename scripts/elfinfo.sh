#!/bin/bash

# readelf
#
# -h: Display the ELF file header
# -S: Display the sections' header
# -d: Display the dynamic section (if present)

# grep
#
# -E: PATTERNS are extended regular expression
# -q: suppress all normal output

# elf.sections
#
# .text             代码段（机器指令）
# .rodata           只读数据段（常量）
# .data             已初始化的全局变量
# .bss              未初始化的全局变量（不占磁盘）
# .dynsym           动态符号表（给动态链接器使用）
# .symtab           完整符号表，包含所有函数、局部符号、静态函数。没有此节说明被 release strip
# .dynamic          动态链接配置表，包含 NEEDED RPATH RUNPATH SONAME
# .plt              函数跳转跳板（延迟绑定）（Procedure Linkage Table）
# .got              全局地址跳转表（Global Offset Table）
# .eh_frame         异常展开表（Unwind 信息）
# .gnu.version_r    本文件依赖了哪些版本（verneed 表）
# .debug_info       DWARF（一套调试系统/约定格式） 主调试信息，变量、结构、类型、行号映射的核心数据库
#                   此外还有 DWARF 还有 .debug_abbrev .debug_line .debug_str .debug_ranges 这些节

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <elf-file> [more-elf-files...]" >&2
  exit 1
fi

print_elf_info() {
  local file="$1"

  echo "============================================================"
  echo "File: $file"

  if ! [[ -f "$file" ]]; then
    echo "  [!] Not a regular file."
    return
  fi

  # 先粗略检测是不是 ELF
  if ! readelf -h "$file" >/dev/null 2>&1; then
    echo "  [!] Not an ELF file (or readelf can't parse it)."
    return
  fi

  echo "------------------------------------------------------------"
  echo "[ELF Header]"
  # Class / Data / OS/ABI / Type / Machine / Entry
  readelf -h "$file" 2>/dev/null | \
    grep -E 'Class:|Data:|OS/ABI:|Type:|Machine:|Entry point address:' | \
    sed 's/^/  /'

  echo
  echo "------------------------------------------------------------"
  echo "[Dynamic Section]"
  # NEEDED / SONAME / RPATH / RUNPATH
  readelf -d "$file" 2>/dev/null | \
    grep -E 'NEEDED|SONAME|RPATH|RUNPATH' | \
    sed 's/^/  /' || echo "  (no dynamic section or no NEEDED/SONAME/RPATH/RUNPATH)"

  echo
  echo "------------------------------------------------------------"
  echo "[Sections Overview]"
  # 列出常见 section 是否存在
  local sections=(".text" ".data" ".bss" ".rodata" ".plt" ".got" ".dynsym" ".symtab" ".dynamic" ".eh_frame" ".gnu.version" ".gnu.version_r" .debug_info)
  for s in "${sections[@]}"; do
    if readelf -S "$file" 2>/dev/null | grep -q " $s"; then
      echo "  [x] $s"
    else
      echo "  [ ] $s"
    fi
  done

  echo
}

for f in "$@"; do
  print_elf_info "$f"
done
