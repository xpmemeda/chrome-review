# Release, Debug, RelWithDebInfo, MinSizeRel 分别对应什么编译选项？

- Release: -O3 -DNDEBUG
- Debug: -O0 -g
- RelWithDebInfo: -O2 -g -DNDEBUG
- MinSizeRel: -Os -DNDEBUG

# 常见的 gcc 编译选项

GCC / Clang 命令行是从左到右解析的，同一类选项，通常是后写的覆盖前面的。

- ``-O0``: Disable optimization, mainly used for debugging.
- ``-O1``: Enable light optimizations.
- ``-O2``: Enable standard optimizations, the most commonly used level in production.
- ``-O3``: Enable aggressive optimizations for maximum performance.
- ``-Wall``: Enable most common compiler warnings.
- ``-Werror``: Treat all warnings as errors.
- ``-fPIC``: Generate position-independent code (required for shared libraries).
- ``-g``: Generate debugging information.
- ``-fvisibility=hidden``: Hide symbols by default to reduce ABI exposure.
- ``-fno-omit-frame-pointer``: Preserve the frame pointer for better stack tracing and profiling.
- ``-l<lib>``: Link against a specific library.
- ``-Wl,rpath,<path>``: Embed the runtime library search path (RPATH) into the binary.
