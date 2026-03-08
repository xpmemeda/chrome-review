- Debug模式编译
- 从github下载FlameGraph

```bash
perf record -g <program>  # perf.data
perf script -i perf.data > out.perf  # out.perf
stackcollapse-perf.pl out.perf > out.fold  # out.fold
flamegraph.pl out.fold > perf.svg  # open perf.svg on chrome
```