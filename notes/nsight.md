# 给长时间运行的 Server 采样 nsight system

```bash
nsys launch --trace=cublas,cuda,cudnn,nvtx --trace-fork-before-exec=true --cuda-graph-trace=node --show-output=true --session-new mysess $SERVER_CMD
```

```bash
nsys start --force-overwrite=true --stats=true --session mysess && sleep 5 && nsys stop --session mysess
```