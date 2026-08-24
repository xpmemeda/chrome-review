---
name: model-artifact-fetch
description: 将缺失的模型文件获取到 ~/workspace/models，优先使用内部 Haruna HDFS；仅当 HDFS 客户端或模型不存在时才回退到 Hugging Face git clone。用于远程模型服务、推理、基准测试或开发任务缺少本地模型文件的情况。
---

# 获取模型文件

在不覆盖现有模型目录的前提下，将用户请求的模型准备到本地。

## 数据源策略

- 本地根目录：`~/workspace/models`
- HDFS 可执行文件：`/opt/tiger/yarn_deploy/hadoop/bin/hdfs`
- HDFS 根目录：`hdfs://haruna/home/byte_device_intelligence_model/xiongpeng.123`
- 除非用户指定其他目录名，否则将 `black-forest-labs/FLUX.2-klein-4B` 等 Hugging Face 仓库映射为本地/HDFS 基名 `FLUX.2-klein-4B`。
- 已存在完整本地目录时立即复用。
- 当 HDFS 可执行文件存在，且对 `<HDFS root>/<model basename>` 执行 `hdfs dfs -test -e` 成功时，优先使用 HDFS。
- 仅当 HDFS 可执行文件不存在或 HDFS 存在性检查显示模型不存在时，才回退到 `https://huggingface.co/<repo-id>`。
- 如果 HDFS 显示模型存在但 `hdfs dfs -get` 失败，停止并报告 HDFS 错误，不要静默回退。

## 工作流程

使用 `scripts/fetch_model.sh REPO_ID`，不要重新拼装下载命令。只有 HDFS/本地基名与 Hugging Face 仓库基名不同时才传入 `--model-name`。

脚本会下载到同级 `.partial.<pid>` 路径，拒绝未解析的 Git LFS 指针文件，并在验证后将目录重命名到正式位置。脚本拒绝覆盖非空或不完整的目标目录。保留失败的临时目录以供诊断或续传；未经用户指示不要删除。

通过网络从 Hugging Face 克隆前，应用 `AGENTS.md` 中的远程机器代理规则。不要在 Skill、命令输出或仓库文件中写入 Hugging Face token、HDFS 凭据、Cookie 或其他秘密。

报告实际选择的数据源、最终路径，以及是否复用了现有本地副本。长时间下载期间及时报告进度和环境故障。
