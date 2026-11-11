# NVIDIA Nsight Systems 官方 Skill

从 NVIDIA 官方 Nsight Systems CLI 2026.6.1.218 安装包提取完整的 `skills/nsight-systems/` 目录，安装日期为 2026-11-08。

## 来源与本地调整

- [官方安装说明](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#installing-the-ai-skill)
- [官方软件包](https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2026.6.1.218-39266636.deb)
- [官方软件包索引](https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/Packages.gz)
- 包 SHA-256：`e71440978244e3883fc769da91a44a03451129ce21988d0db656a6bf7c1cb56b`，已与软件包索引核对。
- 仅将 `SKILL.md` 的 description 翻译为中文；正文、作者、许可元数据和三个配套文件保持上游原样。
- `UPSTREAM.sha256.json` 记录原始四个文件的校验值；`EULA.txt` 来自同一安装包。

## 使用条件

本次仅安装 Skill，不安装 Nsight Systems 二进制或 Python 分析依赖。执行分析的机器需要安装匹配的 Nsight Systems（本版工作流使用 2026.6 新增的 `docs`、`survey` 和 `query` 命令）。

定位脚本依次检查 `NSYS_PATH`、`PATH` 和常用安装目录。`NSYS_PATH` 应指向包含 `nsys` 可执行文件的目录，而不是文件本身。原始脚本注释提到的 README.md 没有随上游 Skill 目录发布；本文件是本地说明。

远程任务遵循仓库 AGENTS.md：先通过 remote_fleet 确认目标机器，在实际执行机器上定位 nsys；报告和产物保存到 main-dev 的用户目录。用户和仓库的中文回复、进度汇报要求优先于上游 Response style。

## 安装位置

仓库内本目录为唯一事实来源，`/home/tiger/.codex/skills/nsight-systems` 通过绝对符号链接指向本目录。后续更新时应重新提取完整 Skill 目录，核对版本和文件校验值，并保留必要的本地说明。
