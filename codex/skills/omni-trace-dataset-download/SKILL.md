---
name: omni-trace-dataset-download
description: 从 Omni Trace 的 omni_viewer 页面按 step 导出模型请求，下载并解密图片，将图片内嵌为 Base64，生成本地请求数据集。用于用户提供 Omni Trace 链接或 log_id 并要求下载数据集，也可处理已导出的逐步请求 JSON；不用于服务日志下载或性能压测。
---

# Omni Trace 请求数据集

输入为 Omni Trace 链接、log_id，或已导出的逐步请求目录。默认输出到 `~/workspace/dataset/<log_id>/`，每步一个 `step-01-request.json`。文件保持页面导出的 messages 数组结构；图片块保持 `type: image_url`，其中 `image_url` 对象仅保留 Base64 Data URL 字段 `url`。若用户指定不同路径或格式，以用户要求为准。

## 获取每步请求

1. 用户要求打开页面时使用当前可用的 Browser 技能，按其文档操作。只有 log_id 时，可使用 `https://oassistant-zeus-api.bytedance.net/omni_viewer?log_id=<log_id>`。若用户只要求处理本地导出文件，直接进入本地转换。
2. 等待 Trace 加载完成，确认页面 Log ID 和实际 step 列表。记录 step 数；不要把加载中的“0 步”当成结果，也不要把“Step 0 · 前置消息”当成一次模型请求。
3. 在每个 `Step N` 对应卡片内点击“下载请求”。页面顶部“导出”得到的是整个 Trace，不能直接充当每步模型请求；“模型原始输出”也不是请求。
4. 下载前记录本次开始时间或已有文件清单，确认每次生成的下载文件属于本次操作。系统可能添加 ` (1)` 后缀，不要仅取 Downloads 中同名文件或假定最大修改时间的文件必然正确。用浏览器实际下载结果和文件完成状态定位文件。
5. 将本次各 step 的文件复制到独立的 `work/omni-<log_id>/requests/`，统一命名为 `step-01-request.json` 等。只放每步请求，保留原始图片 `uri` 和 `extra`，完成解密后脚本才会删除它们。

页面曾支持以 `article` 内精确的 `Step N` 按钮定位卡片，再定位其“下载请求”按钮；每次应以当前 DOM 为依据。复制请求曾返回空剪贴板，遇到这种情况可使用下载按钮，不要写出空请求。不要硬编码 14 步或按固定消息数量推算完整性：工具、额外上下文会改变每步消息数。

## 转换本地请求

使用 Skill 下的 `scripts/prepare_dataset.py`，通过交互式 zsh 加载本地环境。将命令中的路径、log_id、步数替换为本次实际值：

```bash
zsh -ic 'python3 /Users/bytedance/workspace/github/chrome-review/codex/skills/omni-trace-dataset-download/scripts/prepare_dataset.py --log-id LOG_ID --input-dir /absolute/work/omni-LOG_ID/requests --expected-steps N'
```

- `--output-dir`：覆盖默认输出路径。
- `--workers`：图片并发数，默认 6。
- `--cache-dir`：默认在输入目录旁的 `image-cache/`，用于失败后重试复用已下载、解密的图片。
- 输入文件必须是连续的 `step-<数字>-request.json`，不接收未经拆分的整份 Trace。`--expected-steps` 必须来自页面或用户确认的本地导出数量。
- 原始请求保存在输出目录 `original-urls/`，转换统计保存在 `manifest.json`。原始文件、缓存、图片密钥和实际请求内容都不能加入 Skill 仓库。
- 已有输出完全相同时可重复执行；若输出不同，脚本报冲突并保留现有数据。需要生成修订版时，使用新的输出目录，或在用户已授权替换的情况下由调用方妥善处理旧目录。

脚本先转换并校验所有请求，再发布整个输出目录，避免下载失败留下半套成品。图片缓存会保留以便重试。失败时报告对应 step 或图片短标识，不输出完整签名 URL、解密密钥或 Base64 内容。

## 图片处理约定

- 以 URL 和密钥共同去重，避免同一 URL 的不同密钥被错误合并。
- 加密内容以 **16 字节** `aes256cfb-direct` 开头；随后 16 字节为 IV，其余为密文。使用 `extra.imagex_encrypt_key` 的 UTF-8 字节作为 32 字节 AES-256-CFB 密钥。头部不含 `!`，不要把 IV 的首字节误认为固定头部。
- 先解密，再识别文件格式。HTTP 200 或 `application/octet-stream` 不代表已经得到普通图片。
- JPEG、PNG、GIF、WebP 保留原始编码；HEIC/HEIF/AVIF 使用 macOS `sips` 转成 JPEG（质量 95）。需要 `openssl`；其他平台或缺少转换能力时明确报错，不把密文、HTML 或未知字节标成 JPEG。
- 已内嵌的 Data URL 也检查 Base64 和实际图片解码。转换成功后才将 `image_url` 替换为 `{"url": "data:image/...;base64,..."}`，删除其中 `uri`、`extra` 和其他字段，保留图片块外的字段。
- 403 等签名失效错误应重新从页面导出请求获取新 URL；不要无限重试旧签名。

## 完成判定

确认 step 编号连续且与页面一致、所有 JSON 可解析、图片都为可解码的 Data URL、`image_url` 仅含 `url`。保留每步消息顺序、文本和工具调用。最终告知目录、step 数、图片引用数，以及存在的失败；未全部成功时不能声称数据集完整。
