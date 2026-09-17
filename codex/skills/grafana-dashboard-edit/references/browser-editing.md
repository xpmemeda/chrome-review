# 浏览器编辑经验

以下来自 grafana.byted.org 的 Grafana v10 实际操作，UI 更新后以现场为准。先读取当前 Browser skill 和对应运行时文档；不要用独立网络请求、隐藏应用状态或另一套浏览器框架替代受支持的操作。

## JSON Model

通过 Dashboard settings → JSON Model 读取看板。在界面或已观察 URL 支持时，可使用 `editview=dashboard_json`。常见 Monaco 输入框的 accessible name 为 `Editor content;Press Alt+F1 for Accessibility Options.`，应先从 DOM 核对。

读取完整 JSON：聚焦编辑器，全选、复制，然后在下一次调用中读取标签页剪贴板并 JSON.parse。可见 DOM 常只有虚拟化片段，不适合当完整 JSON。

修改时全选后用 Browser 支持的输入操作粘贴完整 JSON。本环境曾观察到 Monaco `.fill()` 插入而非替换，不能假设 fill 已清空全文。写入后确保是单个有效 JSON，再保存。

此版本的 **Save changes** 已直接持久化 JSON；不要紧接着用旧页面缓存再保存一次。其他 Grafana 版本可能还有保存对话框，应根据 UI 完成。

## 验证

- 新开同 UID 页面确认，避免旧标签页缓存或本地草稿误导。
- 可用已观察到的 `viewPanel=<ID>` 定位；显式携带用户 PSM、dc 和时间范围，避免新标签页退回默认服务。
- 等待真实数据返回后检查图例行数和数值。必要时用 Explore / Query inspector 查看查询与返回数据；Explore 载入后可能还需点 Run query。
- 空序列排查顺序：时间范围和 PSM → tenant → 指标后缀 → 过滤标签 → 是否确实有埋点上报。不要在没有证据时重写服务代码。
- 浏览器关闭或清理旧标签页后，从现有浏览器 binding 获取新标签页；不要把已失效的 tab ID 写进流程。交付标签页按当前 Browser 文档支持的方式保留。

## 备份与恢复

将 UI 读取的 JSON 作为本地备份，不通过额外网络通道取得数据。恢复时优先把这次修改的字段恢复到最新配置，而非直接覆盖整份旧 JSON，以免抹掉用户期间的其他修改。未经用户要求，不回滚其他字段。
