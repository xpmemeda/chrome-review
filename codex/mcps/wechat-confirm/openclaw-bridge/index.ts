import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const DEFAULT_ENDPOINT = "http://127.0.0.1:18767/bridge/inbound";
const DEFAULT_SECRET_FILE = join(homedir(), ".local/state/wechat-confirm/bridge.secret");

export default definePluginEntry({
  id: "wechat-confirm-bridge",
  name: "WeChat Confirm Bridge",
  description: "Route WeChat replies to pending wechat_confirm MCP questions instead of the agent.",
  register(api) {
    const cfg = (api.pluginConfig ?? {}) as {
      endpoint?: string;
      secretFile?: string;
      timeoutMs?: number;
    };
    const endpoint = cfg.endpoint ?? DEFAULT_ENDPOINT;
    const secretFile = cfg.secretFile ?? DEFAULT_SECRET_FILE;
    const timeoutMs = cfg.timeoutMs ?? 3000;

    api.on("before_dispatch", async (event, ctx) => {
      const channel = event.channel ?? ctx.channelId;
      if (channel !== "openclaw-weixin" || event.isGroup) return;
      let secret: string;
      try {
        secret = readFileSync(secretFile, "utf-8").trim();
      } catch {
        return; // MCP service not set up: leave the message to the agent.
      }
      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "content-type": "application/json", "x-bridge-secret": secret },
          body: JSON.stringify({
            channel,
            accountId: ctx.accountId,
            messageId: event.messageId,
            senderId: event.senderId,
            content: event.content,
            body: event.body,
            isGroup: event.isGroup,
            replyToId: event.replyToId,
            replyToBody: event.replyToBody,
            timestamp: event.timestamp,
          }),
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!res.ok) return;
        const result = (await res.json()) as { handled?: boolean; text?: string };
        if (result.handled) return { handled: true, text: result.text };
      } catch (err) {
        api.logger?.warn?.(`wechat-confirm-bridge: ${String(err)}`);
      }
      return;
    });
  },
});
