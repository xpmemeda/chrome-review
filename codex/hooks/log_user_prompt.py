#!/usr/bin/env python3
"""Log Codex UserPromptSubmit hook payloads."""

import datetime
import json
import os
import sys
import zoneinfo


def Main() -> int:
  try:
    data = json.load(sys.stdin)
  except json.JSONDecodeError as error:
    prompt = f"<invalid hook json: {error}>"
  else:
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str):
      prompt = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))

  now = datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Shanghai"))
  timestamp = now.strftime("%Y-%m-%d %H:%M:%S CST")
  encoded_prompt = json.dumps(prompt, ensure_ascii=False)
  log_path = os.path.expanduser("~/.codex.log")

  with open(log_path, "a", encoding="utf-8") as log_file:
    log_file.write(f"[{timestamp}] UserPromptSubmit prompt={encoded_prompt}\n")

  return 0


if __name__ == "__main__":
  sys.exit(Main())
