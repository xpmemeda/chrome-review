import re
import os
import sys

log_file = sys.argv[1]  # 从命令行参数获取日志文件路径
out_dir = sys.argv[2]  # 输出目录

os.makedirs(out_dir, exist_ok=True)

# 匹配：2026-01-26 13:05:12 或 2026-01-26 13:05:12.345
ts_pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s+(\d{2}):\d{2}:\d{2}"
)

files = {}  # hour_str -> file handle

def get_outfile(date, hour):
    key = f"{date}_{hour}"
    if key not in files:
        path = os.path.join(out_dir, f"log_{date}_{hour}.log")
        files[key] = open(path, "a", encoding="utf-8")
    return files[key]

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        m = ts_pattern.search(line)
        if not m:
            continue  # 没时间戳的行直接丢，或你也可以写到 unknown.log

        date, hour = m.group(1), m.group(2)
        out = get_outfile(date, hour)
        out.write(line)

# 关闭所有文件
for fp in files.values():
    fp.close()

print(f"Done. Split logs written to: {out_dir}")
