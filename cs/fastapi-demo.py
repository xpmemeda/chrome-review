"""

特性	GET	POST
用途	获取资源	提交数据
数据传递	URL 参数	请求体
安全性	较低（数据在 URL 中）	较高（数据在请求体中）
缓存	可以被缓存	默认不会被缓存
幂等性	幂等	非幂等
使用场景	读取操作（如查询数据）	写入操作（如提交表单）
"""

import json
import time
import asyncio
import uvicorn
import argparse
import subprocess

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse

app = FastAPI()


class Request(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


@app.get("/help")
def help():
    return "fastaip demo."


@app.get("/sleep")
def sleep(seconds: float):
    time.sleep(seconds)
    return f"sleep {seconds=} done."


@app.get("/async-sleep")
async def async_sleep(seconds: float):
    await asyncio.sleep(seconds)
    return f"async sleep {seconds=} done."


@app.post("/request")
def api_request(request: Request):
    return {"request": request}


@app.post("/request-stream")
def api_request_stream(request: Request):
    def generate_data():
        for i in range(3):
            time.sleep(1)
            yield json.dumps({i: request.model_dump()}) + "\n"

    return StreamingResponse(generate_data(), media_type="application/json")


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        "--role", type=str, choices=["server", "client"], required=True
    )
    cmd_arguments = cmd_parser.parse_args()
    if cmd_arguments.role == "server":
        uvicorn.run(app=app, host="0.0.0.0", port=8000)
    else:
        subprocess.run("curl http://127.0.0.1:8000/help", shell=True)
        subprocess.run("curl http://127.0.0.1:8000/sleep?seconds=1", shell=True)
        subprocess.run("curl http://127.0.0.1:8000/async-sleep?seconds=1", shell=True)
        # -H: head.
        # -d: data.
        cmd = """curl http://127.0.0.1:8000/request -H "Content-Type: application/json" -d '{"name": "Book", "price": 1.6}'"""
        subprocess.run(cmd, shell=True)
        # -N: no buffer.
        cmd = """curl http://127.0.0.1:8000/request-stream -N -H "Content-Type: application/json" -d '{"name": "Book", "price": 1.6}'"""
        subprocess.run(cmd, shell=True)
