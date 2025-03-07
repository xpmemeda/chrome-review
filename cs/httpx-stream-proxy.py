import argparse
import multiprocessing
import subprocess
import json
import httpx
import asyncio
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def run_server(cmd_arguments):

    app = FastAPI()

    @app.post("/server")
    async def _(request: Request):
        body = await request.body()
        body = json.loads(body)

        async def generate_data():
            for i in range(3):
                await asyncio.sleep(1)
                yield json.dumps({i: body}) + "\n"

        return StreamingResponse(generate_data(), media_type="application/json")

    uvicorn.run(app, host="0.0.0.0", port=cmd_arguments.server_port)


def run_proxy(cmd_arguments):

    server_host = cmd_arguments.server_host
    server_port = cmd_arguments.server_port
    server_url = f"http://{server_host}:{server_port}"

    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def _(request: Request, path: str):
        body = await request.body()
        body = json.loads(body)
        body["proxy"] = True
        body = json.dumps(body).encode()
        headers = dict(request.headers)
        headers["content-length"] = str(len(body))

        async def stream_resp():
            async with httpx.AsyncClient() as client:
                url = f"{server_url}/{path}"
                async with client.stream(
                    method=request.method,
                    url=url,
                    content=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_resp(), media_type="application/json")

    uvicorn.run(app, host="0.0.0.0", port=cmd_arguments.proxy_port)


def run_client(cmd_arguments):
    cmd = (
        f" curl -N http://127.0.0.1:{cmd_arguments.proxy_port}/server "
        f' -H "Content-Type: application/json" '
        """ -d '{"name": "Book", "price": 1.6}' """
    )

    while subprocess.run(cmd, shell=True).returncode != 0:
        pass


def main(cmd_arguments):
    server = multiprocessing.Process(target=run_server, args=[cmd_arguments])
    server.start()

    proxy = multiprocessing.Process(target=run_proxy, args=[cmd_arguments])
    proxy.start()

    client = multiprocessing.Process(target=run_client, args=[cmd_arguments])
    client.start()
    client.join()

    server.kill()
    server.join()
    proxy.kill()
    proxy.join()


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--server-host", type=str, default="127.0.0.1")
    cmd_parser.add_argument("--server-port", type=int, default=8000)
    cmd_parser.add_argument("--proxy-host", type=str, default="127.0.0.1")
    cmd_parser.add_argument("--proxy-port", type=int, default=8001)
    cmd_arguments = cmd_parser.parse_args()
    main(cmd_arguments)
