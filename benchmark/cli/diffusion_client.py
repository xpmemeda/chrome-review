import asyncio
import mimetypes
import time
import typing as ty
import urllib.parse
import uuid

import dataset as dataset_lib
from metrics import RequestMetrics

from .messages import decode_data_url, extract_first_image_url, iter_message_text

JsonDict = ty.Dict[str, ty.Any]


def split_http_url(url: str) -> ty.Tuple[str, int, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http":
        raise ValueError("Only http:// URLs are supported by the stdlib async client.")
    if not parsed.hostname:
        raise ValueError(f"Invalid URL: {url}")
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return parsed.hostname, port, path


class AsyncMultipartHttpClient:
    def __init__(self, url: str, timeout: float) -> None:
        self.host, self.port, self.path = split_http_url(url)
        self.timeout = timeout

    async def post_multipart(
        self,
        data: ty.Dict[str, str],
        files: ty.Dict[str, ty.Tuple[str, bytes, str]],
    ) -> ty.Tuple[float, bytes]:
        body, content_type = encode_multipart_form_data(data, files)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=self.timeout
        )
        try:
            headers = [
                f"POST {self.path} HTTP/1.1",
                f"Host: {self.host}:{self.port}",
                f"Content-Type: {content_type}",
                f"Content-Length: {len(body)}",
                "Connection: close",
                "",
                "",
            ]
            writer.write("\r\n".join(headers).encode() + body)
            await asyncio.wait_for(writer.drain(), timeout=self.timeout)

            status_line = await asyncio.wait_for(
                reader.readline(), timeout=self.timeout
            )
            if not status_line:
                raise RuntimeError("empty HTTP response")
            parts = status_line.decode("latin1").strip().split(" ", 2)
            status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0

            resp_headers: JsonDict = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin1").partition(":")
                resp_headers[name.lower()] = value.strip().lower()

            chunks: ty.List[bytes] = []
            if resp_headers.get("transfer-encoding") == "chunked":
                async for _, chunk in self._read_chunked(reader):
                    chunks.append(chunk)
            else:
                while True:
                    chunk = await asyncio.wait_for(
                        reader.read(65536), timeout=self.timeout
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
            data_bytes = b"".join(chunks)

            if status < 200 or status >= 300:
                err = data_bytes[:4096].decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {status}: {err}")
            return time.perf_counter(), data_bytes
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_chunked(
        self, reader: asyncio.StreamReader
    ) -> ty.AsyncIterator[ty.Tuple[float, bytes]]:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
            if not line:
                break
            size_text = line.split(b";", 1)[0].strip()
            if not size_text:
                continue
            size = int(size_text, 16)
            if size == 0:
                await asyncio.wait_for(reader.readline(), timeout=self.timeout)
                break
            data = await asyncio.wait_for(
                reader.readexactly(size), timeout=self.timeout
            )
            await asyncio.wait_for(reader.readexactly(2), timeout=self.timeout)
            yield time.perf_counter(), data


def encode_multipart_form_data(
    data: ty.Dict[str, str],
    files: ty.Dict[str, ty.Tuple[str, bytes, str]],
) -> ty.Tuple[bytes, str]:
    boundary = f"----benchmark-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in data.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode())
        body.extend(b"\r\n")
    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


class DiffusionClient:
    def __init__(
        self,
        url: str,
        timeout: float,
        style: ty.Optional[str],
        seed: ty.Optional[int],
        steps: ty.Optional[int],
        extra_fields: ty.Optional[JsonDict],
    ) -> None:
        self.http_client = AsyncMultipartHttpClient(url, timeout)
        self.style = style
        self.seed = seed
        self.steps = steps
        self.extra_fields = extra_fields or {}

    async def send_request(
        self,
        req_idx: int,
        request: dataset_lib.StdChatApiRequest,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        del sampling_params
        stime = time.perf_counter()
        try:
            messages = request["messages"]
            image_url = extract_first_image_url(messages)
            if image_url is None:
                raise RuntimeError("DiffusionClient requires an image request.")
            image_mime, image_bytes = decode_data_url(image_url)
            ext = mimetypes.guess_extension(image_mime) or ".png"
            data = {str(k): str(v) for k, v in self.extra_fields.items()}
            prompt = "\n".join(iter_message_text(messages))
            data["style"] = self.style or prompt or "摄影后期"
            if self.seed is not None:
                data["seed"] = str(self.seed + req_idx)
            if self.steps is not None:
                data["steps"] = str(self.steps)
            files = {
                "image": (
                    f"request-{req_idx}{ext}",
                    image_bytes,
                    image_mime,
                )
            }
            first_byte_ts, output = await self.http_client.post_multipart(data, files)
            etime = time.perf_counter()
            return RequestMetrics(
                req_idx,
                True,
                first_byte_ts - stime,
                etime - stime,
                0,
                len(output),
                1,
            )
        except Exception as e:
            etime = time.perf_counter()
            return RequestMetrics(
                req_idx,
                False,
                0.0,
                etime - stime,
                0,
                0,
                0,
                (),
                repr(e),
            )
