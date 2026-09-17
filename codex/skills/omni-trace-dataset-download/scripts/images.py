"""Omni 图片下载、解密、格式转换和 Data URL 校验。"""

import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def image_id(url, key):
    return hashlib.sha256((url + '\0' + (key or '')).encode()).hexdigest()


def run_command(command, data=None):
    # 命令可能包含密钥，不让 CalledProcessError 把完整参数带到日志。
    try:
        result = subprocess.run(command, input=data, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError(f'{command[0]} 无法执行或超时') from None
    if result.returncode:
        raise ValueError(f'{command[0]} 执行失败')
    return result.stdout


def mime_type(data):
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    raise ValueError('内容不是支持的标准图片')


def verify_image(data):
    """通过系统解码器检查图片尺寸，避免只校验文件头。"""
    mime = mime_type(data)
    with tempfile.TemporaryDirectory(prefix='omni-verify-') as directory:
        source = Path(directory) / ('image.' + mime.split('/')[1])
        source.write_bytes(data)
        output = run_command(['sips', '-g', 'pixelWidth', '-g', 'pixelHeight', str(source)]).decode()
        dimensions = {}
        for line in output.splitlines():
            key, _, value = line.strip().partition(': ')
            if key in ('pixelWidth', 'pixelHeight') and value.isdigit():
                dimensions[key] = int(value)
        if len(dimensions) != 2 or min(dimensions.values()) <= 0:
            raise ValueError('图片解码失败或尺寸无效')
    return mime


def normalize(data, key):
    if data.startswith(b'aes256cfb-direct'):
        if len(data) <= 32 or not isinstance(key, str) or len(key.encode()) != 32:
            raise ValueError('加密图片缺少有效的 32 字节密钥或数据不完整')
        data = run_command(['openssl', 'enc', '-d', '-aes-256-cfb',
                            '-K', key.encode().hex(), '-iv', data[16:32].hex()], data[32:])
    if data[4:8] == b'ftyp':
        with tempfile.TemporaryDirectory(prefix='omni-convert-') as directory:
            source = Path(directory) / 'source.heic'
            target = Path(directory) / 'image.jpg'
            source.write_bytes(data)
            run_command(['sips', '-s', 'format', 'jpeg', '-s', 'formatOptions', '95',
                         str(source), '--out', str(target)])
            data = target.read_bytes()
    verify_image(data)
    return data


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise ValueError(f'HTTP {error.code}，请重新导出请求刷新图片 URL') from None
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise ValueError(f'图片下载失败：HTTP {error.code}') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise ValueError('图片下载失败：网络连接异常或超时') from None
        time.sleep(attempt + 1)


def to_data_url(data):
    return 'data:' + mime_type(data) + ';base64,' + base64.b64encode(data).decode('ascii')


def decode_data_url(value):
    header, separator, payload = value.partition(',')
    if not separator or not header.startswith('data:image/') or not header.endswith(';base64'):
        raise ValueError('无效的图片 Data URL')
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError:
        raise ValueError('无效的图片 Base64') from None
    if header != 'data:' + verify_image(data) + ';base64':
        raise ValueError('Data URL MIME 与图片格式不符')
    return data


def load_image(url, key, cache):
    if url.startswith('data:'):
        return to_data_url(decode_data_url(url))
    if not url.startswith(('https://', 'http://')):
        raise ValueError('图片 URL 必须是 HTTP(S) 或 Data URL')
    target = cache / (image_id(url, key) + '.image')
    if target.exists():
        data = target.read_bytes()
        try:
            verify_image(data)
            return to_data_url(data)
        except ValueError:
            pass  # 损坏缓存从源重新下载，成功前不覆盖。
    data = normalize(fetch(url), key)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return to_data_url(data)


def require_tools():
    for tool in ('openssl', 'sips'):
        if not shutil.which(tool):
            raise ValueError(f'缺少 {tool}；此脚本使用 macOS 的 sips 进行图片转换和解码校验')
