#!/usr/bin/env python3
"""把逐 step 导出的 messages JSON 转换为内嵌图片的独立数据集。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from images import image_id, load_image, require_tools


def image_blocks(value):
    if isinstance(value, dict):
        if 'image_url' in value:
            image = value['image_url']
            if not isinstance(image, dict) or not isinstance(image.get('url'), str):
                raise ValueError('image_url 必须是带 url 字符串的对象')
            yield value
        for child in value.values():
            yield from image_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from image_blocks(child)


def identity(block):
    image = block['image_url']
    extra = image.get('extra') or {}
    if not isinstance(extra, dict):
        raise ValueError('image_url.extra 必须是对象')
    key = extra.get('imagex_encrypt_key')
    if key is not None and not isinstance(key, str):
        raise ValueError('图片密钥必须是字符串')
    return image['url'], key


def read_requests(directory, expected):
    requests = {}
    for path in directory.glob('step-*-request.json'):
        match = re.fullmatch(r'step-(\d+)-request\.json', path.name)
        if not match:
            raise ValueError(f'请求文件名不规范：{path.name}')
        step = int(match[1])
        if step in requests:
            raise ValueError(f'Step {step} 有多个输入文件')
        raw = path.read_bytes()
        try:
            doc = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ValueError(f'{path.name} 不是有效的 JSON') from None
        if not isinstance(doc, list) or not doc or any(
            not isinstance(message, dict) or not isinstance(message.get('role'), str)
            for message in doc
        ):
            raise ValueError(f'{path.name} 必须是非空 messages 数组')
        requests[step] = (raw, doc)
    if set(requests) != set(range(1, expected + 1)):
        raise ValueError(f'需要 Step 1–{expected}，实际编号为 {sorted(requests)}')
    return dict(sorted(requests.items()))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def same_tree(left, right):
    if left.is_symlink() or not left.is_dir():
        return False
    actual = {p.relative_to(left) for p in left.rglob('*') if p.is_file()}
    expected = {p.relative_to(right) for p in right.rglob('*') if p.is_file()}
    return actual == expected and all(
        not (left / p).is_symlink() and (left / p).read_bytes() == (right / p).read_bytes()
        for p in expected
    )


def prepare(args):
    source = args.input_dir.expanduser().resolve()
    output = (args.output_dir or Path.home() / 'workspace/dataset' / args.log_id).expanduser().absolute()
    if source == output.resolve() or source in output.resolve().parents or output.resolve() in source.parents:
        raise ValueError('输入和输出必须是互不包含的独立目录')
    requests = read_requests(source, args.expected_steps)
    references = {}
    for step, (_, doc) in requests.items():
        for block in image_blocks(doc):
            references.setdefault(identity(block), set()).add(step)
    cache = (args.cache_dir or source.parent / 'image-cache').expanduser().resolve()
    if cache == output.resolve() or output.resolve() in cache.parents:
        raise ValueError('图片缓存不能放在输出目录内')
    converted = {}
    if references:
        require_tools()
        cache.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(load_image, url, key, cache): (url, key)
                       for url, key in references}
            for future in as_completed(pending):
                ref = pending[future]
                try:
                    converted[ref] = future.result()
                except Exception as error:
                    for task in pending:
                        task.cancel()
                    raise ValueError(f'图片 {image_id(*ref)[:12]}，Step {sorted(references[ref])}：{error}') from None
                print(f'图片完成 {len(converted)}/{len(references)}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{args.log_id}-', dir=output.parent))
    try:
        (staging / 'original-urls').mkdir()
        stats = []
        for step, (raw, doc) in requests.items():
            filename = f'step-{step:02d}-request.json'
            count = 0
            for block in image_blocks(doc):
                block['image_url'] = {'url': converted[identity(block)]}
                count += 1
            path = staging / filename
            write_json(path, doc)
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved != doc or any(set(block['image_url']) != {'url'} for block in image_blocks(saved)):
                raise ValueError(f'Step {step} 写入校验失败')
            (staging / 'original-urls' / filename).write_bytes(raw)
            stats.append({'step': step, 'file': filename, 'messages': len(doc), 'images': count,
                          'source_sha256': sha256(raw), 'sha256': sha256(path.read_bytes())})
        manifest = {'log_id': args.log_id, 'step_count': len(stats),
                    'image_references': sum(row['images'] for row in stats),
                    'unique_images': len(references), 'steps': stats}
        write_json(staging / 'manifest.json', manifest)
        if output.exists() or output.is_symlink():
            if not same_tree(output, staging):
                raise ValueError('输出目录已存在且内容不同；请指定新的 --output-dir')
            print(f'输出已存在且内容一致：{output}')
        else:
            staging.rename(output)
            print(f'已保存：{output}')
        print(f'{len(stats)} 步，{manifest["image_references"]} 处图片引用，{len(references)} 张独立图片')
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('必须为正整数')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-id', required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--expected-steps', type=positive_int, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--workers', type=positive_int, default=6)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.log_id):
        parser.error('log_id 仅允许字母、数字、下划线和连字符')
    try:
        prepare(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f'失败：{error}\n')


if __name__ == '__main__':
    main()
