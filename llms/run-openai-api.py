import argparse
import openai
import asyncio


def sync_nonstream(cmd_arguments):
    client = openai.OpenAI(
        base_url=cmd_arguments.base_url, api_key=cmd_arguments.api_key
    )

    prompt = "你好！"
    completion = client.chat.completions.create(
        model=cmd_arguments.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
    )
    print(completion)


def sync_stream(cmd_arguments):
    client = openai.OpenAI(
        base_url=cmd_arguments.base_url, api_key=cmd_arguments.api_key
    )

    prompt = "你好！"
    completion = client.chat.completions.create(
        model=cmd_arguments.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        stream=True,
    )
    for chunk in completion:
        print(chunk)
        print(chunk.choices[0].delta.content)


async def async_nonstream(cmd_arguments):
    client = openai.AsyncClient(
        base_url=cmd_arguments.base_url, api_key=cmd_arguments.api_key
    )
    prompt = "你好！"
    completion = await client.chat.completions.create(
        model=cmd_arguments.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
    )
    print(completion)


async def async_stream(cmd_arguments):
    client = openai.AsyncOpenAI(
        base_url=cmd_arguments.base_url, api_key=cmd_arguments.api_key
    )

    prompt = "你好！"
    completion = await client.chat.completions.create(
        model=cmd_arguments.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        stream=True,
    )
    async for chunk in completion:
        print(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat-mode",
        choices=["sync-nonstream", "sync-stream", "async-nonstream", "async-stream"],
        required=True,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="example: http://29.39.224.137:8080/v1",
    )
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    cmd_arguments = parser.parse_args()

    if cmd_arguments.chat_mode == "sync-nonstream":
        sync_nonstream(cmd_arguments)
    if cmd_arguments.chat_mode == "sync-stream":
        sync_stream(cmd_arguments)
    if cmd_arguments.chat_mode == "async-nonstream":
        asyncio.run(async_nonstream(cmd_arguments))
    if cmd_arguments.chat_mode == "async-stream":
        asyncio.run(async_stream(cmd_arguments))
