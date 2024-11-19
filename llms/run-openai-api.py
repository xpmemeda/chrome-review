import argparse
from openai import OpenAI


def main(ip, port, api_key, model):
    prompts = ["你好！", "螃蟹是什么颜色？", "早餐吃什么比较健康？"]

    for prompt in prompts:
        client = OpenAI(
            base_url=f"http://{ip}:{port}/v1",
            api_key=api_key,
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        print(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    arguments = parser.parse_args()
    main(arguments.ip, arguments.port, arguments.api_key, arguments.model)
