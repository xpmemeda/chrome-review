import argparse
import json
import random
import tqdm
import wonderwords


def main(cmd_arguments):
    num_tokens_list = []
    num_prefix_list = []
    random_words = wonderwords.RandomWord().random_words(cmd_arguments.num_requests)

    random.seed(0)
    """
    1. 生成一个随机长度。
    2. 根据长度生成命中前缀的长度。

    共同前缀用 hi ，后面跟一个 random word 分割避免命中，最后用随机的 hi / hello 补齐到前面生成的长度。
    """
    for _ in range(cmd_arguments.num_requests):
        num_tokens = random.randint(
            cmd_arguments.num_prompt_tokens - cmd_arguments.num_prompt_tokens_rand_size,
            cmd_arguments.num_prompt_tokens + cmd_arguments.num_prompt_tokens_rand_size,
        )
        num_tokens_list.append(num_tokens)
        num_prefix_list.append(int(num_tokens * cmd_arguments.request_prefix_hit_ratio))

    json_line = []

    iters = tqdm.tqdm(range(cmd_arguments.num_requests))
    for i in iters:
        num_tokens = num_tokens_list[i]
        prefix_hit_size = num_prefix_list[i]

        prefix = "hi " * prefix_hit_size
        random_word = random_words[i]
        suffix = "".join(
            random.choice(["hi ", "hello "])
            for _ in range(num_tokens - prefix_hit_size - 1)
        )

        messages = [{"role": "user", "content": prefix + random_word + suffix}]

        json_line.append(json.dumps(messages))

    with open(cmd_arguments.output_path, "w") as f:
        for line in json_line:
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    cmd_arguments = argparse.ArgumentParser()
    cmd_arguments.add_argument("--output-path", type=str, required=True)
    cmd_arguments.add_argument("--num-requests", type=int, required=True)
    cmd_arguments.add_argument("--num-prompt-tokens", type=int, required=True)
    cmd_arguments.add_argument("--num-prompt-tokens-rand-size", type=int, required=True)
    cmd_arguments.add_argument("--request-prefix-hit-ratio", type=float, required=True)
    cmd_arguments = cmd_arguments.parse_args()
    main(cmd_arguments)
