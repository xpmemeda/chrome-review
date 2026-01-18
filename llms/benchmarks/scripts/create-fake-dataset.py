import argparse
import json
import random
import tqdm


def main(cmd_arguments):
    num_tokens_list = []
    num_prefix_list = []
    random.seed(0)
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
        suffix = "".join(
            random.choice(["hi ", "hello "])
            for _ in range(num_tokens - prefix_hit_size)
        )

        messages = [{"role": "user", "content": prefix + suffix}]

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
