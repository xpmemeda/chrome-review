import json
import argparse


class WechatShop:
    def __init__(self, dataset_path):
        self.dataset = []

        with open(dataset_path, "r") as f:
            while system_prompt := f.readline():
                if system_prompt.startswith(
                    r"##角色\n你是专业的微信小店的智能客服"
                ) or system_prompt.startswith(r"你是一个微信小店的智能客服，回答用户问题。"):
                    user_prompt = f.readline()
                    self.dataset.append(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )

    def format_save(self, dataset_path):
        with open(dataset_path, "w") as f:
            for messages in self.dataset:
                f.write(json.dumps(messages))
                f.write("\n")


if __name__ == "__main__":
    cmd_arguments = argparse.ArgumentParser()
    cmd_arguments.add_argument("--src-path", type=str, required=True)
    cmd_arguments.add_argument("--dst-path", type=str, required=True)
    cmd_arguments = cmd_arguments.parse_args()

    WechatShop(cmd_arguments.src_path).format_save(cmd_arguments.dst_path)
