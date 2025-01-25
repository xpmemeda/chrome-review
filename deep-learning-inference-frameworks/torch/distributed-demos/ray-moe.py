import re
import ray
import torch
import argparse
import safetensors
import safetensors.torch


def save_net(num_experts, m, n, k, safetensors_path):
    class MoE(torch.nn.Module):
        def __init__(self, num_experts, k, n):
            super().__init__()
            self.experts = torch.nn.ModuleList(
                [torch.nn.Linear(k, n, bias=False) for _ in range(num_experts)]
            )

        def forward(self, x):
            return torch.stack([expert(x) for expert in self.experts]).sum(0)

    torch.manual_seed(0)
    x = torch.rand(m, k, device="cuda:0", dtype=torch.float32)
    net = MoE(num_experts, n, k).to(device="cuda:0")
    r = net(x)

    safetensors_dict = net.state_dict()
    safetensors_dict.update({"x": x, "r": r})
    safetensors.torch.save_file(safetensors_dict, safetensors_path)


def load_net(num_experts, m, n, k, safetensors_path, world_size):
    ray.init()

    @ray.remote(num_gpus=1)
    class TPMoE(torch.nn.Module):
        def __init__(self, num_experts, k, n, rank, ws, safetensors_path):
            super().__init__()
            assert n % ws == 0, "n cannot split by world size."

            split_n = n // ws
            self.experts = torch.nn.ModuleList(
                [torch.nn.Linear(k, split_n, bias=False) for _ in range(num_experts)]
            )
            self.eval()

            self.load_weights(rank, ws, safetensors_path)
            self.to("cuda")

        def load_weights(self, rank, ws, safetensors_path):
            with torch.no_grad():
                with safetensors.safe_open(safetensors_path, framework="pt") as f:
                    for name, param in self.named_parameters():
                        full_weight = f.get_tensor(name)
                        split_size = full_weight.size(0) // ws
                        start_col = split_size * rank
                        end_col = split_size * (rank + 1)
                        param.copy_(full_weight[start_col:end_col, :])

        def forward(self, x):
            return torch.stack([expert(x) for expert in self.experts]).sum(0)

        @classmethod
        def merge(cls, results):
            return torch.cat(results, dim=1)

    @ray.remote(num_gpus=1)
    class EPMoE(torch.nn.Module):
        def __init__(self, num_experts, k, n, rank, ws, safetensors_path):
            super().__init__()
            assert num_experts % ws == 0, "experts cannot split by world size."

            self.experts = torch.nn.ModuleList(
                [torch.nn.Linear(k, n, bias=False) for _ in range(num_experts // ws)]
            )
            self.eval()

            self.load_weights(rank, ws, safetensors_path)
            self.to("cuda")

        def load_weights(self, rank, ws, safetensors_path):
            with torch.no_grad():
                with safetensors.safe_open(safetensors_path, framework="pt") as f:
                    for name, param in self.named_parameters():
                        match = re.match("experts\.(?P<expert_id>\d+)", name)
                        if match is None:
                            pass
                        else:
                            local_expert_id = int(match.group("expert_id"))
                            num_experts_per_card = len(self.experts)
                            global_expert_id = (
                                local_expert_id + rank * num_experts_per_card
                            )
                            name = name.replace(
                                str(local_expert_id), str(global_expert_id)
                            )
                        full_weight = f.get_tensor(name)
                        param.copy_(full_weight)

        def forward(self, x):
            return torch.stack([expert(x) for expert in self.experts]).sum(0)

        @classmethod
        def merge(cls, results):
            return torch.stack(results).sum(0)

    for moe_cls in [TPMoE, EPMoE]:
        print("moe cls: %s" % ("TPMoE" if moe_cls == TPMoE else "EPMoE"))
        workers = [
            moe_cls.remote(num_experts, k, n, rank, world_size, safetensors_path)
            for rank in range(world_size)
        ]

        with safetensors.safe_open(safetensors_path, framework="pt") as f:
            x = f.get_tensor("x").to("cuda")
            r = f.get_tensor("r").to("cuda")
            assert x.size(0) == m and x.size(1) == k
        futures = [worker.forward.remote(x) for worker in workers]
        results = ray.get(futures)
        actual = moe_cls.merge(results)
        torch.testing.assert_close(actual, r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument(
        "--f", type=str, default="/moe.safetensors", help="the file to save tensors."
    )
    cmd_arguments = parser.parse_args()
    if cmd_arguments.prepare:
        save_net(
            cmd_arguments.num_experts,
            cmd_arguments.m,
            cmd_arguments.n,
            cmd_arguments.k,
            cmd_arguments.f,
        )
    else:
        world_size = 2
        load_net(
            cmd_arguments.num_experts,
            cmd_arguments.m,
            cmd_arguments.n,
            cmd_arguments.k,
            cmd_arguments.f,
            world_size,
        )
