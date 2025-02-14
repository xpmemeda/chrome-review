import ray
import torch
import safetensors

import torch
import safetensors.torch as sft


torch.manual_seed(0)


class MoE(torch.nn.Module):
    def __init__(self, num_experts, k, n):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(k, n) for _ in range(num_experts)]
        )

    def forward(self, x):
        return torch.stack([expert(x) for expert in self.experts]).sum(0)


x = torch.rand(1024, 1024, device="cuda:0", dtype=torch.float32)
net = MoE(8, 1024, 1024).to(device="cuda:0")
r = net(x)
print(r)

sft.save_file(net.state_dict(), "0.safetensors")
sft.save_file({"x": x, "r": r}, "1.safetensors")


ray.init()


@ray.remote(num_gpus=1)
class TPMoE(torch.nn.Module):
    def __init__(self, num_experts, k, n, rank, ws, weight_path):
        super().__init__()
        assert n % ws == 0, "n cannot split by world size."

        split_n = n // ws
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(k, split_n, bias=False) for _ in range(num_experts)]
        )
        self.eval()

        self.load_weights(rank, ws, weight_path)
        self.to("cuda")

    def forward(self, x):
        return torch.stack([expert(x) for expert in self.experts]).sum(0)

    def load_weights(self, rank, ws, weight_path):
        with safetensors.safe_open(weight_path, framework="pt") as f:
            print(f.keys())
            for name, param in self.named_parameters():
                print(name)
                full_weight = f.get_tensor(name)
                split_size = full_weight.size(0) // ws
                start_col = split_size * rank
                end_col = split_size * (rank + 1)
                with torch.no_grad():
                    print(param.size())
                    print(full_weight.size())
                    param.copy_(full_weight[start_col:end_col, :])


def tp(x, ws, weight_path):
    num_experts = 8
    k = 1024
    n = 1024
    workers = [TPMoE.remote(8, k, n, rank, ws, weight_path) for rank in range(ws)]
    futures = [worker.forward.remote(x) for worker in workers]
    results = ray.get(futures)
    return torch.cat(results, dim=1)


if __name__ == "__main__":
    a = torch.randn(1024, 1024).cuda()  # 输入 A
    world_size = 2  # 假设使用 4 个 Workers
    result = tp(a, world_size, "/0.safetensors")
    print("Result shape:", result.shape)  # 应输出 torch.Size([1024, 1024])

# @ray.remote(num_gpus=1)
# def block_matmul(a: Tensor, b: Tensor) -> Tensor:
#     print(f"计算分块在节点 {ray.get_runtime_context().node_id}")
#     print(type(a))
#     print(type(b))
#     return torch.matmul(a, b)


# def tp_moe(a: Tensor, b: Tensor, split_dim: int = 0):
#     a_ref = ray.put(a)
#     b_blocks = torch.chunk(b, 2, dim=1)
#     b_block_refs = [ray.put(block) for block in b_blocks]
#     result_refs = [block_matmul.remote(a_ref, b_ref) for b_ref in b_block_refs]
#     partial_results = ray.get(result_refs)
#     return torch.cat(partial_results, dim=1)


# if __name__ == "__main__":
#     A = torch.randn(1024, 10240, device="cuda:0")
#     B = torch.randn(10240, 20480, device="cuda:0")
#     result_split0 = tp_moe(A, B, split_dim=1)
#     print(result_split0)
