import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, safe_open, save_file


class Net(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self._fc = torch.nn.Linear(feature_size, feature_size, True)

    def forward(self, x):
        return self._fc(x)


class ShardNet(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()

        assert feature_size % dist.get_world_size() == 0, "cannot split"
        self._fc = torch.nn.Linear(
            feature_size, feature_size // dist.get_world_size(), True
        )

    def forward(self, x):
        return self._fc(x)


feature_size = 1024
safetensors_path = "net.safetensors"


def run(rank, size):
    with torch.no_grad():
        net = ShardNet(feature_size).to("cuda:%d" % rank)
        params_dict = dict(net.named_parameters(remove_duplicate=False))
        with safe_open(safetensors_path, framework="pt") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                size_per_partition = param.size(0) // size
                src_param = param.narrow(
                    0, size_per_partition * rank, size_per_partition
                )
                dst_param = params_dict[name]
                dst_param.copy_(src_param)
        desired = load_file("desired.safetensors", device="cuda:%d" % rank)
        x = desired["x"]
        rs = net(x)
        r1 = torch.empty_like(rs)
        r2 = torch.empty_like(rs)
        r12 = [r1, r2]
        dist.all_gather(r12, rs, group=dist.new_group([0, 1]))
        r = torch.cat(r12)
        assert torch.allclose(r, desired["r"], atol=1e-6, rtol=1e-6)


def init_process(rank, size, fn, backend="nccl"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    with torch.no_grad():
        feature_size = 1024
        safetensors_path = "net.safetensors"
        net = Net(feature_size)
        save_file(net.state_dict(), safetensors_path)
        x = torch.rand(feature_size)
        save_file({"x": x}, "x.safetensors")
        r = net(x)
        save_file({"x": x, "r": r}, "desired.safetensors")

    world_size = 2
    processes = []

    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
