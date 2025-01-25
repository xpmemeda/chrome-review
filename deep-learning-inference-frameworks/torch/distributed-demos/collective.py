import os
import torch
import argparse
import torch.distributed as dist


def run(cmd_arguments):
    device = torch.device("cuda", cmd_arguments.device)
    tensor = torch.ones(1, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print("Rank ", cmd_arguments.rank, " has data ", tensor[0])


def init_process_group(cmd_arguments):
    os.environ["MASTER_ADDR"] = cmd_arguments.master
    os.environ["MASTER_PORT"] = str(cmd_arguments.port)
    dist.init_process_group(
        backend=cmd_arguments.backend,
        world_size=cmd_arguments.ws,
        rank=cmd_arguments.rank,
    )


def cleanup_process_group(cmd_arguments):
    dist.destroy_process_group()


def main(cmd_arguments):
    init_process_group(cmd_arguments)
    run(cmd_arguments)
    cleanup_process_group(cmd_arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ws", type=int, required=True, help="world size.")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    cmd_arguments = parser.parse_args()
    main(cmd_arguments)
