"""run.py:"""

# https://pytorch.org/tutorials/intermediate/dist_tuto.html#writing-distributed-applications-with-pytorch

#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size):
    """Distributed function to be implemented later."""
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print("Rank 0 started sending")
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print("Rank 1 started receiving")
    req.wait()
    print("Rank ", rank, " has data ", tensor[0])


def init_process(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    # dist.init_process_group(backend, rank=-1, world_size=-1)
    dist.init_process_group(backend, rank=rank, world_size=size)
    # dist.init_process_group(backend)
    print(rank, dist.get_world_size(), dist.get_rank())
    fn(rank, size)


if __name__ == "__main__":
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
