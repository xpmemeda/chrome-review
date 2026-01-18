import torch
import time

import kcuda.kernels as dbgkernels


def main():
    x = torch.empty(1024, device="cuda", dtype=torch.int32)
    print(hex(x.data_ptr()), flush=True)
    time.sleep(1)

    dbgkernels.oob_fill(x, torch.tensor([33], dtype=torch.int32))
    print(x)


if __name__ == "__main__":
    main()
