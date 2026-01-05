import numpy as np
import h5py

# 写入
x = (np.arange(12, dtype=np.float32) / 10).reshape(3, 4)
x = np.random.random(size=[2, 4, 5]).astype(np.float16)

with h5py.File("demo.h5", "w") as f:
    # compression="gzip" 会启用压缩（h5py 会自动选择/推断合理 chunk）
    dset = f.create_dataset("x", data=x, compression="gzip", compression_opts=1)
    dset.attrs["note"] = "written by python"

# 读取
with h5py.File("demo.h5", "r") as f:
    y = f["x"][...]          # numpy.ndarray
    print(type(y))
    print("python read shape:", y.shape, "dtype:", y.dtype)
    print(y)
