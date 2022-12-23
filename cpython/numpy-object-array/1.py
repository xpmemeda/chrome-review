import hello_world
import numpy as np

x = np.array([1, 2]).astype(object)
print(x.dtype)
hello_world.print(x)
print()

x = np.array(["hello", "world"]).astype(object)
print(x.dtype)
hello_world.print(x)
print()

x = np.array([1, 2])
print(x.dtype)
hello_world.print(x)
print()

x = np.array(["hello", "world"])
print(x.dtype)
hello_world.print(x)
print()
