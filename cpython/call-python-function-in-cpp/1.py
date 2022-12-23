import numpy as np
import hello_world


def print_hello_world():
    print("hello world")


hello_world.register_function(print_hello_world)
hello_world.call_function()
