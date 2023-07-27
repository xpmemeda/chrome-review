import hello_world


def function_1(a, b, c, *args, x, y, z):
    return a, b, c, *args, x, y, z


hello_world.call_python_function_with_args_and_kwargs(function_1)


def function_2(x):
    return x


hello_world.list(function_2)
hello_world.dict(function_2)
hello_world.custom(function_2)
