import io


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents[:-1]  # ignore the last '\n'


def print_green(*args, **kwargs):
    s = print_to_string(*args, **kwargs)
    print(bcolors.OKGREEN + s + bcolors.UNDERLINE)


def print_fail(*args, **kwargs):
    s = print_to_string(*args, **kwargs)
    print(bcolors.FAIL + s + bcolors.UNDERLINE)


if __name__ == "__main__":
    print_green("hello, world")
    print_fail("hello, world")
