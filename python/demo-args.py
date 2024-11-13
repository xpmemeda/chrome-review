import argparse


def main(args):
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--str", type=str)
    parser.add_argument("--int", type=int)
    parser.add_argument("--list-str", nargs="+", type=str)
    parser.add_argument("--list-int", nargs="+", type=int)
    args = parser.parse_args()
    main(args)
