import argparse
import kazoo

from kazoo.client import KazooClient
from kazoo.recipe.watchers import ChildrenWatch


def main(cmd_arguments):
    zk = KazooClient(hosts=f"{cmd_arguments.zk_host}:{cmd_arguments.zk_port}")
    zk.start()
    zk.ensure_path("/zk-demo")

    ChildrenWatch(zk, "/zk-demo", lambda x: print("cb:\n", x))

    zk.create(path="/zk-demo/zk-demo-", value=b"zk-demo", ephemeral=True)
    try:
        zk.create(path="/zk-demo/zk-demo-", value=b"zk-demo", ephemeral=True)
    except kazoo.exceptions.NodeExistsError as e:
        print("error.")
    zk.delete(path="/zk-demo/zk-demo-")

    zk.create(path="/zk-demo/zk-demo-", value=b"zk-demo", ephemeral=True, sequence=True)

    for ch in zk.get_children("/zk-demo"):
        chv = zk.get(f"/zk-demo/{ch}")
        print(chv)

    zk.stop()


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--zk-host", type=str, default="127.0.0.1")
    cmd_parser.add_argument("--zk-port", type=int, default=2181)
    cmd_arguments = cmd_parser.parse_args()
    main(cmd_arguments)
