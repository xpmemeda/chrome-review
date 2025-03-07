import zmq
import argparse
import time


def main(cmd_arguments):
    if cmd_arguments.server:
        receiver = zmq.Context().socket(zmq.PULL)
        receiver.connect("tcp://localhost:5555")  # 连接到 PUSH 端
        while True:
            msg = receiver.recv_string()
            print("Received:", msg)
            time.sleep(1)


    if cmd_arguments.client:
        sender = zmq.Context().socket(zmq.PUSH)
        sender.bind("tcp://*:5555")
        for i in range(10):
            sender.send_string(f"Task {i}")


if __name__ == "__main__":
    cmd_arguments = argparse.ArgumentParser()
    cmd_arguments.add_argument("--server", action="store_true")
    cmd_arguments.add_argument("--client", action="store_true")
    cmd_arguments = cmd_arguments.parse_args()
    main(cmd_arguments)
