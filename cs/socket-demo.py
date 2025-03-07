import socket
import argparse


def socket_server(host: str, port: int):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    while True:
        client_socket, client_address = server_socket.accept()
        data: bytes = client_socket.recv(1024)
        print("recv data(%s) from client(%s)" % (data.decode(), client_address))
        client_socket.close()


def socket_client(host: str, port: int):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    client_socket.sendall("Hello".encode())
    client_socket.close()


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--host", type=str, default="127.0.0.1")
    cmd_parser.add_argument("--port", type=int, default=1899)
    cmd_parser.add_argument("--run-server", action="store_true")
    cmd_arguments = cmd_parser.parse_args()
    if cmd_arguments.run_server:
        cmd_arguments.host = "0.0.0.0"
        socket_server(cmd_arguments.host, cmd_arguments.port)
    else:
        socket_client(cmd_arguments.host, cmd_arguments.port)
