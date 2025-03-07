import redis
import argparse


def main(cmd_arguments):
    redis_server = redis.Redis(
        host=cmd_arguments.redis_host, port=cmd_arguments.redis_port
    )
    redis_server.set("redis-demo", "redis-demo", nx=True, ex=10)
    v = redis_server.get("redis-demo")
    print(v)


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--redis-host", type=str, default="127.0.0.1")
    cmd_parser.add_argument("--redis-port", type=int, default=6379)
    cmd_arguments = cmd_parser.parse_args()
    main(cmd_arguments)
