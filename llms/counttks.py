import argparse
import tiktoken

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Text to count tokens for", required=True)
    args = parser.parse_args()

    encoding = tiktoken.get_encoding("o200k_base")
    text = args.text
    token_ids = encoding.encode(text)

    print(len(token_ids))
