import sys
import hashlib

def md5(file_name):
    with open(file_name, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    print(file_md5)

if __name__ == "__main__":
    md5(sys.argv[1])
