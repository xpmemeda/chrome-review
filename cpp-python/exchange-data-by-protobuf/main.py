import cr_pb2

with open("cpp.pb", "rb") as f:
    x = cr_pb2.Cr()
    x.ParseFromString(f.read())
    print(x.string)
    print(type(x.string))
    print(x.bytes)
    print(x.bytes.decode())
    print(type(x.bytes))
