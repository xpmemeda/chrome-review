import cr

x = cr.get_str()
print(x)
print(type(x))
print(x.encode("utf-8"))
x = cr.get_bytes()
print(x)
print(type(x))
print(x.decode())
