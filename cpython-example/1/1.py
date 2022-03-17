class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __call__(self):
        return "woof...woof..."

mumu = Dog("mumu", 3)
print("%s is %d years old" %(mumu.name, mumu.age))
print(mumu())
