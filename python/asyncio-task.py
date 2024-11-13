import asyncio


class Consumer:
    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    def __aiter__(self):
        r"""
        NOTE: async for 语法对象必须实现 __aiter__ 接口，返回一个可异步迭代对象
        """
        print("__aiter__")
        return self

    async def __anext__(self):
        r"""
        NOTE: "异步迭代对象"的表示就是实现 async def __anext__ 接口，return 对象或者 raise StopAsyncIteration()
        """
        print("__anext__")
        x = await self._queue.get()
        if x == 4:
            raise StopAsyncIteration()
        return x


async def fn1():
    print("fn1")


async def fn3(queue):
    await queue.put(3)
    # NOTE: asyncio.sleep(0) 是阻塞操作，会交出执行权限。
    await asyncio.sleep(0)
    await queue.put(4)
    print("fn3")


async def fn2(queue):
    print_asyncio_tasks()
    # NOTE: await只有在阻塞时才会交出执行权限，这里queue.put(1)不阻塞，因此会继续执行。
    # NOTE: await可以看成是async def函数的普通调用。调用方会直接进入到被调函数，直到被阻塞。
    await queue.put(1)
    await queue.put(2)
    c = Consumer(queue)
    # NOTE: async for 也是只在 __anext__ 阻塞时交出执行权限，不然会一直调用循环，直到阻塞或者接收到StopAsyncIteration结束循环
    async for x in c:
        print(x)
    print("fn2")


def print_asyncio_tasks():
    print("------------------------------------------------")
    all_tasks = asyncio.all_tasks()
    for task in all_tasks:
        print(
            f"Task: {task}, State: {task._state}"
        )  # _state 是内部属性，通常不建议直接访问


async def main():
    queue = asyncio.Queue()
    # NOTE: 这个时间点，Task只有asyncio.run启动的main()，后面则会有fn2,fn1,fn3这些由gather启动的task。
    print_asyncio_tasks()
    # NOTE: 首次执行按顺序，先fn2，再fn1，最后fn3
    await asyncio.gather(fn2(queue=queue), fn1(), fn3(queue=queue))


if __name__ == "__main__":
    asyncio.run(main())
