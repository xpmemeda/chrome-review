import asyncio


async def fun(name, number):
    for i in range(number):
        await asyncio.sleep(1)
        print("{}: {}".format(name, i))
    return


async def main():
    x = fun("A", 2)
    y = fun("B", 3)
    taskx = asyncio.create_task(x)
    tasky = asyncio.create_task(y)

    await taskx
    await tasky

    return


if __name__ == "__main__":
    asyncio.run(main())
