import abc
import asyncio
import logging
import threading
import time

import numpy


class Pacer(abc.ABC):
    @abc.abstractmethod
    def wait(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def async_wait(self) -> None:
        raise NotImplementedError


class NoopPacer(Pacer):
    def wait(self) -> None:
        return

    async def async_wait(self) -> None:
        return


class QpsPacer(Pacer):
    def __init__(self, qps: float, n: int, distribution: str = "Poisson"):
        """
        distribution: Poisson or Uniform
        """
        self.qps = qps

        if distribution == "Poisson":
            rng = numpy.random.default_rng(seed=0)
            sleep_seconds = rng.exponential(scale=1.0 / qps, size=n).tolist()
        else:
            sleep_seconds = [1.0 / self.qps] * n

        target_total = n / qps
        actual_total = sum(sleep_seconds)
        scale = target_total / actual_total
        sleep_seconds = [x * scale for x in sleep_seconds]

        total = sum(sleep_seconds)
        logging.warning(f"QpsPacer qps {n / total : .2f}")

        self.next_t = [sleep_seconds[0]]
        for i in range(1, n):
            self.next_t.append(self.next_t[-1] + sleep_seconds[i])

        self.base_t = None
        self.next_idx = 0

    def _sleep_seconds(self) -> float:
        if self.base_t is None:
            self.base_t = time.perf_counter()

        now = time.perf_counter()
        sleep_seconds = self.next_t[self.next_idx] + self.base_t - now
        self.next_idx += 1
        return sleep_seconds

    def wait(self) -> None:
        sleep_seconds = self._sleep_seconds()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    async def async_wait(self) -> None:
        sleep_seconds = self._sleep_seconds()
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)


class ConcurrencyGate(abc.ABC):
    @abc.abstractmethod
    def acquire(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def async_acquire(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError


class NoopConcurrencyGate(ConcurrencyGate):
    def acquire(self) -> None:
        return

    async def async_acquire(self) -> None:
        return

    def release(self) -> None:
        return


class SemaphoreConcurrencyGate(ConcurrencyGate):
    def __init__(self, max_concurrency: int):
        self.max_concurrency = max_concurrency
        self.cur_concurrency = 0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                if self.cur_concurrency < self.max_concurrency:
                    self.cur_concurrency += 1
                    return
            time.sleep(0.005)

    async def async_acquire(self) -> None:
        while True:
            with self.lock:
                if self.cur_concurrency < self.max_concurrency:
                    self.cur_concurrency += 1
                    return
            await asyncio.sleep(0.005)

    def release(self) -> None:
        with self.lock:
            self.cur_concurrency -= 1
