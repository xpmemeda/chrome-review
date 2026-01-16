import time
import aiojobs
import asyncio
import abc
import os
import typing as ty
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .datasets import Dataset
from .tokenizer import Tokenizer
from .metrics import MetricsRecorder
from .cli import BaseClient
from .limiter import ConcurrencyGate, Pacer

ClientFactory = ty.Callable[[], BaseClient]


class BaseOrch(abc.ABC):
    def __init__(
        self,
        client_factory: ClientFactory,
        n: int,
        start_at: float,
        dataset: Dataset,
        tokenizer: Tokenizer,
        pacer: Pacer,
        concurrency_gate: ConcurrencyGate,
        metrics_recorder: MetricsRecorder,
    ):
        self.client_factory = client_factory
        self.n = n
        self.start_at = start_at
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.pacer = pacer
        self.concurrency_gate = concurrency_gate
        self.metrics_recorder = metrics_recorder

    @abc.abstractmethod
    def run(self) -> float:
        raise NotImplementedError

    def wait_until_start(self) -> None:
        delay = self.start_at - time.time()
        if delay > 0:
            time.sleep(delay)

    async def async_wait_until_start(self) -> None:
        delay = self.start_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)


class CoOrch(BaseOrch):
    def __init__(
        self,
        client_factory: ClientFactory,
        n: int,
        start_at: float,
        dataset: Dataset,
        tokenizer: Tokenizer,
        pacer: Pacer,
        concurrency_gate: ConcurrencyGate,
        metrics_recorder: MetricsRecorder,
    ):
        super().__init__(
            client_factory=client_factory,
            n=n,
            start_at=start_at,
            dataset=dataset,
            tokenizer=tokenizer,
            pacer=pacer,
            concurrency_gate=concurrency_gate,
            metrics_recorder=metrics_recorder,
        )
        self.client = self.client_factory()

    def run(self):

        async def async_main():

            max_num_workers = 0
            num_workers = 0
            num_requests = 0

            async def send_request_and_report_metrics(*args, **kwargs):
                nonlocal num_workers

                metrics = await self.client.async_send_request(*args, **kwargs)
                self.metrics_recorder.update(metrics)

                num_workers -= 1
                self.concurrency_gate.release()

            async with aiojobs.Scheduler() as scheduler:
                await self.async_wait_until_start()

                last_worker_launch_time = time.time()

                while request := self.dataset.get():

                    if num_requests >= self.n:
                        break

                    await self.pacer.async_wait()
                    await self.concurrency_gate.async_acquire()

                    messages, sampling_params = request

                    await scheduler.spawn(
                        send_request_and_report_metrics(
                            num_requests, messages, sampling_params
                        )
                    )

                    num_workers += 1
                    num_requests += 1
                    max_num_workers = max(max_num_workers, num_workers)

                    logging.warning(
                        f"Add a worker, {num_workers=}, {num_requests=}, sleeped_seconds={time.time() - last_worker_launch_time}"
                    )
                    last_worker_launch_time = time.time()

        asyncio.run(async_main())
        return time.time()


class MtOrch(BaseOrch):
    def __init__(
        self,
        client_factory: ClientFactory,
        n: int,
        start_at: float,
        dataset: Dataset,
        tokenizer: Tokenizer,
        pacer: Pacer,
        concurrency_gate: ConcurrencyGate,
        metrics_recorder: MetricsRecorder,
    ):
        super().__init__(
            client_factory=client_factory,
            n=n,
            start_at=start_at,
            dataset=dataset,
            tokenizer=tokenizer,
            pacer=pacer,
            concurrency_gate=concurrency_gate,
            metrics_recorder=metrics_recorder,
        )
        self.client = self.client_factory()

    def run(self):

        max_num_workers = 0
        num_workers = 0
        num_requests = 0

        def cb(future):
            nonlocal num_workers
            num_workers -= 1

            self.concurrency_gate.release()

            self.metrics_recorder.update(future.result())

        with ThreadPoolExecutor() as executor:
            self.wait_until_start()

            last_worker_launch_time = time.time()

            while request := self.dataset.get():

                if num_requests >= self.n:
                    break

                self.pacer.wait()
                self.concurrency_gate.acquire()

                messages, sampling_params = request

                future = executor.submit(
                    self.client.send_request, num_requests, messages, sampling_params
                )
                future.add_done_callback(cb)

                num_workers += 1
                num_requests += 1
                max_num_workers = max(max_num_workers, num_workers)

                logging.warning(
                    f"Add a worker, {num_workers=}, {num_requests=}, sleeped_seconds={time.time() - last_worker_launch_time}"
                )
                last_worker_launch_time = time.time()

            while num_workers:
                logging.warning(f"{num_workers=}, waiting...")
                time.sleep(0.01)
        return time.time()


class MpOrch(BaseOrch):
    def __init__(
        self,
        client_factory: ClientFactory,
        n: int,
        start_at: float,
        dataset: Dataset,
        tokenizer: Tokenizer,
        pacer: Pacer,
        concurrency_gate: ConcurrencyGate,
        metrics_recorder: MetricsRecorder,
    ):
        super().__init__(
            client_factory=client_factory,
            n=n,
            start_at=start_at,
            dataset=dataset,
            tokenizer=tokenizer,
            pacer=pacer,
            concurrency_gate=concurrency_gate,
            metrics_recorder=metrics_recorder,
        )

        self.executor = ProcessPoolExecutor(
            initializer=MpOrch.init_cli, initargs=(self.client_factory,)
        )
        self.max_workers = self.executor._max_workers or (os.cpu_count() or 1)
        self._executor_closed = False

    def close(self):
        if not self._executor_closed:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self._executor_closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def init_cli(client_factory: ClientFactory):
        MpOrch.cli = client_factory()

    @staticmethod
    def probe_worker() -> int:
        return os.getpid()

    @staticmethod
    def send_request(
        req_idx: int,
        messages: ty.List,
        sampling_params: ty.Optional[ty.Dict[str, ty.Any]] = None,
    ):
        return MpOrch.cli.send_request(req_idx, messages, sampling_params)

    def prepare_workers(self) -> None:
        probe_futures = [
            self.executor.submit(MpOrch.probe_worker) for _ in range(self.max_workers)
        ]
        for future in probe_futures:
            future.result()

    def run(self) -> float:
        self.prepare_workers()
        self.wait_until_start()

        max_num_workers = 0
        num_workers = 0
        num_requests = 0

        def cb(future):
            nonlocal num_workers
            num_workers -= 1

            self.concurrency_gate.release()

            self.metrics_recorder.update(future.result())

        last_worker_launch_time = time.time()

        while request := self.dataset.get():

            if num_requests >= self.n:
                break

            self.pacer.wait()
            self.concurrency_gate.acquire()

            messages, sampling_params = request

            if num_workers >= self.max_workers:
                self.concurrency_gate.release()
                raise RuntimeError(
                    "MpOrch forbids ProcessPoolExecutor backlog; all workers are busy, "
                    "but another task is about to be submitted. Reduce --qps, reduce "
                    "--max-batch, or switch orchestrator."
                )

            future = self.executor.submit(
                MpOrch.send_request, num_requests, messages, sampling_params
            )
            future.add_done_callback(cb)

            num_workers += 1
            num_requests += 1
            max_num_workers = max(max_num_workers, num_workers)

            logging.warning(
                f"Add a worker, {num_workers=}, {num_requests=}, sleeped_seconds={time.time() - last_worker_launch_time}"
            )
            last_worker_launch_time = time.time()

        while num_workers:
            logging.warning(f"{num_workers=}, waiting...")
            time.sleep(0.01)
        return time.time()
