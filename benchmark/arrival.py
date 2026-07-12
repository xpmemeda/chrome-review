import dataclasses
import random
import typing as ty


@dataclasses.dataclass(frozen=True)
class ScheduledRequest:
    req_idx: int
    scheduled_at: float


class ArrivalPlanner:
    def plan(self, num_requests: int) -> ty.List[ScheduledRequest]:
        raise NotImplementedError


class PoissonPlanner(ArrivalPlanner):
    def __init__(self, qps: float, seed: int) -> None:
        self.qps = qps
        self.seed = seed

    def plan(self, num_requests: int) -> ty.List[ScheduledRequest]:
        rng = random.Random(self.seed)
        scheduled_at = 0.0
        scheduled_times = []
        for req_idx in range(num_requests):
            if req_idx:
                scheduled_at += rng.expovariate(self.qps)
            scheduled_times.append(scheduled_at)

        if len(scheduled_times) <= 1 or scheduled_times[-1] <= 0.0:
            return [
                ScheduledRequest(req_idx, scheduled_at)
                for req_idx, scheduled_at in enumerate(scheduled_times)
            ]

        target_last_scheduled_at = (len(scheduled_times) - 1) / self.qps
        scale = target_last_scheduled_at / scheduled_times[-1]
        return [
            ScheduledRequest(req_idx, scheduled_at * scale)
            for req_idx, scheduled_at in enumerate(scheduled_times)
        ]


class ConstantRatePlanner(ArrivalPlanner):
    def __init__(self, qps: float) -> None:
        self.interval = 1.0 / qps

    def plan(self, num_requests: int) -> ty.List[ScheduledRequest]:
        return [
            ScheduledRequest(req_idx, req_idx * self.interval)
            for req_idx in range(num_requests)
        ]
