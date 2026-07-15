import typing as ty

import dataset as dataset_lib
from metrics import RequestMetrics

JsonDict = ty.Dict[str, ty.Any]


class MockClient:
    async def send_request(
        self,
        req_idx: int,
        request: dataset_lib.StdChatApiRequest,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        del request, sampling_params
        return RequestMetrics(req_idx, True, 0.0, 0.0, 0, 0, 0)
