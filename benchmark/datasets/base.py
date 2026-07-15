import typing as ty

JsonDict = ty.Dict[str, ty.Any]
Messages = ty.List[JsonDict]
StdChatApiRequest = JsonDict


class VlmDataset:
    def get(self, req_idx: int) -> StdChatApiRequest:
        raise NotImplementedError()

    def warmup(self, size: int) -> ty.List[StdChatApiRequest]:
        return [self.get(i) for i in range(size)]

    def size(self) -> int:
        raise NotImplementedError()
