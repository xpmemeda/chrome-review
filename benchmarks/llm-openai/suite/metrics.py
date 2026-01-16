import dataclasses
import numpy
import logging
from typing import Optional, List, Dict

from .tokenizer import Tokenizer


@dataclasses.dataclass
class Metrics:
    cid: str
    ttft: float
    itl_list: List[float]
    e2e: float
    messages: Dict
    resp_text: str


class MetricsRecorder:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        self.num_prompt_tokens = []
        self.num_output_tokens = []
        self.ttfts = []
        self.itls = []
        self.e2es = []

    def update(self, metrics: Metrics):
        ttft = metrics.ttft
        itl_list = metrics.itl_list
        avg_itl = sum(itl_list) / len(itl_list) if itl_list else 0.0
        e2e = metrics.e2e
        messages = metrics.messages
        resp_text = metrics.resp_text

        num_prompt_tokens = self.tokenizer.count(messages)
        num_output_tokens = self.tokenizer.count(resp_text)

        self.num_prompt_tokens.append(num_prompt_tokens)
        self.num_output_tokens.append(num_output_tokens)
        self.ttfts.append(ttft)
        if itl_list:
            self.itls.extend(itl_list)
        self.e2es.append(e2e)

        logging.warning(
            "update cid=%s, req_len: %d, resp_len: %d, TTFT: %f, Avg ITL: %f, E2E: %f, RespTxt: %s",
            metrics.cid,
            num_prompt_tokens,
            num_output_tokens,
            ttft,
            avg_itl,
            e2e,
            repr(resp_text),
        )

        if len(self.num_prompt_tokens) % 50 == 0:
            logging.warning(f"{self}")

    def report_table(self):
        if not self.num_prompt_tokens:
            return "EMPTY"

        import numpy as np

        # ---------- 计算 ----------
        ttft_p = [50, 90, 99, 100]
        ttftp50, ttftp90, ttftp99, ttftp100 = np.percentile(self.ttfts, ttft_p)
        avgttft = np.mean(self.ttfts)

        itl_p = [50, 90, 99, 99.2, 99.4, 99.6, 99.8, 100]
        if self.itls:
            (
                itlp50,
                itlp90,
                itlp99,
                itlp992,
                itlp994,
                itlp996,
                itlp998,
                itlp100,
            ) = np.percentile(self.itls, itl_p)
            avgitl = np.mean(self.itls)
        else:
            itlp50 = itlp90 = itlp99 = itlp992 = itlp994 = itlp996 = itlp998 = (
                itlp100
            ) = 0.0
            avgitl = 0.0

        e2e_p = [50, 90, 99, 100]
        e2ep50, e2ep90, e2ep99, e2ep100 = np.percentile(self.e2es, e2e_p)
        avge2e = np.mean(self.e2es)

        # ---------- 表格数据 ----------
        headers = [
            "name",
            "avg",
            "p50",
            "p90",
            "p99",
            "p99.2",
            "p99.4",
            "p99.6",
            "p99.8",
            "p100",
        ]

        rows = [
            [
                "ttft",
                avgttft,
                ttftp50,
                ttftp90,
                ttftp99,
                None,
                None,
                None,
                None,
                ttftp100,
            ],
            [
                "itl",
                avgitl,
                itlp50,
                itlp90,
                itlp99,
                itlp992,
                itlp994,
                itlp996,
                itlp998,
                itlp100,
            ],
            [
                "e2e",
                avge2e,
                e2ep50,
                e2ep90,
                e2ep99,
                None,
                None,
                None,
                None,
                e2ep100,
            ],
        ]

        # ---------- 格式化 ----------
        def fmt(v):
            return "-" if v is None else f"{v:.3f}"

        col_width = 8
        sep = "+" + "+".join(["-" * col_width for _ in headers]) + "+"

        def render_row(items):
            return "|" + "|".join(f"{str(i):^{col_width}}" for i in items) + "|"

        out = []
        out.append(sep)
        out.append(render_row(headers))
        out.append(sep)

        for r in rows:
            out.append(render_row([r[0]] + [fmt(v) for v in r[1:]]))

        out.append(sep)

        return "\n" + "\n".join(out)
