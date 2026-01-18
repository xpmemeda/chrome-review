import dataclasses
import numpy
import logging

from typing import Optional, List, Dict
from tokenizer import Tokenizer


@dataclasses.dataclass
class Metric:
    cid: Optional[str]
    ttft: float
    ttfs: float
    itl_list: List[float]
    e2e: float
    messages: Dict
    resp_text: str


class Metrics:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        self.num_prompt_tokens = []
        self.num_output_tokens = []
        self.ttfts = []
        self.ttfss = []
        self.itls = []
        self.e2es = []

        self.wesearch_block_numbers = []

    def update(self, metric: Metric):
        ttft = metric.ttft
        ttfs = metric.ttfs
        itl_list = metric.itl_list
        avg_itl = sum(itl_list) / len(itl_list)
        e2e = metric.e2e
        messages = metric.messages
        resp_text = metric.resp_text

        num_prompt_tokens = self.tokenizer.count(messages)
        num_output_tokens = self.tokenizer.count(resp_text)

        self.num_prompt_tokens.append(num_prompt_tokens)
        self.num_output_tokens.append(num_output_tokens)
        self.ttfts.append(ttft)
        self.ttfss.append(ttfs)
        self.itls.extend(itl_list)
        self.e2es.append(e2e)
        self.wesearch_block_numbers.append(
            sum(x > 0.5 for x in itl_list)
        )  # 500ms for wesearch.

        logging.warning(
            "Update cid=%s, Req: %d, Resp: %d, TTFT: %f, TTFS: %f, Avg ITL: %f, E2E: %f, RespTxt: %s",
            metric.cid,
            num_prompt_tokens,
            num_output_tokens,
            ttft,
            ttfs,
            avg_itl,
            e2e,
            repr(resp_text),
        )

        if len(self.num_prompt_tokens) % 50 == 0:
            logging.warning(f"{self}")

    def wesearch_metrics(self):
        if not self.num_prompt_tokens:
            return ""

        ttft_percentiles = [50, 90, 99, 100]
        ttftp50, ttftp90, ttftp99, ttftp100 = numpy.percentile(
            self.ttfts, ttft_percentiles
        )
        avgttft = numpy.mean(self.ttfts)

        itl_percentiles = [50, 90, 99, 99.2, 99.4, 99.6, 99.8, 100]
        itlp50, itlp90, itlp99, itlp992, itlp994, itlp996, itlp998, itlp100 = (
            numpy.percentile(self.itls, itl_percentiles)
        )
        avgitl = numpy.mean(self.itls)

        e2e_percentiles = [50, 90, 99, 100]
        e2ep50, e2ep90, e2ep99, e2ep100 = numpy.percentile(self.e2es, e2e_percentiles)
        avge2e = numpy.mean(self.e2es)

        block_numbers = [x for x in self.wesearch_block_numbers if x > 0]
        block_ratio = len(block_numbers) / len(self.wesearch_block_numbers)
        avg_block_number = numpy.mean(block_numbers)

        s1 = f"ttft: avg={avgttft:.3f}, p50={ttftp50:.3f}, p90={ttftp90:.3f}, p99={ttftp99:.3f}, p100={ttftp100:.3f}"
        s2 = f"itl: avg={avgitl:.3f}, p50={itlp50:.3f}, p90={itlp90:.3f}, p99={itlp99:.3f}, p99.2={itlp992:.3f}, p99.4={itlp994:.3f}, p99.6={itlp996:.3f}, p99.8={itlp998:.3f}, p100={itlp100:.3f}"
        s3 = f"e2e: avg={avge2e:.3f}, p50={e2ep50:.3f}, p90={e2ep90:.3f}, p99={e2ep99:.3f}, p100={e2ep100:.3f}"
        s4 = f"block_number: avg={avg_block_number:.3f}, ratio={block_ratio:.3%}"
        return f"\n{s1}\n{s2}\n{s3}\n{s4}"

    def __repr__(self):
        def avg(x: list):
            return sum(x) / len(x) if len(x) else 0.0

        avg_num_prompt_tokens = avg(self.num_prompt_tokens)
        avg_num_output_tokens = avg(self.num_output_tokens)
        avg_ttft = avg(self.ttfts)
        avg_ttfs = avg(self.ttfss)
        avg_itl = avg(self.itls)
        avg_e2e = avg(self.e2es)

        return (
            "Metrics(Avg Req: %f, Avg Resp: %f, Avg TTFT: %f, Avg TTFS: %s, Avg ITL: %s, Avg E2E: %f)"
            % (
                avg_num_prompt_tokens,
                avg_num_output_tokens,
                avg_ttft,
                avg_ttfs,
                avg_itl,
                avg_e2e,
            )
        )
