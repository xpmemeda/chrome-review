import pdb
import argparse

from tensorrt_llm.builder import BuildConfig, build, Engine
from tensorrt_llm.models.automodel import MODEL_MAP, AutoModelForCausalLM
from tensorrt_llm.mapping import Mapping


def load_model_from_hf(hfmodel_path):
    model_cls = AutoModelForCausalLM.get_trtllm_model_class(hfmodel_path)
    model = model_cls.from_hugging_face(hfmodel_path, dtype="float16")
    return model


def main(args):
    model = load_model_from_hf(args.hfmodel_path)
    build_config = BuildConfig(
        max_batch_size=args.max_batch_size, opt_batch_size=args.max_batch_size
    )
    engine: Engine = build(model, build_config)
    engine.save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfmodel-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-batch-size", type=int, required=True)
    main(parser.parse_args())
