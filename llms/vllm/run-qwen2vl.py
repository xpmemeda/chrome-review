import sys
import io
import torch
import requests
import base64

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image


def main(model_dir):
    llm = LLM(
        model=model_dir,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    original_width, original_height = image.size

    image1 = image.resize((original_width // 4, original_height // 4))
    image2 = image.resize((original_width // 5, original_height // 5))

    byte_io = io.BytesIO()
    image2.save(byte_io, format="PNG")
    byte_io.seek(0)
    image2_bytes = byte_io.read()
    image2_b64 = base64.b64encode(image2_bytes).decode("utf-8")

    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": f"data:imagebase64,{image2_b64}"},
                {
                    "type": "text",
                    "text": "List two things these images have in common.",
                },
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_dir)
    prompt = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)


if __name__ == "__main__":
    model_dir = sys.argv[1]
    main(model_dir)
