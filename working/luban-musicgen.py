import time
import torch

from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers import AutoTokenizer


model = "/home/wnr/llms/musicgen-large"

processor = AutoProcessor.from_pretrained(model)
model = MusicgenForConditionalGeneration.from_pretrained(model).to("cuda:0", torch.float16)

inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    # text=["80s pop track with bassy drums and synth"],
    padding=True,
    return_tensors="pt",
).to("cuda:0")


# [2, 13]
print(inputs["input_ids"].size())

for max_new_tokens in [1000, 2000]:
    s = time.time()
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    t = time.time()
    r"""
    L40
    --------------------
    1000
    <class 'torch.Tensor'>
    torch.Size([2, 1, 638080])
    30.951239824295044
    --------------------
    2000
    <class 'torch.Tensor'>
    torch.Size([2, 1, 1278080])
    64.32788109779358
    """
    r"""
    L20
    --------------------
    1000
    <class 'torch.Tensor'>
    torch.Size([2, 1, 638080])
    25.9243323802948
    --------------------
    2000
    <class 'torch.Tensor'>
    torch.Size([2, 1, 1278080])
    60.49712920188904
    """
    print("-" * 20)
    print(max_new_tokens)
    print(type(audio_values))
    print(audio_values.size())
    print(t - s)

# import scipy

# sampling_rate = model.config.audio_encoder.sampling_rate
# scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

R"""
/home/olafxiong/local/cpython/lib/python3.10/site-packages/transformers/models/musicgen/modeling_musicgen.py:2664
            # 11. run sample
            outputs = self._sample(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

            # input_ids: torch.Tensor, S = [8, 1] NOTE: S = [batch * 4, 1]
            # model_kwargs:
                {
                    "attention_mask": torch.Tensor, S = [4, 13] NOTE: S = [batch * 2, seq_len]
                    "guidance_scale": float, 3.0
                    "encoder_outputs": transformers.modeling_outputs.BaseModelOutput.
                                       Only contains last_hidden_state, torch.Tensor, S = [batch * 2, seq_len, 768]
                                       NOTE:batch维度pad到4.
                    "decoder_delay_pattern_mask": torch.Tensor, S = [8, 257]
                }

input_ids: [batch_size * num_codebooks, 1], 第一次value为decoder_input_ids_start
model_kwargs["encoder_outputs"]: bach_size * 2，原因是guidance_scale。细节未知。

"""