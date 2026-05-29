from openai import OpenAI
import base64
import uuid
import httpx

base_url = "https://device-intelligence.bytedance.net/api/v1"
# model_name="memory_agent.seed_m8_2b5_server_v2"
# model_name="ocean.assistant.m12_12b_optimization"

# omni-demo-m13-v21.6  -->  ocean.assistant.m13_12b_cotv216_v2

model_name = "omni-demo-m13-2b5.xp.0525"
# omni-demo-m13-2b5.xp.0525 --> ocean.assistant.omniagent_m13_2b5_xptest0
client = OpenAI(
    base_url=base_url,
    api_key="empty",
    default_headers={
        "x-tt-env": "ppe_model_center",
        "x-use-ppe": "1",
    },
    http_client=httpx.Client(trust_env=False),
)
path = rf"C:\Users\Admin\Downloads\12b_test\1.jpg"
path = "/home/tiger/workspace/resources/cat.png"
path = "/home/tiger/workspace/resources/cat-632x1400.png.jpeg"
with open(path, "rb") as f:
    base64string = base64.b64encode(f.read()).decode("utf-8")
messages = [
    # {
    #     "role": "system",
    #     "content": "You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:\n<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here. \n \nYou have different modes of thinking:\nUnrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.\nEfficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking. \nNo think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. \nBudgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.\n\nBased on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.\n \nProvided Mode(s):\nNo think"
    # },
    {
        "content": [
            {
                "image_url": {"url": f"data:image/jpeg;base64,{base64string}"},
                "type": "image_url",
            },
            {"text": f"图片主要讲了什么?", "type": "text"},
        ],
        "role": "user",
    }
]
# 根据各个模型实际使用传入
inference_params = {
    "max_tokens": 15000,
    "temperature": 0,
    "top_p": 0.9,
    "extra_body": {
        "extra_body": {
            "repetition_penalty": 1.1,
            "top_k": 1,
        }
    },
}


def stream_to_text(stream):
    text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            text += delta.content
    return text


res = client.chat.completions.create(model=model_name, messages=messages, stream=True)
# print(res)
result = stream_to_text(res)
print(result)
