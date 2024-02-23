import math
import torch
import argparse
import torch.onnx
from transformers import BertModel, BertTokenizer


def scaled_dot_product_attention_symbolic(
    g,
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa=False,
):
    r"""
    Custom operator export function.The scaled_dot_product_attention operator is only available in ONNX opset versions greater than 14.
    Therefore, a custom implementation is provided when the target opset version is less than 14.

    hf source code at transformers/models/bert/modeling_bert.py:439
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )
    """
    input_shape = torch.onnx.symbolic_helper._get_tensor_sizes(query)
    assert len(input_shape) == 4 and isinstance(input_shape[3], int)
    head_size = input_shape[3]
    softmax_scale = g.op(
        "Constant", value_t=torch.tensor(1.0 / math.sqrt(float(head_size)))
    )

    attn_weights = g.op("MatMul", query, key)
    attn_weights = g.op("Mul", attn_weights, softmax_scale)
    attn_weights = g.op("Add", attn_weights, attn_mask)
    attn_weights = g.op("Softmax", attn_weights)
    output = g.op("MatMul", attn_weights, value)
    return output


def export_onnx(hf_model, onnx_model):
    model = BertModel.from_pretrained(hf_model)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(hf_model)

    text = ["Hello, how are you?", "I'm fine."]
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]

    torch.onnx.export(
        model,
        (input_ids, token_type_ids, attention_mask),
        onnx_model,
        input_names=["input_ids", "token_type_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=11,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str, required=True)
    parser.add_argument("--onnx-model", type=str, required=True)
    arguments = parser.parse_args()

    torch.onnx.register_custom_op_symbolic(
        "aten::scaled_dot_product_attention", scaled_dot_product_attention_symbolic, 11
    )
    export_onnx(arguments.hf_model, arguments.onnx_model)
