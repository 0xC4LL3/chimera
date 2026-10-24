# Stand-in for flash-attn's interface module, for SkinTokens: its calls are plain scaled dot-product
# attention, so PyTorch's own does the job where no flash-attn build exists.
import torch


def flash_attn_func(q, k, v, causal=False, softmax_scale=None, **_):
    # flash-attn layout (batch, seq, heads, dim) in and out; returns (out, lse) like its v3 API.
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2), None
