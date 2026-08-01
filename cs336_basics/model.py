import torch
import torch.nn as nn
from einops import rearrange, einsum


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int | None = None,
            rope_theta=None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype) # [bsz, seq_pos] -> seq
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, # norm -> 4d, attn -> 
                             d_ff, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = Linear(d_model, vocab_size, dtype=dtype, device=device)


    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        norm_x = self.ln_final(x)
        logits = self.lm_head(norm_x)  # [bsz, ..., seq_len, vocab_size]
        return logits


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            d_ff: int | None = None,
            rope_theta=None,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.attn = MultiHeadCausalSelfAttention(d_model, num_heads, max_seq_len, rope_theta=rope_theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUMLP(d_model, d_ff=d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        norm_x = self.ln1(x)
        res_x = self.attn(norm_x, token_positions)
        y = x + res_x
        z = y + self.ffn(self.ln2(y))
        return z

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        var = 2 / (in_features + out_features)
        self.weight = nn.Parameter(
            _init_weights(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                var,
            ))


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            _init_weights(
                torch.empty(
                    num_embeddings, embedding_dim, device=device, dtype=dtype), 1))

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) ** 0.5
        normed_x = x / rms
        res = normed_x * self.weight
        return res.to(in_dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = round((d_model * 8/3) / 64) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.w1(x)
        hidden = silu(hidden)
        gate = self.w3(x)
        gated_hidden = hidden * gate
        out = self.w2(gated_hidden)
        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        # theta_array is shape [max_seq_len, d_k // 2]
        seq_len_vector = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inverse_frequencies = torch.tensor([theta ** -(2 * k / d_k) for k in range(d_k // 2)], device=device, dtype=torch.float32)
        theta_array = einsum(seq_len_vector, inverse_frequencies, "seq_len, half_d -> seq_len half_d")
        self.register_buffer("cos_array", torch.cos(theta_array))
        self.register_buffer("sin_array", torch.sin(theta_array))


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... seq_len (d_half couple) -> ... seq_len d_half couple", couple=2)
        x_first = x[..., 0] # [bsz, seq_len, d_half]
        x_second = x[..., 1]
        sin_x1 = self.sin_array[token_positions] * x_first
        cos_x1 = self.cos_array[token_positions] * x_first
        sin_x2 = self.sin_array[token_positions] * x_second
        cos_x2 = self.cos_array[token_positions] * x_second

        y_first = cos_x1 - sin_x2 # [bsz, seq_len, d_half]
        y_second = sin_x1 + cos_x2
        y = torch.stack((y_first, y_second), dim=-1)
        y = rearrange(y, "... seq_len d_half couple -> ... seq_len (d_half couple)")
        return y


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, max_seq_len: int, rope_theta=None, device=None, dtype=None):
        super().__init__()
        assert d_model >= heads
        assert d_model % heads == 0
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask)

        if rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(rope_theta, d_model // heads, max_seq_len, device)
        else:
            self.rope = None
        self.heads = heads


    def forward(self, tensor: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = tensor.shape[-2]
        Q = self.q_proj(tensor)
        K = self.k_proj(tensor)
        V = self.v_proj(tensor)
        Q = rearrange(Q, "bsz ... seq_len (heads d_head) -> bsz ... heads seq_len d_head", heads = self.heads)
        K = rearrange(K, "bsz ... seq_len (heads d_head) -> bsz ... heads seq_len d_head", heads = self.heads)
        if token_positions is not None and self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        V = rearrange(V, "bsz ... seq_len (heads d_head) -> bsz ... heads seq_len d_head", heads = self.heads)
        mask = self.causal_mask[:seq_len, :seq_len]
        assert mask.shape == (seq_len, seq_len)
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "bsz ... heads seq_len d_head -> bsz ... seq_len (heads d_head)", heads = self.heads)
        out = self.output_proj(out)
        return out


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x


def _init_weights(tensor: torch.Tensor, var: float) -> torch.Tensor:
    std = var ** 0.5
    return torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)


def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    max_tensor, _ = tensor.max(dim=dim, keepdim=True)
    adjusted_tensor = tensor - max_tensor
    numerator = adjusted_tensor.exp()
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator


def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:

    d_k = Q.shape[-1]
    scores = einsum(Q, K, "bsz ... q_seq_len d_k, bsz ... k_seq_len d_k -> bsz ... q_seq_len k_seq_len")
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores.masked_fill_(~mask, float('-inf'))

    probs = softmax(scores, dim=-1)
    out = einsum(probs, V, "bsz ... q_seq_len k_seq_len, bsz ... k_seq_len d_v -> bsz ... q_seq_len d_v")
    return out

"""
-- Q1 --
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288 (the nearest multiple of 64 to 8/3 x 1, 600)

1. How much holds in memory?
embedding table = 4 bytes * 50,257 x 1,600 = 321,644,800 bytes = 0.3216 GB

each layer:
    2 norms: 2 * 4 bytes * 1,600 = 12,800 bytes = 0.0000128 GB
    q, k, v, o projections = 4 bytes * 4 * 1,600 * 1,600 = 0.04096 GB
    w1, w2, w3 = 4 bytes * 4,288 * 1,600 * 3 = 0.08232 GB

norm: 0.0000064 GB
output head = 4 bytes * 1,600 * 50,257 = 0.3216 GB

1,640,452,800 parameters

Total: 643.21MB + 48 * 123.29MB = 6.56 GB

-- Q2 --
We found earlier
FLOPS =
L * 
    (
      10 * seq * d_model + 
      4 * seq^2 * d_model +
      8 * seq * d_model^2 +
      5 * seq^2 * heads +
      6 * d_ff * d_model * seq +
      3 * d_ff * seq
    ) +
4 * d_model * seq + 2 * seq * d_model * vocab

Plugging in our numbers:
48 (16,384,000 + 6,710,886,400 + 20,971,520,000 + 131,072,000 + 42,152,755,200 + 13,172,736) + 6,553,600 + 164,682,137,600
= 3,524,486,600,000

Most expensive component is the LM head followed by the FFN

-- Q4 --
FLOPS =
L * 
    (
      10 * seq * d_model + 
      4 * seq^2 * d_model +
      8 * seq * d_model^2 +
      5 * seq^2 * head +
      6 * d_ff * d_model * seq +
      3 * d_ff * seq
    ) +
4 * d_model * seq + 2 * seq * d_model * vocab

Plugging in our numbers:

GPT-2 small (12 layers, 768 d_model, 12 heads, 2048 d_ff)
12 (7,864,320 + 3,221,225,472 + 4,831,838,208 + 5,242,880 + 20,233,322,496 + 13,172,736) + 3,145,728 + 79047426048
= 418,802,565,120 FLOPS


GPT-2 medium 
(24 layers, 1024 d_model, 16 heads)

GPT-2 large (36 layers, 1280 d_model, 20 heads). 
"""