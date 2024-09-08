import math
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass
class ModelConfig:
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    n_text_ctx: int
    n_vocab: Optional[int] = None

    def set_vocab_size(self, vocab_size: int):
        self.n_vocab = vocab_size


class KVCache(nn.Module):
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dtype = dtype
        self.cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(self.cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(self.cache_shape, dtype=dtype))

    def get(self):
        return self.k_cache, self.v_cache

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor):
        # input_pos: [S], k_val, v_val: [B, H, L, D]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

    def zero(self):
        self.k_cache.zero_()
        self.v_cache.zero_()


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class CrossAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        assert n_state % n_head == 0, "n_head does not evenly divide n_state"

        self.n_head = n_head
        self.d_head = n_state // n_head
        self.query = nn.Linear(n_state, n_state, bias=False)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state, bias=False)
        self.out = nn.Linear(n_state, n_state, bias=False)
        self.kv_cache: KVCache | None = None

    def get_kv(
        self,
        encoder_outputs: torch.Tensor,
        encoder_cache_pos: torch.Tensor,
        use_cache: bool,
    ):
        assert self.kv_cache is not None, "No kv_cache"
        if use_cache:
            return self.kv_cache.get()

        k: torch.Tensor = self.key(encoder_outputs[:, encoder_cache_pos])
        v: torch.Tensor = self.value(encoder_outputs[:, encoder_cache_pos])

        # Reshape for correct format
        batch_size, source_seq_len, _ = k.shape
        k = k.view(batch_size, source_seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, source_seq_len, self.n_head, self.d_head).transpose(1, 2)

        k, v = self.kv_cache.update(k_val=k, v_val=v, input_pos=encoder_cache_pos)

        return k, v

    def forward(
        self,
        input_ids: Tensor,
        encoder_outputs: Tensor,
        encoder_cache_pos: Tensor,
        use_cache: bool,
    ):
        q = self.query(input_ids)
        batch_size, target_seq_len, _ = q.shape
        q = q.view(batch_size, target_seq_len, self.n_head, self.d_head).transpose(1, 2)

        k, v = self.get_kv(encoder_outputs, encoder_cache_pos, use_cache)
        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False,
        )
        wv = wv.transpose(1, 2).reshape(
            batch_size,
            target_seq_len,
            self.n_head * self.d_head,
        )

        return self.out(wv)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        assert n_state % n_head == 0, "n_head does not evenly devide n_state"

        self.n_state = n_state
        self.n_head = n_head
        self.d_head = n_state // n_head
        self.out = nn.Linear(n_state, n_state, bias=False)
        self.kv_cache: KVCache | None = None

        # Add this back after
        self.combined_qkv = nn.Linear(n_state, 3 * n_state, bias=False)
        self._register_load_state_dict_pre_hook(self.combined_qkv_hook)

    def get_kv(self, k: Tensor, v: Tensor, cache_pos: Tensor):
        assert self.kv_cache is not None, "No kv_cache"
        k, v = self.kv_cache.update(k_val=k, v_val=v, input_pos=cache_pos)

        return k, v

    def combined_qkv_hook(self, state_dict: Any, prefix: str, *args: Any):
        if prefix + "query.weight" in state_dict:
            wq = state_dict.pop(prefix + "query.weight")
            wk = state_dict.pop(prefix + "key.weight")
            wv = state_dict.pop(prefix + "value.weight")
            state_dict[prefix + "combined_qkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        input_ids: Tensor,
        mask: Optional[Tensor] = None,
        cache_pos: Optional[Tensor] = None,
    ):
        combined_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            self.combined_qkv(input_ids).split(
                [self.n_state, self.n_state, self.n_state], dim=-1
            )
        )
        q, k, v = combined_qkv
        batch_size, target_seq_len, _ = q.shape
        q = q.view(batch_size, target_seq_len, self.n_head, self.d_head).transpose(1, 2)

        batch_size, source_seq_len, _ = k.shape
        k = k.view(batch_size, source_seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, source_seq_len, self.n_head, self.d_head).transpose(1, 2)

        assert cache_pos is not None, "No input_pos"
        k, v = self.get_kv(k, v, cache_pos=cache_pos)
        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
        )

        # (bz, nh, L, dh) -> (bz, L, nh, dh) -> (bz, L, d)
        wv = wv.transpose(1, 2).reshape(
            batch_size, target_seq_len, self.n_head * self.d_head
        )

        return self.out(wv)


class DecoderAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = CausalSelfAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = CrossAttention(n_state, n_head)
        self.cross_attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp, bias=False),
            nn.GELU(),
            nn.Linear(n_mlp, n_state, bias=False),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        input_ids: Tensor,
        encoder_outputs: Tensor,
        use_encoder_cache: bool,
        mask: Optional[Tensor] = None,
        cache_pos: Optional[Tensor] = None,
        encoder_cache_pos: Optional[Tensor] = None,
    ):
        input_ids = input_ids + self.attn(
            self.attn_ln(input_ids),
            mask=mask,
            cache_pos=cache_pos,
        )

        input_ids = input_ids + self.cross_attn(
            self.cross_attn_ln(input_ids),
            encoder_outputs=encoder_outputs,
            encoder_cache_pos=encoder_cache_pos,
            use_cache=use_encoder_cache,
        )
        input_ids = input_ids + self.mlp(self.mlp_ln(input_ids))

        return input_ids


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.zeros(n_ctx, n_state))

        self.blocks: Iterable[DecoderAttentionBlock] = nn.ModuleList(  # type: ignore
            [DecoderAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_state)
        self.output = nn.Linear(n_state, n_vocab, bias=False)
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(
        self,
        input_ids: Tensor,
        encoder_outputs: Tensor,
        cache_pos: Tensor,
        encoder_cache_pos: Tensor,
        use_encoder_cache: bool = False,
    ):
        mask = self.causal_mask[None, None, cache_pos]
        input_ids = (
            self.token_embedding(input_ids) + self.positional_embedding[cache_pos]
        )

        for block in self.blocks:
            input_ids = block(
                input_ids=input_ids,
                encoder_outputs=encoder_outputs,
                mask=mask,
                cache_pos=cache_pos,
                encoder_cache_pos=encoder_cache_pos,
                use_encoder_cache=use_encoder_cache,
            )

        input_ids = self.ln(input_ids)
        logits = self.output(input_ids)

        return logits

    def setup_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        max_patches: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        ).to(device)
        # Init cache
        for b in self.blocks:
            b.attn.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_length=max_seq_len,
                n_heads=self.n_head,
                head_dim=self.n_state // self.n_head,
                dtype=dtype,
            ).to(device)
            b.cross_attn.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_length=max_patches,
                n_heads=self.n_head,
                head_dim=self.n_state // self.n_head,
                dtype=dtype,
            ).to(device)

    def reset_cache(self):
        for b in self.blocks:
            b.attn.kv_cache.zero()
            b.cross_attn.kv_cache.zero()
