from typing import Any
import torch.nn as nn
from ..domain import DecoderArgs
import torch.nn.functional as F
from ..timer import Timer
from .text_decoder import TextDecoder
import torch

__all__ = [
    "get_decoder",
    "Decoder",
]


class EndOfSequenceChecker:
    def __init__(self, end_of_seq_token_id: int, device: str | torch.device):
        self.end_of_seq_token_id = torch.tensor(
            [end_of_seq_token_id], dtype=torch.long, device=device
        )

    def test(self, other: torch.Tensor) -> bool:
        if other.dim() == 1:
            return torch.equal(self.end_of_seq_token_id, other[-1:])
        elif other.dim() == 2:
            return torch.equal(self.end_of_seq_token_id, other[:, -1])
        raise ValueError("Input tensor must be 1D or 2D")


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_args: DecoderArgs,
        max_patches: int,
        dim: int,
        dtype: torch.dtype,
        batch_size: int,
    ):
        super().__init__()
        self.decoder_args = decoder_args
        self.max_patches = max_patches
        self.dtype = dtype
        self.batch_size = batch_size
        self.model = TextDecoder(
            n_vocab=decoder_args.vocab_size,
            n_ctx=decoder_args.max_seq_len,
            n_state=dim,
            n_layer=decoder_args.layers,
            n_head=decoder_args.heads,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: we have to clone this because otherwise torch.compile gets confused
        # because while the input_ids is always a length of 1, the "stride" of the
        # parent tensor is whatever its index is. Therefore, torch.compile thinks
        # that the input_ids is changing when in fact it's not. This works around that
        # by creating a new tensor with the same value but a consistent stride of 0
        last_token_id = torch.tensor([[input_ids[0][-1]]], device=input_ids.device)

        return last_token_id, torch.tensor(
            [input_ids.shape[1] - 1], device=input_ids.device
        )

    def next_token(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_cache_pos: torch.Tensor,
        use_encoder_cache: bool,
        timer: Timer | None = None,
    ):
        last_input_ids, cache_position = self.prepare_inputs_for_generation(
            input_ids=input_ids,
        )

        if timer:
            timer.push_event("Prepared inputs for generation")
        outputs = self.model(
            input_ids=last_input_ids,
            encoder_outputs=encoder_outputs,
            cache_pos=cache_position,
            encoder_cache_pos=encoder_cache_pos,
            use_encoder_cache=use_encoder_cache,
        )
        if timer:
            timer.push_event("Generated token")
        logits = outputs
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((input_ids, xcol), dim=1)
        if timer:
            timer.push_event("Selected token")
        if timer:
            timer.stop()
            timer.report()
        return x

    def generate(
        self,
        encoder_outputs: torch.Tensor,
        max_length: int | None,
        timer: Timer | None = None,
    ) -> torch.Tensor:
        if (
            max_length
            and max_length > self.decoder_args.max_seq_len
            or max_length
            and max_length < 0
        ):
            raise ValueError(
                f"max_length must be in the range [0, {self.decoder_args.max_seq_len}]"
            )

        max_length = max_length or self.decoder_args.max_seq_len
        x = torch.full(
            (1, 1),
            self.decoder_args.beg_of_seq_token_id,
            dtype=torch.long,
            device=encoder_outputs.device,
        )
        eos_checker = EndOfSequenceChecker(
            end_of_seq_token_id=self.decoder_args.end_of_seq_token_id,
            device=encoder_outputs.device,
        )
        try:
            encoder_cache_pos = torch.arange(
                0, encoder_outputs.shape[1], device=encoder_outputs.device
            )
            self.model.setup_cache(
                1,
                self.decoder_args.max_seq_len,
                self.max_patches,
                device=encoder_outputs.device,
                dtype=self.dtype,
            )

            for i in range(max_length - 1):
                x = self.next_token(
                    x,
                    encoder_outputs,
                    timer=timer.gen_child("Next token") if timer else None,
                    use_encoder_cache=i > 0,
                    encoder_cache_pos=encoder_cache_pos,
                )
                if timer:
                    timer.push_event(f"Generated token {i}")
                if eos_checker.test(x):
                    break
            else:
                # if the loop completes without breaking, add the end token
                x = torch.cat(
                    (
                        x,
                        torch.tensor(
                            [[self.decoder_args.end_of_seq_token_id]], device="cuda"
                        ),
                    ),
                    dim=1,
                )
        finally:
            self.model.reset_cache()

        if timer:
            timer.push_event("Finished generating")
            timer.stop()

        return x

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.model(*args, **kwargs)


def get_decoder(
    decoder_args: DecoderArgs,
    max_patches: int,
    dim: int,
    batch_size: int,
    kv_cache_dtype: torch.dtype,
):
    return Decoder(
        decoder_args=decoder_args,
        dim=dim,
        max_patches=max_patches,
        dtype=kv_cache_dtype,
        batch_size=batch_size,
    )
