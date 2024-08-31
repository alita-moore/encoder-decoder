from typing import Any
import torch
import torch.nn as nn
from .timer import Timer

from .domain import Config
from .decoder import get_decoder, Decoder
from .encoder import Encoder

__all__ = ["get_model", "Model"]


class Model(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        max_seq_len: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def _adapt_encoder_output(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.dim() == 4:
            b, h, w, c = encoded.shape
            encoded = encoded.reshape(b, h * w, c)

        return encoded

    def forward(
        self,
        x: torch.Tensor,
        tgt_seq: torch.Tensor,
        attention_mask: torch.Tensor,
        timer: Timer,
    ) -> Any:
        encoded = self.encoder(x)
        timer.push_event("Encoder")
        encoded = self._adapt_encoder_output(encoded)
        timer.push_event("Adapted encoder output")

        out = self.decoder(
            input_ids=tgt_seq[:, :-1],
            encoder_outputs=encoded,
            attention_mask=attention_mask[:, :-1],
            labels=tgt_seq[:, 1:],
        )
        timer.push_event("Decoder")
        timer.stop()
        return out

    @torch.no_grad()  # type: ignore
    def generate(
        self,
        x: torch.Tensor,
        timer: Timer | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(x)
        if timer:
            timer.push_event("Encoder")
        encoded = self._adapt_encoder_output(encoded)
        if timer:
            timer.push_event("Adapted encoder output")

        outputs = self.decoder.generate(
            encoder_outputs=encoded,
            max_length=max_length,
            timer=timer.gen_child("Decoder") if timer else None,
        )

        if timer:
            timer.push_event("Decoder")
            timer.stop()

        return outputs


def get_model(
    config: Config,
    device: str,
) -> Model:
    encoder = Encoder(
        input_size=(
            config.encoder_args.max_img_dimensions[1],
            config.encoder_args.max_img_dimensions[0],
        ),
        align_long_axis=config.encoder_args.align_long_axis,
        window_size=config.encoder_args.window_size,
        encoder_layer=config.encoder_args.layers,
        patch_size=config.encoder_args.patch_size,
        embed_dim=config.encoder_args.input_dimensions,
        num_heads=config.encoder_args.heads,
        in_chans=config.encoder_args.input_channels,
        grad_checkpointing=config.encoder_args.grad_checkpointing,
    )
    decoder = get_decoder(
        decoder_args=config.decoder_args,
        max_patches=config.encoder_args.max_output_patches,
        dim=config.encoder_args.output_dimensions,
        batch_size=config.batch_size,
        kv_cache_dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
    )
    encoder.to(device)
    decoder.to(device)
    model = Model(
        encoder,
        decoder,
        max_seq_len=config.decoder_args.max_seq_len,
        bos_token_id=config.decoder_args.beg_of_seq_token_id,
        eos_token_id=config.decoder_args.end_of_seq_token_id,
        pad_token_id=config.decoder_args.pad_token_id,
    ) 

    return model.to(device)
