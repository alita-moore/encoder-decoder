import torch
from torch import nn
from timm.models.swin_transformer_v2 import SwinTransformerV2


# NOTE: we compile just this method because otherwise it takes too long to compile.
# this gives an approximate 2x speedup on the forward pass.
# WindowAttention.forward = torch.compile(WindowAttention.forward, mode="reduce-overhead")

__all__ = ["Encoder"]

class Encoder(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: tuple[int, ...],
        patch_size: int,
        embed_dim: int,
        num_heads: tuple[int, ...],
        in_chans: int,
        grad_checkpointing: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.model = SwinTransformerV2(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_classes=0,
            in_chans=in_chans,
            strict_img_size=False,
        )
        self.model.set_grad_checkpointing(grad_checkpointing)

    def _pad_img(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        max_h, max_w = self.input_size
        assert h <= max_h and w <= max_w, f"Input image is too large, got ({w}, {h}) expected less than ({max_w}, {max_h})"
        pw = max_w - w
        ph = max_h - h
        if pw > 0 or ph > 0:
            x = nn.functional.pad(x, (0, pw, 0, ph))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        x = self.model.patch_embed(self._pad_img(x))
        x = self.model.layers(x)
        return x
