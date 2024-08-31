from pydantic import BaseModel, ValidationInfo, field_validator


def greater_than_or_equal_to_zero_validator(v: int, info: ValidationInfo):
    if v < 0:
        raise ValueError("Value must be greater than or equal to 0")
    return v


class DecoderArgs(BaseModel):
    heads: int
    layers: int
    max_seq_len: int
    pad_token: str
    beg_of_seq_token: str
    end_of_seq_token: str
    pad_token_id: int
    beg_of_seq_token_id: int
    end_of_seq_token_id: int
    vocab_size: int

    @field_validator(
        "layers",
        "heads",
        "max_seq_len",
        "vocab_size",
        "pad_token_id",
        "beg_of_seq_token_id",
        "end_of_seq_token_id",
    )
    @classmethod
    def greater_than_or_equal_to_zero_validator(cls, v: int, info: ValidationInfo):
        return greater_than_or_equal_to_zero_validator(v, info)

class EncoderArgs(BaseModel):
    layers: tuple[int, ...]
    heads: tuple[int, ...]
    window_size: int
    align_long_axis: bool
    output_patch_size: int
    max_img_dimensions: tuple[int, int]
    min_img_dimensions: tuple[int, int]
    heads: tuple[int, ...]
    output_dimensions: int
    input_dimensions: int
    patch_size: int
    grad_checkpointing: bool
    input_channels: int
    blocks: int

    @property
    def max_output_patches(self):
        return (self.max_img_dimensions[0] * self.max_img_dimensions[1]) // self.patch_size

class Config(BaseModel):
    decoder_args: DecoderArgs
    encoder_args: EncoderArgs
    batch_size: int
    mixed_precision: bool