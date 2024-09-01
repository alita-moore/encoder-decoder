# %%
from img_to_text.domain import Config, DecoderArgs, EncoderArgs
from img_to_text.architecture import get_model
config = Config(
    decoder_args=DecoderArgs(
        heads=16,
        layers=10,
        max_seq_len=1024,
        pad_token="<pad>",
        beg_of_seq_token="<bos>",
        end_of_seq_token="<eos>",
        pad_token_id=0,
        beg_of_seq_token_id=1,
        end_of_seq_token_id=2,
        vocab_size=50257,
    ),
    encoder_args=EncoderArgs(
        layers=(2, 2, 18, 2),
        heads=(4, 8, 16, 32),
        window_size=7,
        align_long_axis=True,
        output_patch_size=2,
        max_img_dimensions=(224, 224),
        min_img_dimensions=(56, 56),
        output_dimensions=1024,
        input_dimensions=128,
        patch_size=4,
        grad_checkpointing=False,
        input_channels=1,
        blocks=3,
    ),
    batch_size=1,
    mixed_precision=True,
)

model = get_model(config, "cuda", False)

# %%
import torch
from img_to_text.timer import Timer
import logging
logging.getLogger().setLevel(logging.DEBUG)

img = torch.randn(1, 1, 224, 224).to("cuda")

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        timer = Timer(name="Inference no compile")
        output = model.generate(img, timer=timer, max_length=30)

# %%
import os

os.makedirs("temp", exist_ok=True)

compiled_model = get_model(config, "cuda", False)
compiled_model.decoder.model = torch.compile(compiled_model.decoder.model, mode="max-autotune") # type: ignore

img = torch.randn(1, 1, 224, 224).to("cuda")

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        timer = Timer(name="Inference with compile (warmup)")
        compiled_model.generate(img, timer=timer, max_length=10)

with torch.profiler.profile() as prof:
    with torch.no_grad(): 
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                # NOTE: 3 is equivalent to 1 token + 2 special tokens
                output = compiled_model.generate(img, max_length=3)
                prof.step()

prof.export_chrome_trace("temp/torch_trace.json")

# %%
