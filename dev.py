# %%
from img_to_text.domain import Config, DecoderArgs, EncoderArgs
from img_to_text.model import get_model
import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

config = Config(
    decoder_args=DecoderArgs(
        heads=16,
        layers=12,
        max_seq_len=4096,
        pad_token="<pad>",
        beg_of_seq_token="<bos>",
        end_of_seq_token="<eos>",
        pad_token_id=0,
        beg_of_seq_token_id=1,
        end_of_seq_token_id=2,
        vocab_size=8000,
    ),
    encoder_args=EncoderArgs(
        layers=(2, 2, 18, 2),
        heads=(4, 8, 16, 32),
        window_size=7,
        align_long_axis=True,
        output_patch_size=32,
        max_img_dimensions=(1152, 1536),
        min_img_dimensions=(32, 32),
        output_dimensions=1024,
        input_dimensions=128,
        patch_size=4,
        grad_checkpointing=False,
        input_channels=1,
        blocks=4,
    ),
    batch_size=1,
    mixed_precision=True,
)

model = get_model(config, "cuda")


# %%
import torch
from img_to_text.timer import Timer
import logging

logging.getLogger().setLevel(logging.DEBUG)

img = torch.randn(1, 1, 1536, 1152).to("cuda")

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        timer = Timer(name="Inference no compile")
        output = model.generate(img, timer=timer, max_length=30)

# %%
import os

os.makedirs("temp", exist_ok=True)

compiled_model = get_model(config, "cuda")
compiled_model.decoder.model.forward = torch.compile(compiled_model.decoder.model.forward, mode="max-autotune", fullgraph=True)  # type: ignore

img = torch.randn(1, 1, 1536, 1152).to("cuda")

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        timer = Timer(name="Inference with compile (warmup)")
        compiled_model.generate(img, timer=timer, max_length=10)

# with torch.profiler.profile() as prof:
#     with torch.no_grad():
#         with torch.autocast("cuda", dtype=torch.bfloat16):
#             # NOTE: 3 is equivalent to 1 token + 2 special tokens
#             output = compiled_model.generate(img, max_length=3)
#             prof.step()

# prof.export_chrome_trace("temp/torch_trace.json")

# %%
