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

model.decoder.model.setup_cache(
    1,
    config.decoder_args.max_seq_len,
    config.encoder_args.max_output_patches,
    device="cuda",
)

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        timer = Timer(name="Inference no compile")
        output = model.generate(img, timer=timer, max_length=5)

# %%
import os

os.makedirs("temp", exist_ok=True)

compiled_model = get_model(config, "cuda")
compiled_model.to(torch.bfloat16)
compiled_model.decoder.model.forward = torch.compile(compiled_model.decoder.model.forward, mode="max-autotune", fullgraph=True)  # type: ignore
compiled_model.decoder.model.setup_cache(
    1,
    config.decoder_args.max_seq_len,
    config.encoder_args.max_output_patches,
    device="cuda",
)


# %%
from torch.nn.attention import SDPBackend

img = torch.randn(1, 1, 1536, 1152).to("cuda")
with torch.inference_mode():
    # with torch.autocast("cuda", dtype=torch.bfloat16):
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
        timer = Timer(name="Inference with compile (warmup)")
        compiled_model.generate(
            img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10
        )
        compiled_model.generate(
            img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10
        )
        compiled_model.generate(
            img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10
        )
        compiled_model.generate(
            img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10
        )
        print(output.shape)

# %%
# # %%
# import torch
# import torch.nn as nn

# inputs = [
#     (
#         "INPUT__input_ids",
#         torch.full(
#             (1, 1),
#             1,
#             dtype=torch.long,
#             device="cuda",
#         ),
#     ),
#     (
#         "INPUT__encoder_outputs",
#         torch.randn((1, 1728, 1024), device="cuda").to(torch.float16),
#     ),
#     (
#         "INPUT__encoder_cache_pos",
#         torch.arange(0, 1728, device="cuda"),
#     ),
#     (
#         "INPUT__cache_position",
#         torch.tensor([0], device="cuda"),
#     ),
# ]

# # %%
# from img_to_text.model import get_model


# class Wrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         encoder_outputs: torch.Tensor,
#         encoder_cache_pos: torch.Tensor,
#         cache_position: torch.Tensor,
#     ):
#         return self.model(
#             input_ids=input_ids,
#             encoder_outputs=encoder_outputs,
#             encoder_cache_pos=encoder_cache_pos,
#             cache_pos=cache_position,
#         )


# model = get_model(config, "cuda")
# wrapper_ = Wrapper(model.decoder.model)
# wrapper_.model.setup_cache(
#     batch_size=1,
#     max_seq_len=4096,
#     max_patches=1728,
#     device="cuda",
#     dtype=torch.float16,
# )
# # %%
# with torch.no_grad():
#     with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
#         with torch.autocast("cuda", dtype=torch.float16):
#             torch.onnx.export(
#                 wrapper_,
#                 tuple(map(lambda x: x[1], inputs)),
#                 "temp/decoder.onnx",
#                 input_names=list(map(lambda x: x[0], inputs)),
#                 verbose=False,
#             )

# # %%
# from onnxruntime import (
#     GraphOptimizationLevel,
#     InferenceSession,
#     SessionOptions,
#     get_all_providers,
# )
# from img_to_text.timer import Timer
# import logging

# logging.getLogger().setLevel(logging.DEBUG)


# def execute_onnx_model(input_data: dict, use_gpu: bool = False) -> list:
#     options = SessionOptions()
#     # options.graph_optimization_level = (
#     #     GraphOptimizationLevel.ORT_ENABLE_ALL
#     # )  # Disable all optimizations
#     # options.intra_op_num_threads = 8
#     # options.inter_op_num_threads = 8
#     # options.log_severity_level = 2
#     # options.optimized_model_filepath = "temp/optimized2.onnx"
#     # providers = ["CPUExecutionProvider"]  # Default to CPU provider
#     # if use_gpu and "CUDAExecutionProvider" in get_all_providers():
#     #     providers = [
#     #         "CUDAExecutionProvider"
#     #     ]  # Switch to CUDA if available and requested

#     try:
#         sess = InferenceSession(
#             "temp/decoder.onnx",
#             options,
#             providers=[
#                 (
#                     "TensorrtExecutionProvider",
#                     {
#                         "device_id": 0,  # Select GPU to execute
#                         "trt_max_workspace_size": 2147483648,  # Set GPU memory usage limit
#                         "trt_fp16_enable": True,  # Enable FP16 precision for faster inference
#                     },
#                 ),
#                 (
#                     "CUDAExecutionProvider"
#                     # {
#                     #     "device_id": 0,
#                     #     "arena_extend_strategy": "kNextPowerOfTwo",
#                     #     "gpu_mem_limit": 8 * 1024 * 1024 * 1024,
#                     #     "cudnn_conv_algo_search": "EXHAUSTIVE",
#                     #     "do_copy_in_default_stream": True,
#                     # },
#                 ),
#             ],
#         )
#     except Exception as e:
#         raise RuntimeError(f"Failed to load the model: {e}")

#     try:
#         timer = Timer(
#             enabled=True,
#             name="ONNX Inference",
#             synchronize=lambda: torch.cuda.synchronize(),
#         )
#         for _ in range(10):
#             output = sess.run(None, input_data)
#             timer.push_event("Forward")
#         timer.stop()
#         return output
#     except Exception as e:
#         raise RuntimeError(f"Failed during model execution: {e}")


# inputs__ = {
#     "INPUT__input_ids": inputs[0][1].cpu().numpy(),
#     "INPUT__encoder_outputs": inputs[1][1].cpu().to(torch.float16).numpy(),
#     "INPUT__encoder_cache_pos": inputs[2][1].cpu().numpy(),
#     "INPUT__cache_position": inputs[3][1].cpu().numpy(),
# }


# execute_onnx_model(
#     inputs__,
#     True,
# )

# # %%
# import tensorrt as trt
# import numpy as np
# import pycuda.driver as cuda
# import pycuda.autoinit
# import logging
# from contextlib import contextmanager
# from typing import Dict, List, Tuple
# import os

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


# class TensorRTModel:
#     def __init__(
#         self,
#         onnx_file_path: str,
#         trt_file_path: str,
#         input_shapes: Dict[str, Tuple[int, ...]],
#     ):
#         self.onnx_file_path = onnx_file_path
#         self.trt_file_path = trt_file_path
#         self.input_shapes = input_shapes
#         self.engine = None
#         self.context = None
#         self.stream = None
#         self.input_buffers = {}
#         self.output_buffers = {}

#     def build_engine(self):
#         logger.info("Building TensorRT engine...")
#         TRT_LOGGER = trt.Logger(trt.Logger.INFO)

#         builder = trt.Builder(TRT_LOGGER)
#         config = builder.create_builder_config()
#         config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

#         # Enable FP16 mode if desired
#         if builder.platform_has_fast_fp16:
#             config.set_flag(trt.BuilderFlag.FP16)

#         # Create network
#         EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#         network = builder.create_network(EXPLICIT_BATCH)

#         # Create ONNX parser
#         parser = trt.OnnxParser(network, TRT_LOGGER)

#         # Parse ONNX file
#         with open(self.onnx_file_path, "rb") as model:
#             if not parser.parse(model.read()):
#                 for error in range(parser.num_errors):
#                     logger.error(f"ONNX parsing error: {parser.get_error(error)}")
#                 raise RuntimeError("Failed to parse the ONNX file.")

#         # Set input shapes
#         profile = builder.create_optimization_profile()
#         for input_name, shape in self.input_shapes.items():
#             profile.set_shape(input_name, shape, shape, shape)
#         config.add_optimization_profile(profile)

#         # Build engine
#         plan = builder.build_serialized_network(network, config)
#         if plan is None:
#             raise RuntimeError("Failed to build TensorRT engine.")

#         # Create engine from plan
#         self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(plan)

#         # Save engine to file
#         with open(self.trt_file_path, "wb") as f:
#             f.write(plan)
#         logger.info(f"TensorRT engine saved to {self.trt_file_path}")

#     def load_engine(self):
#         if not os.path.exists(self.trt_file_path):
#             logger.info("TensorRT engine not found. Building a new one...")
#             self.build_engine()

#         logger.info("Loading TensorRT engine...")
#         with open(self.trt_file_path, "rb") as f, trt.Runtime(
#             trt.Logger(trt.Logger.INFO)
#         ) as runtime:
#             self.engine = runtime.deserialize_cuda_engine(f.read())

#         self.context = self.engine.create_execution_context()
#         self.stream = cuda.Stream()

#         # Allocate buffers
#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             dtype = trt.nptype(self.engine.get_tensor_dtype(name))
#             shape = self.engine.get_tensor_shape(name)
#             size = trt.volume(shape) * np.dtype(dtype).itemsize
#             buf = cuda.mem_alloc(size)
#             if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
#                 self.input_buffers[name] = buf
#             else:
#                 self.output_buffers[name] = buf

#             logger.info(
#                 f"Tensor {i}: {name} ({'input' if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else 'output'}) - Shape: {shape}, Dtype: {dtype}"
#             )

#     @contextmanager
#     def use_engine(self):
#         if self.engine is None:
#             self.load_engine()
#         yield

#     def infer(self, input_data: Dict[str, np.ndarray]) -> List[np.ndarray]:
#         with self.use_engine():
#             # Set input shapes and transfer input data to device
#             for name, data in input_data.items():
#                 if name not in self.input_buffers:
#                     raise ValueError(
#                         f"Input tensor '{name}' not found in engine bindings"
#                     )
#                 self.context.set_input_shape(name, data.shape)
#                 cuda.memcpy_htod_async(self.input_buffers[name], data, self.stream)
#                 self.context.set_tensor_address(name, int(self.input_buffers[name]))

#             # Run inference
#             if not self.context.all_binding_shapes_specified:
#                 raise RuntimeError("Not all input shapes have been specified.")

#             self.context.execute_async_v3(stream_handle=self.stream.handle)

#             # Transfer predictions back from device
#             output_data = []
#             for name, buf in self.output_buffers.items():
#                 shape = self.context.get_tensor_shape(name)
#                 dtype = trt.nptype(self.engine.get_tensor_dtype(name))
#                 output = np.empty(shape, dtype=dtype)
#                 cuda.memcpy_dtoh_async(output, buf, self.stream)
#                 output_data.append(output)

#             # Synchronize the stream
#             self.stream.synchronize()

#             return output_data


# def main():
#     onnx_file_path = "temp/decoder.onnx"
#     trt_file_path = "temp/model.plan"

#     input_shapes = {
#         "INPUT__input_ids": (1, 1),
#         "INPUT__encoder_outputs": (1, 1728, 1024),
#         "INPUT__encoder_cache_pos": (1728,),
#         "INPUT__cache_position": (1,),
#     }

#     model = TensorRTModel(onnx_file_path, trt_file_path, input_shapes)

#     # Prepare input data using the actual input names
#     input_data = {
#         "INPUT__input_ids": np.array([[0]], dtype=np.int64),
#         "INPUT__encoder_outputs": np.random.randn(1, 1728, 1024).astype(np.float16),
#         "INPUT__encoder_cache_pos": np.array([0] * 1728, dtype=np.int64),
#         "INPUT__cache_position": np.array([0], dtype=np.int64),
#     }

#     # Warm-up run
#     _ = model.infer(input_data)

#     # Measure inference time
#     import time

#     num_inferences = 1000
#     start_time = time.time()
#     for _ in range(num_inferences):
#         output = model.infer(input_data)
#     end_time = time.time()

#     avg_inference_time = (end_time - start_time) / num_inferences
#     logger.info(
#         f"Average inference time over {num_inferences} runs: {avg_inference_time:.4f} seconds"
#     )
#     logger.info(f"Inference output shapes: {[o.shape for o in output]}")


# main()

# # %%
