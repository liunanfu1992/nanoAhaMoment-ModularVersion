from transformers import AutoModelForCausalLM
import torch
import deepspeed
from vllm import LLM
from src.config.configs import *


def load_model():
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    policy_model=AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )

    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name"],
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=DEEPSPEED_CONFIG,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=REF_DEEPSPEED_CONFIG,
    )

    reference_model.module.cpu()

    inference_engine = LLM(
        model=MODEL_CONFIG["model_name"],
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=True,
        swap_space=1,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=2048,
        enable_sleep_mode=True,
    )

    return policy_model,reference_model,inference_engine


