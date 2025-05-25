from transformers import AutoModelForCausalLM
import torch
import deepspeed
from vllm import LLM
import wandb
from src.configs.config import *
from utils.utils import find_last_checkpoint, load_model_into_vllm

policy_model=AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="0",
    attn_implementation="flash_attention_2",
)

reference_model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG["model_name"],
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map=0,
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

wandb.init(
    project="r1-aha-moment",
    name=RUN_NAME,
    config={
        "model_name": MODEL_CONFIG["model_name"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "num_iterations": TRAINING_CONFIG["num_iterations"],
        "episodes_per_iteration": TRAINING_CONFIG["episodes_per_iteration"],
        "rollouts_per_episode": TRAINING_CONFIG["generations_per_sample"],
        "kl_coefficient": TRAINING_CONFIG["kl_coefficient"],
        "temperature": SAMPLING_CONFIG["temperature"],
    },
)

# Load checkpoint if it exists
begin_iter = 0
ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
if ckpt_path is not None:
    print(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
    out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
    if out is None:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
    begin_iter = ckpt_iter + 1
    load_model_into_vllm(policy_model, inference_engine)
