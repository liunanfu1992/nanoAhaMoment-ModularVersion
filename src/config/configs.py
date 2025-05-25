from pathlib import Path

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B",
    "model_chat_name": "Qwen/Qwen2.5-3B-Instruct",
    "attn_implementation": "flash_attention_2",
    "torch_dtype": "bfloat16",
}

# 数据集配置
DATASET_CONFIG = {
    "dataset_name": "Jiayi-Pan/Countdown-Tasks-3to4",
    "test_size": 500,
    "seed": 42,
}

# 训练配置
TRAINING_CONFIG = {
    "num_iterations": 1000,
    "episodes_per_iteration": 64,
    "generations_per_sample": 4,
    "kl_coefficient": 0.001,
    "per_device_batch_size": 4,
    "learning_rate": 1e-6,
    "eval_interval": 25,
    "checkpoint_interval": 50,
}

# 采样配置
SAMPLING_CONFIG = {
    "max_response_tokens": 1024,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,  # 禁用 top-k 采样
}

# DeepSpeed配置
DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": False
    },
    "train_batch_size": TRAINING_CONFIG["episodes_per_iteration"],
    "train_micro_batch_size_per_gpu": TRAINING_CONFIG["per_device_batch_size"],
    "gradient_accumulation_steps": (
        TRAINING_CONFIG["episodes_per_iteration"] // 
        TRAINING_CONFIG["per_device_batch_size"]
    ),
    "gradient_clipping": 1.0,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": TRAINING_CONFIG["learning_rate"],
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "torch_adam": True,
        },
    },
}

REF_DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    # Note that we don't train the reference model
    # These are just for compatibility with DeepSpeed.
    "train_batch_size": TRAINING_CONFIG["episodes_per_iteration"],
    "train_micro_batch_size_per_gpu": TRAINING_CONFIG["per_device_batch_size"],
    "gradient_accumulation_steps": TRAINING_CONFIG["episodes_per_iteration"] // TRAINING_CONFIG["per_device_batch_size"],
}

RUN_NAME = "r1-zero"
EXP_DIR = Path.home() / "scratch" / "deepseek_r1z_hackathon" / RUN_NAME

def init_experiment_dir():
    try:
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")
    except Exception as e:
        print(f"Error creating experiment directory: {e}")
        raise
