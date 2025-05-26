from typing import Dict,Any
from transformers import AutoTokenizer
from datasets import load_dataset
from .prompt import SYSTEM_MESSAGE,PROMPT_TEMPLATE
from src.config.configs import DATASET_CONFIG
from src.model.model_tokenizer import ModelTokenizer


def preprocess_example(example:Dict[str,Any],tokenizer:AutoTokenizer):
    numbers=example["nums"]
    target=example["target"]

    prefix=[
        {"role": "system", "content":SYSTEM_MESSAGE},
        {"role": "user", "content":PROMPT_TEMPLATE.format(numbers=numbers,target=target)},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]

    input_ids = tokenizer.apply_chat_template(prefix,tokenize=True,continue_final_message=True)

    prompt=tokenizer.decode(input_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)

    return {"prompt":prompt,"input_ids":input_ids}

def load_and_preprocess_dataset(tokenizer:ModelTokenizer):
    
    dataset=load_dataset(DATASET_CONFIG["dataset_name"],split="train")
    dataset=dataset.map(
        lambda x: preprocess_example(x,tokenizer.getModelChatTokenizer()), # 这个tokenizer是model_chat_tokenizer
        num_proc=6
    )

    train_test_split=dataset.train_test_split(test_size=DATASET_CONFIG["test_size"],seed=DATASET_CONFIG["seed"])

    return {
        "train":train_test_split["train"],
        "test":train_test_split["test"],
    }
