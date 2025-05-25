from typing import Dict,Any
from transformers import AutoTokenizer
from datasets import load_dataset
from .prompt import SYSTEM_MESSAGE,PROMPT_TEMPLATE

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

def load_and_preprocess_dataset(dataset_name:str,model_name:str,test_size:int=500,seed:int=42):
    
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    dataset=load_dataset(dataset_name,split="train")
    dataset=dataset.map(
        lambda x: preprocess_example(x,tokenizer),
        num_proc=6
    )
    train_test_split=dataset.train_test_split(test_size=test_size,seed=seed)

    return {
        "train":train_test_split["train"],
        "test":train_test_split["test"],
        "tokenizer":tokenizer
    }
