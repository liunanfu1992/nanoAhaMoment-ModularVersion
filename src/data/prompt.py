from typing import Any, Dict, List

class PromptTemplates:

    SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in the mind "
    "and then provide the user with the answer."
    ) 

    PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
    )   

    @classmethod
    def preprocess_example(cls, example: Dict[str, Any]):
        numbers: List[int] = example["numbers"]
        target:int = example["target"]

        prefix = [
            {"role": "system", "content":cls.SYSTEM_MESSAGE},
            {"role": "user", "content": cls.PROMPT_TEMPLATE.format(numbers=numbers, target=target)},
            {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
        ]

        


