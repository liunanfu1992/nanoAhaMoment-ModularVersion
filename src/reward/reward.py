import re
from typing import Any, Dict, List
from src.model.model_tokenizer import ModelTokenizer

def format_reward_func(completion:str,tokenizer:ModelTokenizer)->float:
    try:
        completion="<think>"+completion

        _,EOS_TOKEN=tokenizer.get_eos_token_and_id()
        
        if completion.endswith(EOS_TOKEN):
            completion=completion[:-len(EOS_TOKEN)]

        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            return 0.0
        else:
            answer_content=match.group(2).strip()
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern,answer_content):
                return 0.5
            else:
                return 1.0
    except Exception:
        return 0.0
    
def equation_reward_func(completion:str,nums:List[int],target:int)->float:
    try:
        
        match=re.search(r"<answer>(.*?)<\/answer>",completion)
        if match is None:
            return 0.0
        
        equation=match.group(1).strip()

        used_numbers=[int(n) for n in re.findall(r'\d+',equation)]

        if sorted(used_numbers)!=sorted(nums):
            return 0.0
        allowed_pattern=r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern,equation):
            return 0.0
        
        result=eval(equation,{"__builtins__": None}, {})

        if abs(result-float(target))<1e-5:
            return 1.0
        else:
            return 0.0
        
    except Exception:
        return 0.0
        
def compute_reward(completion:str,sample:Dict[str,Any],tokenizer:ModelTokenizer)->float:
    nums=sample["nums"]
    target=sample["target"]

    equation_reward=equation_reward_func(completion,nums,target)
    format_reward=format_reward_func(completion,tokenizer)

    reward=equation_reward+format_reward

    metrics={
        "format_reward":format_reward,
        "equation_reward":equation_reward,
    }
    
    return reward,metrics

    