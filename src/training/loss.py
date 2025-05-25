import torch
from typing import Dict, Tuple, Union
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel
from src.utils.utils import compute_token_log_probs
from src.config.configs import *


def compute_pg_loss(policy_model:Union[DeepSpeedEngine,PreTrainedModel],
                    reference_model:Union[DeepSpeedEngine,PreTrainedModel],
                    batch:Dict[str,torch.Tensor],
                    total_response_len:int)->Tuple[torch.Tensor,Dict[str,float]]:
    
    input_ids=batch["input_ids"]
    attention_mask=batch["attention_mask"]
    labels=batch["labels"]
    advantages=batch["advantages"]

    model_inputs={
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "labels":labels,
    }

    labels_mask=(labels[...,1:] != -100).float()

    with torch.no_grad():
        ref_logps=compute_token_log_probs(reference_model,model_inputs,SAMPLING_CONFIG["temperature"])
    
    logps=compute_token_log_probs(policy_model,model_inputs,SAMPLING_CONFIG["temperature"])

    kl_penalty=torch.exp(ref_logps-logps)-(ref_logps-logps)-1
    kl_penalty=kl_penalty*labels_mask

    entropy=-logps.sum()/labels_mask.sum()

    policy_loss=-logps*advantages[...,1:]
    policy_loss=policy_loss*labels_mask

    loss=(policy_loss+kl_penalty*TRAINING_CONFIG["kl_coefficient"]).sum()/total_response_len

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item() / total_response_len,
    }

    return loss,metrics