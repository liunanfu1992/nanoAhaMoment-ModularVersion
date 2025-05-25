from deepspeed import DeepSpeedEngine
from typing import Dict, List, Union
from transformers import PreTrainedModel
import torch

def prepare_model_inputs(
        query_token_ids:List[List[int]],
        response_token_ids:List[List[int]],
        advantages:List[float],
        device:torch.device,
)->Dict[str,torch.Tensor]:

    max_seq_len=max(len(q)+len(r) for q,r in zip(query_token_ids,response_token_ids))
    inputs={"input_ids":[],"attention_mask":[],"labels":[],"advantages":[]}

    pad_token_id=0
    ignore_index=-100
    for query,response,advantage in zip(query_token_ids,response_token_ids,advantages):
        combined_ids=query+response
        seq_len=len(combined_ids)

        input_ids=combined_ids+[pad_token_id]*(max_seq_len-seq_len)
        attention_mask=[1]*seq_len+[0]*(max_seq_len-seq_len)
        labels=[ignore_index]*len(query)+response+[ignore_index]*(max_seq_len-seq_len)
        advantages_seq=[ignore_index]*len(query)+advantage+[ignore_index]*(max_seq_len-seq_len)

        assert len(input_ids)==max_seq_len
        assert len(attention_mask)==max_seq_len
        assert len(labels)==max_seq_len
        assert len(advantages_seq)==max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)

    return {
        k:torch.tensor(v,dtype=torch.long if k!="advantages" else torch.float,device=device)
        for k,v in inputs.items()
    }

def compute_token_log_probs(
        model:Union[DeepSpeedEngine,PreTrainedModel],
        inputs:Dict[str,torch.Tensor],
        temperature:float,
)->torch.Tensor:
    
    outputs=model(
        input_ids=input["input_ids"],
        attention_mask=input["attention_mask"],
        return_dict=True,
        use_cache=False
    )

    logits=outputs.logits.float()/temperature
    shift_logits=logits[...,:-1,:].contiguous()
    shift_labels=inputs["labels"][...,1:].contiguous()

    label_mask=(shift_labels!=-100).float()
    shift_labels[shift_labels==-100]=0

    log_probs=torch.log_softmax(shift_logits,dim=-1)
    log_prob=torch.gather(log_probs,dim=-1,index=shift_labels.unsqueeze(2))
    log_prob=log_prob.squeeze(2)
    log_prob=log_prob*label_mask

    return log_prob