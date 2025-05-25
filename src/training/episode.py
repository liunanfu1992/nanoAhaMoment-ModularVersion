
from typing import Any, Dict, List, Tuple
import numpy as np
from transformers import AutoTokenizer

from src.data.reward import compute_reward



def create_training_episodes(sample:List[Dict[str,Any]],all_generations:List[List[int]],all_finish_reasons:List[str],tokenizer:AutoTokenizer,GENERATIONS_PER_SAMPLE:int=4)->Tuple[Dict[str,Any],Dict[str,Any]]:
    
    assert len(all_generations)==len(all_finish_reasons)
    assert len(all_generations)==len(sample)*GENERATIONS_PER_SAMPLE

    groups=[
        list(range(i,i+GENERATIONS_PER_SAMPLE))
        for i in range(0,len(all_generations),GENERATIONS_PER_SAMPLE)
    ]

    all_query_token_ids,all_responses_token_ids,all_advantages=[],[],[]

    stats={
        "response_lengths":[],
        "rewards":[],
        "non_stop_rate":[],
    }

    for sample,group_indices in zip(sample,groups):
        finish_reasons=[all_finish_reasons[i] for i in group_indices]
        response_token_ids=[all_generations[i] for i in group_indices]
        responses=tokenizer.decode(response_token_ids,skip_special_tokens=False)
        
        rewards_and_metrics=[compute_reward(response,sample) for response in responses]
        rewards,reward_metrics=zip(*rewards_and_metrics)

        rewards=np.array(rewards)
        response_advantages=(rewards-rewards.mean())/(rewards.std()+1e-4)

        advantages=[
            [resp_adv]*len(resp)
            for resp_adv,resp in zip(response_advantages,response_token_ids)
        ]
        
        all_query_token_ids.extend([sample["input_ids"]]*GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes,stats