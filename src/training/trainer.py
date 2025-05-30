import gc
import time
import numpy as np
from tqdm import trange
import wandb
from src.reward.reward import compute_reward
from src.config.configs import *
from src.utils.utils import *
from src.model.model import load_model
from src.model.model_tokenizer import ModelTokenizer
from src.data.dataset import load_and_preprocess_dataset
from src.training.episode import create_training_episodes
from src.training.loss import compute_pg_loss
import os


if __name__ == "__main__":
    policy_model,reference_model,inference_engine=load_model()
    tokenizer=ModelTokenizer()
    init_experiment_dir()
    EOS_TOKEN_ID,EOS_TOKEN=tokenizer.get_eos_token_and_id()
    dataset=load_and_preprocess_dataset(tokenizer)
    train_dataset=dataset["train"]
    test_dataset=dataset["test"]

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

    for iteration in trange(TRAINING_CONFIG["num_iterations"]):
        print(f"Iteration {iteration}/{TRAINING_CONFIG['num_iterations']}")

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if iteration % 25 == 0:
            print("Evaluating on eval set...")

            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer.getModelChatTokenizer(),
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=1024,
                    n=1,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
                reward_func=lambda completion, sample: compute_reward(
                    completion, sample, tokenizer
                ),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer.getModelChatTokenizer(),
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})


        #########################################################
        # Generate Episodes
        #########################################################

        # Sample training batch
        num_samples = TRAINING_CONFIG["episodes_per_iteration"] // TRAINING_CONFIG["generations_per_sample"]
        indices = np.random.choice(
            len(train_dataset), size=num_samples, replace=False
        )
        samples = train_dataset.select(indices)

        # Sample responses
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=TRAINING_CONFIG["generations_per_sample"],
                temperature=SAMPLING_CONFIG["temperature"],
                top_p=SAMPLING_CONFIG["top_p"],
                top_k=SAMPLING_CONFIG["top_k"],
                max_tokens=SAMPLING_CONFIG["max_response_tokens"],
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            )
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        inference_engine.sleep(1)

        print(f"Generated {len(all_generations)} responses")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        # Process responses and calculate rewards
        episodes, episodes_stats = create_training_episodes(
            samples,
            all_generations,
            all_finish_reasons,
            tokenizer,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer.getModelChatTokenizer(),
            iteration=iteration,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device="cuda"
        )

        # Calculate losses and update model
        policy_model.train()
        reference_model.module.cuda()
        reference_model.eval()

        total_response_len = (model_inputs["labels"] != -100).sum().item()

        for i in trange(0, TRAINING_CONFIG["episodes_per_iteration"], TRAINING_CONFIG["per_device_batch_size"], desc="Gradient Accumulation"):
            batch = {
                k: v[i : i + TRAINING_CONFIG["per_device_batch_size"]]
                for k, v in model_inputs.items()
            }

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                total_response_len=total_response_len,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation and optimization step
            policy_model.backward(loss, scale_wrt_gas=False)
            
            # Free memory
            del loss, loss_metrics
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()

            policy_model.step()

        #########################################################
        # Update inference engine weights
        #########################################################
        
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)


        #########################################################
        # Log metrics
        #########################################################

        train_metrics = {
            k: np.mean(v) for k, v in metrics.items() if None not in v
        }
        train_metrics["learning_rate"] = policy_model.get_lr()[0]
        logs = {
            "iteration": iteration,
            f"episodes/iter_{iteration:06d}": episode_table,
            **{f"train/{k}": v for k, v in train_metrics.items()},
        }
        if eval_stats is not None:
            eval_metrics = {k: np.mean(v) for k, v in eval_stats.items() if None not in v}
            logs.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        wandb.log(logs)

        selected_keys = [
            "train/kl_penalty",
            "train/rewards",
            "train/reward_metrics/format_reward",
            "train/reward_metrics/equation_reward",
            "eval/rewards",
            "eval/reward_metrics/format_reward",
            "eval/reward_metrics/equation_reward",
        ]
        selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
        print(f"KEY METRICS: {selected_metrics}")

        if iteration % 50 == 0 and iteration != 0:
            policy_model.module.save_pretrained(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model")
            )
            policy_model.save_checkpoint(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "deepspeed")
            )
    


   