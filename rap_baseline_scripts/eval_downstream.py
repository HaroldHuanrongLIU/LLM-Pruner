"""Evaluate LLM-Pruner pruned models on downstream tasks (HumanEval, LongBench, Summarization).

Must be run with LLM-Pruner's own venv to avoid transformers version mismatch.

Usage (from repo root):
    cd baselines/LLM-Pruner
    .venv/bin/python rap_baseline_scripts/eval_downstream.py \
        --ckpt prune_log/llama3_8b_ratio_0.35/pytorch_model.bin \
        --tasks humaneval longbench summarization \
        --output outputs/reviewer_experiments/code_generation/llama3_8b_llmpruner_b0.8.json

Note: Results are saved in the same format as our eval scripts for easy comparison.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import torch

# Set up path for lm-eval
try:
    import lm_eval
except ImportError:
    # Try installing in LLM-Pruner venv
    os.system(f"{sys.executable} -m pip install lm-eval --quiet")
    import lm_eval


def load_pruned_model(ckpt_path: str, device: str = "cuda"):
    """Load LLM-Pruner checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    pruned_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = pruned_dict["model"]
    tokenizer = pruned_dict["tokenizer"]
    del pruned_dict
    model.half().to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Fix model dimensions after LLM-Pruner column pruning:
    # The pruner reduces hidden_size but doesn't update model attributes
    actual_hidden = model.model.embed_tokens.weight.shape[1]
    if model.config.hidden_size != actual_hidden:
        print(f"  Fixing hidden_size: {model.config.hidden_size} -> {actual_hidden}")
        model.config.hidden_size = actual_hidden

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        head_dim = attn.head_dim
        # Fix attention head counts based on actual pruned weight shapes
        new_num_heads = attn.q_proj.weight.shape[0] // head_dim
        new_num_kv_heads = attn.k_proj.weight.shape[0] // head_dim
        # hidden_size in attention = num_heads * head_dim (used in view() after attention)
        new_hidden_size = new_num_heads * head_dim
        if attn.num_heads != new_num_heads:
            print(f"  Layer {i}: num_heads {attn.num_heads}->{new_num_heads}, "
                  f"kv_heads {attn.num_key_value_heads}->{new_num_kv_heads}, "
                  f"hidden_size {attn.hidden_size}->{new_hidden_size}")
        attn.num_heads = new_num_heads
        attn.num_key_value_heads = new_num_kv_heads
        attn.hidden_size = new_hidden_size

    return model, tokenizer


def evaluate_with_lm_eval(model, tokenizer, tasks: list, **kwargs) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    batch_size = kwargs.pop("batch_size", 1)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=tasks, **kwargs)
    return results["results"]


def run_humaneval(model, tokenizer) -> dict:
    print("Running HumanEval (0-shot)...")
    results = evaluate_with_lm_eval(
        model, tokenizer, tasks=["humaneval"],
        num_fewshot=0,
        confirm_run_unsafe_code=True,
        gen_kwargs="do_sample=True,temperature=0.2,top_p=0.95,max_gen_toks=512",
    )
    task_result = results.get("humaneval", {})
    pass_at_1 = task_result.get("pass@1,none",
                task_result.get("pass@1,create_test",
                task_result.get("pass@1", 0.0)))
    print(f"  HumanEval pass@1: {float(pass_at_1):.4f}")
    return {"humaneval": float(pass_at_1)}


def run_longbench(model, tokenizer) -> dict:
    print("Running LongBench (trec + triviaqa)...")
    tasks = ["longbench_trec", "longbench_triviaqa"]
    try:
        results = evaluate_with_lm_eval(model, tokenizer, tasks=tasks, num_fewshot=0, batch_size=1)
        scores = {}
        for task in tasks:
            task_result = results.get(task, {})
            score = task_result.get("rouge_score,none",
                     task_result.get("score,none",
                     task_result.get("acc,none",
                     task_result.get("exact_match,none", 0.0))))
            scores[task] = float(score)
            print(f"  {task}: {float(score):.4f}")
        return scores
    except Exception as e:
        print(f"  LongBench failed: {e}")
        return {}


def run_summarization(model, tokenizer) -> dict:
    print("Running Summarization (samsum + qmsum)...")
    tasks = ["longbench_samsum", "longbench_qmsum"]
    try:
        results = evaluate_with_lm_eval(model, tokenizer, tasks=tasks, num_fewshot=0, batch_size=1)
        scores = {}
        for task in tasks:
            task_result = results.get(task, {})
            score = task_result.get("rouge_score,none",
                     task_result.get("score,none",
                     task_result.get("acc,none", 0.0)))
            scores[task] = float(score)
            print(f"  {task}: {float(score):.4f}")
        return scores
    except Exception as e:
        print(f"  Summarization failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to pytorch_model.bin")
    parser.add_argument("--tasks", nargs="+", default=["humaneval", "longbench", "summarization"],
                        choices=["humaneval", "longbench", "summarization"])
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-name", type=str, default="", help="Model name for result metadata")
    parser.add_argument("--budget", type=float, default=0.8, help="Budget label for results")
    args = parser.parse_args()

    model, tokenizer = load_pruned_model(args.ckpt, args.device)

    all_scores = {}

    if "humaneval" in args.tasks:
        scores = run_humaneval(model, tokenizer)
        all_scores.update(scores)

    if "longbench" in args.tasks:
        scores = run_longbench(model, tokenizer)
        all_scores.update(scores)

    if "summarization" in args.tasks:
        scores = run_summarization(model, tokenizer)
        all_scores.update(scores)

    result = {
        "model_name": args.model_name,
        "method": "llmpruner",
        "budget": args.budget,
        "scores": all_scores,
        "ckpt": args.ckpt,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output}")
    print(f"Scores: {all_scores}")


if __name__ == "__main__":
    main()
