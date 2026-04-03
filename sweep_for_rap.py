"""Sweep LLM-Pruner at multiple pruning ratios for RAP comparison.

Run in the LLM-Pruner venv:
    .venv/bin/python sweep_for_rap.py

Outputs: ../../outputs/dynamic_memory_comparison/llmpruner_sweep.json
"""
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

RATIOS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

SCENARIOS = [
    {"id": 1,  "bs": 1,  "seq_len": 4096},
    {"id": 2,  "bs": 2,  "seq_len": 2048},
    {"id": 3,  "bs": 4,  "seq_len": 2048},
    {"id": 4,  "bs": 4,  "seq_len": 4096},
    {"id": 5,  "bs": 8,  "seq_len": 1024},
    {"id": 6,  "bs": 8,  "seq_len": 2048},
    {"id": 7,  "bs": 8,  "seq_len": 4096},
    {"id": 8,  "bs": 16, "seq_len": 512},
    {"id": 9,  "bs": 16, "seq_len": 1024},
    {"id": 10, "bs": 16, "seq_len": 2048},
]


def evaluate_ppl_wikitext2(model, tokenizer, device, max_seq_len=2048):
    """Evaluate perplexity on WikiText-2."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"].squeeze(0)

    n_tokens = (len(input_ids) // max_seq_len) * max_seq_len
    input_ids = input_ids[:n_tokens].reshape(-1, max_seq_len)

    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), 2):
            batch = input_ids[i:i+2].to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.sum().item()
            total_tokens += loss.numel()

    return math.exp(total_loss / total_tokens)


def measure_peak_memory(model, bs, seq_len, device="cuda"):
    """Run forward pass and return peak GPU memory in bytes, or 'OOM'."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = torch.randint(1, 32000, (bs, seq_len), device=device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except torch.cuda.OutOfMemoryError:
            del input_ids
            torch.cuda.empty_cache()
            return "OOM"

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del input_ids
    torch.cuda.empty_cache()
    return peak


def prune_and_save(ratio):
    """Run hf_prune.py via subprocess and return checkpoint path."""
    import subprocess

    ckpt_name = f"llama2_ratio_{ratio}"
    ckpt_path = Path(f"prune_log/{ckpt_name}/pytorch_model.bin")

    if ckpt_path.exists():
        print(f"  Cached: {ckpt_path}")
        return ckpt_path

    print(f"  Running hf_prune.py (ratio={ratio})...")
    cmd = [
        sys.executable, "hf_prune.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--pruning_ratio", str(ratio),
        "--block_wise",
        "--block_mlp_layer_start", "4", "--block_mlp_layer_end", "30",
        "--block_attention_layer_start", "4", "--block_attention_layer_end", "30",
        "--pruner_type", "taylor",
        "--device", "cpu", "--eval_device", "cuda",
        "--save_ckpt_log_name", ckpt_name,
        "--save_model",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"hf_prune.py failed:\n{proc.stderr[-500:]}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Expected checkpoint at {ckpt_path}")
    return ckpt_path


def main():
    torch.manual_seed(42)
    results = []
    out_dir = Path("../../outputs/dynamic_memory_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llmpruner_sweep.json"

    for ratio in RATIOS:
        print(f"\n{'='*50}")
        print(f"  LLM-Pruner ratio={ratio}")
        print(f"{'='*50}")

        # Step 1: Prune
        try:
            ckpt_path = prune_and_save(ratio)
        except Exception as e:
            print(f"  Pruning failed: {e}")
            results.append({"ratio": ratio, "status": "failed", "error": str(e)[:200]})
            # Save partial results
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            continue

        # Step 2: Load pruned model
        print(f"  Loading pruned model...")
        pruned_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = pruned_dict["model"]
        tokenizer = pruned_dict["tokenizer"]
        model.half().cuda()
        model.eval()
        model.config.use_cache = False

        # Step 3: Evaluate PPL
        try:
            ppl = evaluate_ppl_wikitext2(model, tokenizer, "cuda")
            print(f"  PPL (WikiText-2): {ppl:.2f}")
        except Exception as e:
            print(f"  PPL eval failed: {e}")
            ppl = None

        # Step 4: Measure peak memory for each scenario
        memory_at_scenarios = {}
        for s in SCENARIOS:
            peak = measure_peak_memory(model, s["bs"], s["seq_len"])
            if peak == "OOM":
                memory_at_scenarios[s["id"]] = "OOM"
                print(f"    Scenario {s['id']} (bs={s['bs']},sl={s['seq_len']}): OOM")
            else:
                memory_at_scenarios[s["id"]] = round(peak / 1e9, 2)
                print(f"    Scenario {s['id']} (bs={s['bs']},sl={s['seq_len']}): {peak/1e9:.2f}GB")

        results.append({
            "ratio": ratio, "status": "ok",
            "ppl_wikitext2": round(ppl, 2) if ppl else None,
            "memory_at_scenarios": memory_at_scenarios,
        })

        del model, tokenizer, pruned_dict
        torch.cuda.empty_cache()
        gc.collect()

        # Save after each ratio (incremental)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone! Saved to {out_path}")


if __name__ == "__main__":
    main()
