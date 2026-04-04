"""Evaluate LLM-Pruner pruned model — PPL and GPU memory (nvidia-smi).

Uses the same evaluation logic as RAP for fair comparison:
- WikiText-2 test split, concatenated and chunked to seq_len
- Global token-level average NLL → exp → PPL
- GPU memory via nvidia-smi (process-level)

Usage:
    cd baselines/LLM-Pruner
    .venv/bin/python rap_baseline_scripts/eval_ppl.py --ratio 0.4
    .venv/bin/python rap_baseline_scripts/eval_ppl.py  # dense model
"""
import argparse
import gc
import math
import subprocess
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_eval_dataloader(tokenizer, dataset_name="wikitext2", seq_len=2048, batch_size=8):
    """Load test split — same logic as RAP's load_calibration_dataset."""
    if dataset_name == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in ds if row["text"].strip()]
    elif dataset_name == "ptb":
        ds = load_dataset("ptb_text_only", split="test", revision="refs/convert/parquet")
        texts = [row["sentence"] for row in ds if row["sentence"].strip()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wikitext2' or 'ptb'.")

    full_text = " ".join(texts)
    tokens = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"].squeeze(0)

    n_tokens = (len(input_ids) // seq_len) * seq_len
    input_ids = input_ids[:n_tokens].reshape(-1, seq_len)
    attention_mask = torch.ones_like(input_ids)

    dataset = TensorDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device):
    """Evaluate PPL — same logic as RAP's evaluate_perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for batch in dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        shift_mask = attention_mask[:, 1:].contiguous().view(-1)
        loss = loss * shift_mask
        total_tokens += shift_mask.sum().item()
        total_loss += loss.sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def get_nvidia_smi_memory():
    """Get GPU memory usage from nvidia-smi in GB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return int(result.stdout.strip()) / 1024
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-Pruner model (RAP-consistent)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--ratio", type=float, default=None, help="Pruning ratio (None = dense model)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(42)
    datasets_to_eval = ["wikitext2", "ptb"]

    # Load model
    if args.ratio is not None:
        ckpt_path = f"prune_log/llama2_ratio_{args.ratio}/pytorch_model.bin"
        print(f"Loading pruned model: {ckpt_path}", flush=True)
        pruned_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = pruned_dict["model"]
        tokenizer = pruned_dict["tokenizer"]
        del pruned_dict
    else:
        print(f"Loading dense model: {args.model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.half().to(args.device)
    model.eval()
    model.config.use_cache = False

    # Evaluate on all datasets
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    ppl_results = {}
    total_time = 0.0
    for ds_name in datasets_to_eval:
        dataloader = load_eval_dataloader(tokenizer, dataset_name=ds_name, seq_len=args.seq_len, batch_size=args.batch_size)
        t0 = time.time()
        ppl = evaluate_perplexity(model, dataloader, torch.device(args.device))
        elapsed = time.time() - t0
        ppl_results[ds_name] = ppl
        total_time += elapsed
        print(f"  {ds_name}: PPL={ppl:.2f} ({elapsed:.1f}s)", flush=True)

    torch.cuda.synchronize()
    gpu_nvidia_smi = get_nvidia_smi_memory()

    print(f"\n{'='*40}", flush=True)
    for ds_name, ppl in ppl_results.items():
        print(f"  PPL ({ds_name}): {ppl:.2f}", flush=True)
    print(f"  GPU (nvidia-smi): {gpu_nvidia_smi:.2f} GB", flush=True)
    print(f"  Total eval time: {total_time:.1f} s", flush=True)
    print(f"{'='*40}", flush=True)


if __name__ == "__main__":
    main()
