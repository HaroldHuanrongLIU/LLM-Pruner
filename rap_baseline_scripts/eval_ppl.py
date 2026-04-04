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
    """Evaluate PPL and latency — same logic as RAP's evaluate_perplexity.

    Returns: (ppl, latency_sec, n_samples)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = 0

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Warmup
    warmup_batch = next(iter(dataloader))
    model(input_ids=warmup_batch[0].to(device), attention_mask=warmup_batch[1].to(device))
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
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
        n_samples += input_ids.size(0)
    end_event.record()

    torch.cuda.synchronize()
    latency_sec = start_event.elapsed_time(end_event) / 1000.0

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss), latency_sec, n_samples


def get_gpu_memory_gb(device_index=0):
    """Get GPU memory usage via nvidia-ml-py (pynvml) in GB."""
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return mem_info.used / 1024**3


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

    results = {}
    for ds_name in datasets_to_eval:
        dataloader = load_eval_dataloader(tokenizer, dataset_name=ds_name, seq_len=args.seq_len, batch_size=args.batch_size)
        ppl, latency, n_samples = evaluate_perplexity(model, dataloader, torch.device(args.device))
        results[ds_name] = {"ppl": ppl, "latency_sec": latency, "n_samples": n_samples}
        print(f"  {ds_name}: PPL={ppl:.2f}, latency={latency:.2f}s, samples={n_samples}, per_sample={latency/n_samples*1000:.1f}ms", flush=True)

    torch.cuda.synchronize()
    gpu_mem_gb = get_gpu_memory_gb()

    print(f"\n{'='*40}", flush=True)
    for ds_name, r in results.items():
        print(f"  PPL ({ds_name}): {r['ppl']:.2f}", flush=True)
        print(f"  Latency ({ds_name}): {r['latency_sec']:.2f}s ({r['latency_sec']/r['n_samples']*1000:.1f}ms/sample)", flush=True)
    print(f"  GPU memory: {gpu_mem_gb:.2f} GB", flush=True)
    print(f"{'='*40}", flush=True)


if __name__ == "__main__":
    main()
