"""Worker: load pruned model, forward at (bs, seq_len), print peak memory.

Uses torch.cuda.max_memory_allocated() for per-process peak measurement.
"""
import argparse
import gc
import json
import os
import sys

import torch

# Add LLM-Pruner root for unpickling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    args = parser.parse_args()

    ckpt = f"prune_log/llama2_ratio_{args.ratio}/pytorch_model.bin"
    pruned = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = pruned["model"]
    del pruned
    model.half().cuda()
    model.eval()
    model.config.use_cache = False

    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**3
    input_ids = torch.randint(1, 32000, (args.bs, args.seq_len), device="cuda")
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
            torch.cuda.synchronize()
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            print(json.dumps({"status": "ok", "mem_before_gb": round(mem_before, 2), "mem_peak_gb": round(mem_peak, 2)}))
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"status": "OOM"}))


if __name__ == "__main__":
    main()
