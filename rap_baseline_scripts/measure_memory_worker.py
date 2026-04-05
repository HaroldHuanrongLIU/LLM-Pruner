"""Worker: load pruned model, forward at (bs, seq_len), print peak memory."""
import argparse
import gc
import json
import os
import sys

import pynvml
import torch

# Add LLM-Pruner root for unpickling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_gpu_memory_gb(idx=0):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    pynvml.nvmlShutdown()
    return m.used / 1024**3


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

    mem_before = get_gpu_memory_gb()
    input_ids = torch.randint(1, 32000, (args.bs, args.seq_len), device="cuda")
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
            torch.cuda.synchronize()
            mem_peak = get_gpu_memory_gb()
            print(json.dumps({"status": "ok", "mem_before_gb": round(mem_before, 2), "mem_peak_gb": round(mem_peak, 2)}))
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"status": "OOM"}))


if __name__ == "__main__":
    main()
