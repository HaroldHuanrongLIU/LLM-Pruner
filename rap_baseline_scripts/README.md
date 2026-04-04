# RAP Baseline Scripts — LLM-Pruner

Scripts for evaluating LLM-Pruner as a baseline for the RAP project. All scripts use **identical evaluation logic** as RAP to ensure fair comparison.

## Scripts

### eval_ppl.py

Evaluate perplexity and GPU memory of a pruned (or dense) model.

**Evaluation protocol** (same as RAP):
- Dataset: WikiText-2 or PTB test split
- Tokenization: concatenate all text, chunk into `seq_len` sequences
- Loss: CrossEntropyLoss per token, masked by attention_mask
- PPL: exp(mean of all token-level losses)
- GPU memory: nvidia-smi process-level measurement

**Usage:**

```bash
cd baselines/LLM-Pruner

# Dense model (wikitext2)
.venv/bin/python rap_baseline_scripts/eval_ppl.py

# Pruned model (wikitext2)
.venv/bin/python rap_baseline_scripts/eval_ppl.py --ratio 0.4

# PTB dataset
.venv/bin/python rap_baseline_scripts/eval_ppl.py --ratio 0.4 --dataset ptb

# Custom batch size and seq length
.venv/bin/python rap_baseline_scripts/eval_ppl.py --ratio 0.4 --batch-size 4 --seq-len 1024
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | meta-llama/Llama-2-7b-hf | Base HuggingFace model |
| `--ratio` | None | Pruning ratio (None = dense model) |
| `--batch-size` | 8 | Batch size for evaluation |
| `--seq-len` | 2048 | Sequence length |
| `--dataset` | wikitext2 | Evaluation dataset: `wikitext2` or `ptb` |
| `--device` | cuda | PyTorch device |

**Note:** Pruned models are loaded from `prune_log/llama2_ratio_{ratio}/pytorch_model.bin`. Run LLM-Pruner's `hf_prune.py` first to generate these checkpoints.
