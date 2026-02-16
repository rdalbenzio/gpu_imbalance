# GPU Load Imbalancing Scripts

Tools to create and test GPU load imbalance when round-robin load balancing is used.

## Scripts

### 1. Generator Script (`gpu_imbalance_generator.py`)

Generates prompts dynamically and sends them to an LLM endpoint while saving them to a file.

**Algorithm:**
- **Warmup phase (M loops):** Sends prompts with k^n words for GPU0, k^(n-1) for GPU1, down to k^1 for GPUn (k is configurable, default 2)
- **Main phase:** Retrieves current GPU concurrency and sends prompts of length `concurrency²` (capped at `max_prompt`)

```bash
python gpu_imbalance_generator.py \
    --endpoint http://localhost:8000/v1/chat/completions \
    --num-gpus 4 \
    --duration 3600 \
    --max-concurrency 512 \
    --max-prompt 65536 \
    --warmup-loops 10 \
    --exp-base 2 \
    --prompts-file prompts.txt
```

### 2. Replay Script (`gpu_imbalance_replay.py`)

Replays prompts from a previously generated `prompts.txt` file.

```bash
python gpu_imbalance_replay.py \
    --endpoint http://localhost:8000/v1/chat/completions \
    --max-concurrency 512 \
    --prompts-file prompts.txt \
    --loop  # Optional: continuously loop through prompts
```

## Options

### Generator Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000/v1/chat/completions` | LLM API endpoint |
| `--num-gpus`, `-n` | 4 | Number of GPUs behind load balancer |
| `--duration`, `-t` | 3600 | Test duration in seconds |
| `--max-concurrency` | 512 | Maximum concurrent requests |
| `--max-prompt` | 65536 | Maximum prompt length in words |
| `--warmup-loops`, `-m` | 10 | Loops before switching to concurrency-based sizing |
| `--exp-base`, `-k` | 2 | Base for exponential prompt sizing in warmup phase (minimum: 2) |
| `--prompts-file` | `prompts.txt` | Output file for generated prompts |

### Replay Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000/v1/chat/completions` | LLM API endpoint |
| `--max-concurrency` | 512 | Maximum concurrent requests |
| `--prompts-file` | `prompts.txt` | Input file with prompts to replay |
| `--loop` | false | Continuously loop through prompts |

## Output

Both scripts print a summary table at completion:

```
============================================================
RESULTS SUMMARY
============================================================
Metric                                              Value
------------------------------------------------------------
Total prompts sent                                   1000
Total prompts received                                950
Total prompts failed                                   50
Total tokens generated                             125000
Total tokens received                               95000
Avg tokens/sec generated                            34.72
Avg tokens/sec received                             26.39
Elapsed time (seconds)                            3600.00
============================================================
```

## Requirements

```bash
pip install aiohttp
```

## How It Works

The scripts simulate a round-robin load balancer by tracking which GPU would receive each request. The imbalance is created by:

1. Initially sending exponentially different prompt sizes to each GPU (e.g., with `--exp-base 3 --num-gpus 4`: GPU0=81, GPU1=27, GPU2=9, GPU3=3 words)
2. Then adapting prompt size based on current GPU concurrency (more load = larger prompts)

This creates a feedback loop where busy GPUs receive increasingly larger prompts, exacerbating the imbalance.
