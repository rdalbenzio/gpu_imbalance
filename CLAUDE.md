# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU load imbalancing test suite for evaluating round-robin load balancers in front of LLM inference servers. Creates artificial imbalance by varying prompt sizes based on simulated GPU concurrency.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install aiohttp

# Run mock server for testing
python mock_llm_server.py --port 8000 &

# Generator mode (creates prompts and sends them)
python gpu_imbalance_generator.py --duration 60 --num-gpus 4

# Replay mode (replays saved prompts)
python gpu_imbalance_replay.py --prompts-file prompts.txt
```

## Architecture

Two main scripts sharing a common async pattern:

- **gpu_imbalance_generator.py**: Generates prompts dynamically using English word dictionary (`words.txt`, auto-downloads if missing). Simulates round-robin GPU selection via `GPULoadBalancer` class. Prompt size algorithm:
  - Warmup phase: 2^n words for GPU0, 2^(n-1) for GPU1, etc.
  - Main phase: concurrency² words (capped at `max_prompt`)
  - Saves all prompts to `prompts.txt` with `---PROMPT_SEPARATOR---` delimiter

- **gpu_imbalance_replay.py**: Reads prompts from `prompts.txt` and replays them. Used for reproducible testing.

- **mock_llm_server.py**: OpenAI-compatible mock server for local testing. Simulates delay based on input size.

## Key Defaults

- Duration: 3600s
- Max concurrency: 512
- Max prompt: 65536 words
- Endpoint: `http://localhost:8000/v1/chat/completions`

## Data Files

- `words.txt`: English dictionary (370k words), auto-downloaded from dwyl/english-words
- `prompts.txt`: Generated prompts separated by `---PROMPT_SEPARATOR---`
