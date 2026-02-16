#!/usr/bin/env python3
"""
GPU Load Imbalancing Script - Warmup Only Mode
Generates prompts with exponentially different sizes based on GPU index.
Simpler version that only uses the warmup algorithm (k^n, k^(n-1), etc.)
"""

import argparse
import asyncio
import aiohttp
import time
import random
import os
from dataclasses import dataclass, field
from typing import List, Optional


# Global word list
WORD_LIST: List[str] = []


def load_word_list(words_file: str) -> List[str]:
    """Load English words from file."""
    if not os.path.exists(words_file):
        print(f"Warning: {words_file} not found, downloading...")
        import urllib.request
        url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
        urllib.request.urlretrieve(url, words_file)

    with open(words_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


@dataclass
class Stats:
    prompts_sent: int = 0
    prompts_received: int = 0
    prompts_failed: int = 0
    tokens_generated: int = 0
    tokens_received: int = 0
    start_time: float = field(default_factory=time.time)

    def print_table(self):
        elapsed = time.time() - self.start_time
        elapsed = max(elapsed, 0.001)  # Avoid division by zero

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<35} {'Value':>20}")
        print("-" * 60)
        print(f"{'Total prompts sent':<35} {self.prompts_sent:>20}")
        print(f"{'Total prompts received':<35} {self.prompts_received:>20}")
        print(f"{'Total prompts failed':<35} {self.prompts_failed:>20}")
        print(f"{'Total tokens generated':<35} {self.tokens_generated:>20}")
        print(f"{'Total tokens received':<35} {self.tokens_received:>20}")
        print(f"{'Avg tokens/sec generated':<35} {self.tokens_generated / elapsed:>20.2f}")
        print(f"{'Avg tokens/sec received':<35} {self.tokens_received / elapsed:>20.2f}")
        print(f"{'Elapsed time (seconds)':<35} {elapsed:>20.2f}")
        print("=" * 60)


def generate_prompt(word_count: int) -> str:
    """Generate a prompt with the specified number of English words."""
    words = random.choices(WORD_LIST, k=word_count)
    return ' '.join(words)


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 1.3 tokens per word for English)."""
    return int(len(text.split()) * 1.3)


async def send_prompt(
    endpoint: str,
    prompt: str,
    model: str,
    stats: Stats,
    semaphore: asyncio.Semaphore
) -> Optional[dict]:
    """Send a prompt to the LLM endpoint with a new connection."""
    async with semaphore:
        stats.prompts_sent += 1
        tokens_sent = estimate_tokens(prompt)
        stats.tokens_generated += tokens_sent

        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }

            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        stats.prompts_received += 1
                        if "usage" in result:
                            stats.tokens_received += result["usage"].get("completion_tokens", 0)
                        return result
                    else:
                        stats.prompts_failed += 1
                        return None
        except Exception as e:
            stats.prompts_failed += 1
            print(f"Request failed: {e}")
            return None


async def run_warmup(
    endpoint: str,
    model: str,
    num_gpus: int,
    duration: int,
    max_concurrency: int,
    max_prompt: int,
    exp_base: int,
    prompts_file: str
):
    """Main loop for generating exponentially sized prompts."""
    stats = Stats()
    semaphore = asyncio.Semaphore(max_concurrency)

    start_time = time.time()
    loop_count = 0
    tasks = []
    current_gpu = 0

    print()
    print("=" * 60)
    print("GPU IMBALANCE WARMUP-ONLY")
    print("=" * 60)
    print(f"  Endpoint:         {endpoint}")
    print(f"  Model:            {model}")
    print(f"  GPUs:             {num_gpus}")
    print(f"  Duration:         {duration}s")
    print(f"  Max concurrency:  {max_concurrency}")
    print(f"  Max prompt words: {max_prompt}")
    print(f"  Exponential base: {exp_base}")
    print(f"  Prompts file:     {prompts_file}")
    print()
    print("  Prompt sizes per GPU:")
    for i in range(num_gpus):
        exponent = num_gpus - i
        size = min(exp_base ** exponent, max_prompt)
        print(f"    GPU{i}: {size} words")
    print("=" * 60)
    print()

    with open(prompts_file, 'w') as f:
        while time.time() - start_time < duration:
            # Calculate word count: k^n for GPU0, k^(n-1) for GPU1, etc.
            gpu_index = current_gpu
            exponent = num_gpus - gpu_index
            word_count = min(exp_base ** exponent, max_prompt)

            prompt = generate_prompt(word_count)
            f.write(prompt + "\n")
            f.flush()

            task = asyncio.create_task(
                send_prompt(endpoint, prompt, model, stats, semaphore)
            )
            tasks.append(task)

            # Round-robin to next GPU
            current_gpu = (current_gpu + 1) % num_gpus
            loop_count += 1

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)

            # Progress update every 100 prompts
            if loop_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {loop_count} prompts sent, {elapsed:.1f}s elapsed")

    # Wait for remaining tasks
    if tasks:
        print("Waiting for remaining requests to complete...")
        await asyncio.gather(*tasks, return_exceptions=True)

    stats.print_table()
    print(f"\nPrompts saved to: {prompts_file}")


def main():
    global WORD_LIST

    parser = argparse.ArgumentParser(
        description="GPU Load Imbalancing Script - Warmup Only Mode"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="LLM API endpoint (default: http://localhost:8000/v1/chat/completions)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name to use in API requests (default: default)"
    )
    parser.add_argument(
        "--num-gpus", "-n",
        type=int,
        default=4,
        help="Number of GPUs (default: 4)"
    )
    parser.add_argument(
        "--duration", "-t",
        type=int,
        default=3600,
        help="Duration in seconds (default: 3600)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=512,
        help="Maximum concurrency (default: 512)"
    )
    parser.add_argument(
        "--max-prompt",
        type=int,
        default=65536,
        help="Maximum prompt length in words (default: 65536)"
    )
    parser.add_argument(
        "--exp-base", "-k",
        type=int,
        default=2,
        help="Base for exponential prompt sizing (default: 2, minimum: 2)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="prompts.txt",
        help="File to save generated prompts (default: prompts.txt)"
    )
    parser.add_argument(
        "--words-file",
        type=str,
        default="words.txt",
        help="File containing English words (default: words.txt)"
    )

    args = parser.parse_args()

    # Validate exp_base
    if args.exp_base < 2:
        parser.error("--exp-base must be >= 2")

    # Load word list
    print(f"Loading word list from {args.words_file}...")
    WORD_LIST = load_word_list(args.words_file)
    print(f"Loaded {len(WORD_LIST)} words")

    asyncio.run(run_warmup(
        endpoint=args.endpoint,
        model=args.model,
        num_gpus=args.num_gpus,
        duration=args.duration,
        max_concurrency=args.max_concurrency,
        max_prompt=args.max_prompt,
        exp_base=args.exp_base,
        prompts_file=args.prompts_file
    ))


if __name__ == "__main__":
    main()
