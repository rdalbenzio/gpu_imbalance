#!/usr/bin/env python3
"""
GPU Load Imbalancing Script - Generator Mode
Creates imbalance between n GPUs when round-robin load balancing is used.
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
    """Estimate token count (roughly 0.75 tokens per word for English)."""
    return int(len(text.split()) * 1.3)


class GPULoadBalancer:
    """Simulates tracking of round-robin GPU selection and concurrency."""

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.current_gpu = 0
        self.concurrency = [0] * num_gpus

    def get_next_gpu(self) -> int:
        gpu = self.current_gpu
        self.current_gpu = (self.current_gpu + 1) % self.num_gpus
        return gpu

    def get_concurrency(self, gpu_id: int) -> int:
        return self.concurrency[gpu_id]

    def increment_concurrency(self, gpu_id: int):
        self.concurrency[gpu_id] += 1

    def decrement_concurrency(self, gpu_id: int):
        self.concurrency[gpu_id] = max(0, self.concurrency[gpu_id] - 1)


async def send_prompt(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    stats: Stats,
    balancer: GPULoadBalancer,
    gpu_id: int,
    semaphore: asyncio.Semaphore
) -> Optional[dict]:
    """Send a prompt to the LLM endpoint."""
    async with semaphore:
        balancer.increment_concurrency(gpu_id)
        stats.prompts_sent += 1
        tokens_sent = estimate_tokens(prompt)
        stats.tokens_generated += tokens_sent

        try:
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }

            async with session.post(
                endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    stats.prompts_received += 1
                    # Extract tokens from response
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
        finally:
            balancer.decrement_concurrency(gpu_id)


async def run_generator(
    endpoint: str,
    num_gpus: int,
    duration: int,
    max_concurrency: int,
    max_prompt: int,
    warmup_loops: int,
    prompts_file: str
):
    """Main loop for generating and sending prompts."""
    stats = Stats()
    balancer = GPULoadBalancer(num_gpus)
    semaphore = asyncio.Semaphore(max_concurrency)

    start_time = time.time()
    loop_count = 0
    tasks = []

    print(f"Starting GPU imbalance generator...")
    print(f"  Endpoint: {endpoint}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Duration: {duration}s")
    print(f"  Max concurrency: {max_concurrency}")
    print(f"  Max prompt words: {max_prompt}")
    print(f"  Warmup loops: {warmup_loops}")
    print()

    async with aiohttp.ClientSession() as session:
        with open(prompts_file, 'w') as f:
            while time.time() - start_time < duration:
                gpu_id = balancer.get_next_gpu()

                if loop_count < warmup_loops * num_gpus:
                    # Initial phase: 2^n words for GPU0, 2^(n-1) for GPU1, etc.
                    gpu_index = loop_count % num_gpus
                    exponent = num_gpus - gpu_index
                    word_count = min(2 ** exponent, max_prompt)
                else:
                    # After warmup: concurrency^2 words
                    concurrency = balancer.get_concurrency(gpu_id)
                    word_count = min(max((concurrency + 1) ** 2, 4), max_prompt)

                prompt = generate_prompt(word_count)
                f.write(prompt + "\n---PROMPT_SEPARATOR---\n")
                f.flush()

                task = asyncio.create_task(
                    send_prompt(session, endpoint, prompt, stats, balancer, gpu_id, semaphore)
                )
                tasks.append(task)

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
        description="GPU Load Imbalancing Script - Generator Mode"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="LLM API endpoint (default: http://localhost:8000/v1/chat/completions)"
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
        "--warmup-loops", "-m",
        type=int,
        default=10,
        help="Number of warmup loops before switching to concurrency-based (default: 10)"
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

    # Load word list
    print(f"Loading word list from {args.words_file}...")
    WORD_LIST = load_word_list(args.words_file)
    print(f"Loaded {len(WORD_LIST)} words")

    asyncio.run(run_generator(
        endpoint=args.endpoint,
        num_gpus=args.num_gpus,
        duration=args.duration,
        max_concurrency=args.max_concurrency,
        max_prompt=args.max_prompt,
        warmup_loops=args.warmup_loops,
        prompts_file=args.prompts_file
    ))


if __name__ == "__main__":
    main()
