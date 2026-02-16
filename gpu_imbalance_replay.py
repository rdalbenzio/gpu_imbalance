#!/usr/bin/env python3
"""
GPU Load Imbalancing Script - Replay Mode
Replays prompts from a previously generated prompts.txt file.
"""

import argparse
import asyncio
import aiohttp
import time
from dataclasses import dataclass, field
from typing import List, Optional


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


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 1.3 tokens per word for English)."""
    return int(len(text.split()) * 1.3)


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from the prompts.txt file."""
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


async def send_prompt(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    stats: Stats,
    semaphore: asyncio.Semaphore
) -> Optional[dict]:
    """Send a prompt to the LLM endpoint."""
    async with semaphore:
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


async def run_replay(
    endpoint: str,
    max_concurrency: int,
    prompts_file: str,
    loop_prompts: bool
):
    """Main loop for replaying prompts from file."""
    stats = Stats()
    semaphore = asyncio.Semaphore(max_concurrency)

    print(f"Loading prompts from {prompts_file}...")
    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts")

    if not prompts:
        print("Error: No prompts found in file")
        return

    print(f"Starting GPU imbalance replay...")
    print(f"  Endpoint: {endpoint}")
    print(f"  Max concurrency: {max_concurrency}")
    print(f"  Loop prompts: {loop_prompts}")
    print()

    tasks = []
    prompt_index = 0

    async with aiohttp.ClientSession() as session:
        while prompt_index < len(prompts):
            prompt = prompts[prompt_index]

            task = asyncio.create_task(
                send_prompt(session, endpoint, prompt, stats, semaphore)
            )
            tasks.append(task)

            prompt_index += 1

            # Progress update every 100 prompts
            if prompt_index % 100 == 0:
                elapsed = time.time() - stats.start_time
                print(f"Progress: {prompt_index}/{len(prompts)} prompts sent, {elapsed:.1f}s elapsed")

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)

            # If looping and reached end, reset index
            if loop_prompts and prompt_index >= len(prompts):
                prompt_index = 0
                print("Looping back to start of prompts...")

        # Wait for remaining tasks
        if tasks:
            print("Waiting for remaining requests to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)

    stats.print_table()


def main():
    parser = argparse.ArgumentParser(
        description="GPU Load Imbalancing Script - Replay Mode"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="LLM API endpoint (default: http://localhost:8000/v1/chat/completions)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=512,
        help="Maximum concurrency (default: 512)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="prompts.txt",
        help="File containing prompts to replay (default: prompts.txt)"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop through prompts continuously"
    )

    args = parser.parse_args()

    asyncio.run(run_replay(
        endpoint=args.endpoint,
        max_concurrency=args.max_concurrency,
        prompts_file=args.prompts_file,
        loop_prompts=args.loop
    ))


if __name__ == "__main__":
    main()
