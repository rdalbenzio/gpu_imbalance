#!/usr/bin/env python3
"""
Mock LLM Server for testing GPU imbalancing scripts.
Simulates an OpenAI-compatible chat completions endpoint.
"""

import asyncio
import json
import random
import time
from aiohttp import web


async def chat_completions(request: web.Request) -> web.Response:
    """Handle chat completion requests."""
    try:
        data = await request.json()
        messages = data.get("messages", [])

        # Count input tokens (rough estimate)
        input_text = " ".join(m.get("content", "") for m in messages)
        input_tokens = len(input_text.split())

        # Simulate processing delay based on input size
        delay = min(0.01 + (input_tokens / 10000), 2.0)
        await asyncio.sleep(delay)

        # Generate response
        completion_tokens = random.randint(10, 50)

        response = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response for testing purposes."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": input_tokens + completion_tokens
            }
        }

        return web.json_response(response)

    except Exception as e:
        return web.json_response(
            {"error": {"message": str(e), "type": "server_error"}},
            status=500
        )


async def health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/health", health)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mock LLM Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    args = parser.parse_args()

    print(f"Starting mock LLM server on http://localhost:{args.port}")
    print(f"Endpoints:")
    print(f"  POST /v1/chat/completions")
    print(f"  GET  /health")

    app = create_app()
    web.run_app(app, port=args.port, print=None)
