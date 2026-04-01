#!/usr/bin/env python3
"""
Simple launcher script for the standalone vLLM server.
This script provides a minimal interface to launch the vLLM server
with the same configuration approach as verl.
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_server import StandaloneVLLMServer, VLLMServerConfig


async def launch_vllm_server(model_path: str, **config_kwargs):
    """
    Launch a vLLM server with verl-compatible configuration.

    Args:
        model_path: Path to the model or Hugging Face model ID
        **config_kwargs: Configuration parameters matching verl's rollout.yaml

    Returns:
        StandaloneVLLMServer instance
    """
    # Create configuration with defaults from verl
    config = VLLMServerConfig(**config_kwargs)

    # Create server instance
    server = StandaloneVLLMServer(config, model_path)

    # Launch server
    await server.launch_server()

    return server


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python launch_vllm.py <model_path> [options]")
        print("Example: python launch_vllm.py meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2")
        sys.exit(1)

    model_path = sys.argv[1]

    # Parse additional arguments (simplified)
    config_kwargs = {}
    if "--tensor-parallel-size" in sys.argv:
        idx = sys.argv.index("--tensor-parallel-size")
        if idx + 1 < len(sys.argv):
            config_kwargs["tensor_model_parallel_size"] = int(sys.argv[idx + 1])

    if "--dtype" in sys.argv:
        idx = sys.argv.index("--dtype")
        if idx + 1 < len(sys.argv):
            config_kwargs["dtype"] = sys.argv[idx + 1]

    if "--max-model-len" in sys.argv:
        idx = sys.argv.index("--max-model-len")
        if idx + 1 < len(sys.argv):
            config_kwargs["max_model_len"] = int(sys.argv[idx + 1])

    if "--gpu-memory-utilization" in sys.argv:
        idx = sys.argv.index("--gpu-memory-utilization")
        if idx + 1 < len(sys.argv):
            config_kwargs["gpu_memory_utilization"] = float(sys.argv[idx + 1])

    if "--enable-lora" in sys.argv:
        config_kwargs["enable_lora"] = True
        config_kwargs["lora_rank"] = 8  # default

    # Launch server
    try:
        server = asyncio.run(launch_vllm_server(model_path, **config_kwargs))
        print(f"vLLM server launched successfully!")
        print(f"Server address: {server.get_server_address()}")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error launching server: {e}")
        sys.exit(1)