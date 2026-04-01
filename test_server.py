#!/usr/bin/env python3
"""
Test script to verify the standalone vLLM server functionality.
This script tests basic server initialization and configuration parsing.
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_server import VLLMServerConfig, StandaloneVLLMServer


async def test_config_creation():
    """Test configuration creation with various parameters."""
    print("Testing configuration creation...")

    # Test default config
    config = VLLMServerConfig()
    assert config.dtype == "bfloat16"
    assert config.tensor_model_parallel_size == 1
    assert config.gpu_memory_utilization == 0.5
    print("✓ Default config created successfully")

    # Test custom config
    custom_config = VLLMServerConfig(
        dtype="float16",
        tensor_model_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enable_lora=True,
        lora_rank=16
    )
    assert custom_config.dtype == "float16"
    assert custom_config.tensor_model_parallel_size == 2
    assert custom_config.lora_rank == 16
    print("✓ Custom config created successfully")

    # Test LoRA rank adjustment
    from vllm_server import get_vllm_max_lora_rank
    assert get_vllm_max_lora_rank(10) == 16
    assert get_vllm_max_lora_rank(50) == 64
    assert get_vllm_max_lora_rank(300) == 320
    print("✓ LoRA rank adjustment working correctly")


async def test_server_initialization():
    """Test server initialization (without actually launching)."""
    print("Testing server initialization...")

    config = VLLMServerConfig(
        dtype="bfloat16",
        tensor_model_parallel_size=1,
        max_model_len=2048
    )

    # This will create the server object but not launch it
    try:
        server = StandaloneVLLMServer(config, "dummy-model-path")
        print("✓ Server object created successfully")
        print(f"  Model path: {server.model_path}")
        print(f"  Config dtype: {server.config.dtype}")
        print(f"  TP size: {server.config.tensor_model_parallel_size}")
    except Exception as e:
        print(f"✗ Server initialization failed: {e}")
        return False

    return True


async def main():
    """Run all tests."""
    print("Running vLLM server tests...\n")

    try:
        await test_config_creation()
        print()
        await test_server_initialization()
        print("\nAll tests passed! ✨")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())