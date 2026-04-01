#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone vLLM server implementation extracted from verl codebase.
This implementation follows the exact same approach as verl's vLLM integration.
"""

import argparse
import asyncio
import inspect
import json
import logging
import os
import socket
import sys
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from packaging import version

try:
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import build_app, init_app_state
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    # Check vLLM version compatibility
    _VLLM_VERSION = version.parse(vllm.__version__)

    if _VLLM_VERSION > version.parse("0.11.0"):
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.utils.network_utils import get_tcp_uri
    else:
        from vllm.utils import FlexibleArgumentParser, get_tcp_uri

except ImportError as e:
    print(f"Required dependency 'vllm' not found: {e}")
    print("Please install vLLM: pip install vllm")
    sys.exit(1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class VLLMServerConfig:
    """Configuration class mirroring verl's RolloutConfig for vLLM server."""

    def __init__(self, **kwargs):
        # Core configuration from verl's rollout.yaml
        self.dtype = kwargs.get('dtype', 'bfloat16')
        self.load_format = kwargs.get('load_format', 'auto')
        self.max_model_len = kwargs.get('max_model_len', None)
        self.max_num_seqs = kwargs.get('max_num_seqs', 1024)
        self.enable_chunked_prefill = kwargs.get('enable_chunked_prefill', True)
        self.max_num_batched_tokens = kwargs.get('max_num_batched_tokens', 8192)
        self.enable_prefix_caching = kwargs.get('enable_prefix_caching', True)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.5)
        self.disable_log_stats = kwargs.get('disable_log_stats', True)
        self.tensor_model_parallel_size = kwargs.get('tensor_model_parallel_size', 1)
        self.seed = kwargs.get('seed', 0)
        self.enforce_eager = kwargs.get('enforce_eager', False)
        self.cudagraph_capture_sizes = kwargs.get('cudagraph_capture_sizes', None)

        # Generation parameters
        self.temperature = kwargs.get('temperature', 1.0)
        self.top_k = kwargs.get('top_k', -1)
        self.top_p = kwargs.get('top_p', 1.0)
        self.response_length = kwargs.get('response_length', 512)

        # LoRA configuration
        self.lora_rank = kwargs.get('lora_rank', 0)
        self.enable_lora = kwargs.get('enable_lora', False)

        # Quantization
        self.quantization = kwargs.get('quantization', None)
        self.quantization_config_file = kwargs.get('quantization_config_file', None)

        # Engine kwargs
        self.engine_kwargs = kwargs.get('engine_kwargs', {})

        # Prometheus
        self.prometheus = kwargs.get('prometheus', {'enable': False, 'served_model_name': None})

        # Expert parallel (MoE)
        self.expert_parallel_size = kwargs.get('expert_parallel_size', 1)
        self.data_parallel_size = kwargs.get('data_parallel_size', 1)
        self.enable_rollout_routing_replay = kwargs.get('enable_rollout_routing_replay', False)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    This function automatically adjusts the `max_lora_rank` to the nearest allowed value.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank
    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")


def is_valid_ipv6_address(address: str) -> bool:
    """Check if the given address is a valid IPv6 address."""
    try:
        import ipaddress
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    """Get a free port on the given address."""
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        # SO_REUSEPORT not available on Windows
        pass
    sock.bind((address, 0))
    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    """Run uvicorn server asynchronously."""
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server.should_exit = True
            await server.serve()
            server_task = asyncio.create_task(server.main_loop())
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


class StandaloneVLLMServer:
    """Standalone vLLM server implementation based on verl's approach."""

    def __init__(self, config: VLLMServerConfig, model_path: str):
        self.config = config
        self.model_path = model_path
        self.engine = None
        self._server_address = "127.0.0.1"  # Default to localhost
        self._server_port = None

        # Initialize master address and port for potential multi-node setup
        self._master_address = self._server_address
        self._master_port, self._master_sock = get_free_port(self._server_address)
        self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)

        logger.info(f"StandaloneVLLMServer initialized with model: {model_path}")
        logger.info(f"Master address: {self._master_address}, Master port: {self._master_port}")

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self):
        """Launch the vLLM server following verl's exact implementation."""

        # Setup vLLM serve CLI args exactly as in verl
        engine_kwargs = self.config.engine_kwargs.get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # Override default generation config
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        quantization = self.config.quantization
        hf_overrides = {}

        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file

        # Build args dictionary exactly as in verl
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "trust_remote_code": True,  # Default to True for flexibility
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "logprobs_mode": "processed_logprobs",  # Default from verl
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.seed,
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            **engine_kwargs,
        }

        # Handle prometheus configuration
        if self.config.prometheus.get("enable", False):
            served_model_name = self.config.prometheus.get("served_model_name")
            if served_model_name and "/" in served_model_name:
                served_model_name = served_model_name.split("/")[-1]
            if served_model_name:
                args["served_model_name"] = served_model_name

        # Handle expert parallel configuration
        if self.config.expert_parallel_size > 1:
            data_parallel_size_local = 1  # For standalone, assume single node
            args.update({
                "enable_expert_parallel": True,
                "data_parallel_size": self.config.data_parallel_size,
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_start_rank": 0,
                "data_parallel_address": self._master_address,
                "data_parallel_rpc_port": self._master_port,
            })

        # Update LoRA-related args
        if self.config.lora_rank > 0 or self.config.enable_lora:
            args.update({
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(self.config.lora_rank if self.config.lora_rank > 0 else 8),
            })

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        # Build server arguments list
        server_args = ["serve", self.model_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        logger.info("vLLM server arguments:")
        logger.info(server_args)

        # Parse arguments using vLLM's CLI parser
        CMD_MODULES = []
        try:
            import vllm.entrypoints.cli.serve as serve_module
            CMD_MODULES = [serve_module]
        except ImportError:
            # Fallback for older vLLM versions
            pass

        if CMD_MODULES:
            parser = FlexibleArgumentParser(description="vLLM CLI")
            subparsers = parser.add_subparsers(required=False, dest="subparser")
            cmds = {}
            for cmd_module in CMD_MODULES:
                if hasattr(cmd_module, 'cmd_init'):
                    new_cmds = cmd_module.cmd_init()
                    for cmd in new_cmds:
                        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                        cmds[cmd.name] = cmd
            parsed_args = parser.parse_args(args=server_args)
            parsed_args.model = getattr(parsed_args, 'model_tag', self.model_path)
            if parsed_args.subparser in cmds:
                cmds[parsed_args.subparser].validate(parsed_args)
        else:
            # Fallback: create namespace manually
            parsed_args = argparse.Namespace()
            for i in range(0, len(server_args)):
                if server_args[i].startswith("--"):
                    arg_name = server_args[i][2:]
                    if i + 1 < len(server_args) and not server_args[i + 1].startswith("--"):
                        setattr(parsed_args, arg_name.replace("-", "_"), server_args[i + 1])
                    else:
                        setattr(parsed_args, arg_name.replace("-", "_"), True)
            parsed_args.model = self.model_path

        # Launch server
        await self.run_server(parsed_args)

    async def run_server(self, args: argparse.Namespace):
        """Run the vLLM server."""
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        if hasattr(engine_client, 'reset_mm_cache'):
            await engine_client.reset_mm_cache()

        app = build_app(args)
        if _VLLM_VERSION > version.parse("0.11.0"):
            await init_app_state(engine_client, app.state, args)
        else:
            await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> dict:
        """Generate sequence with token-in-token-out (simplified version)."""
        # Calculate max tokens
        max_possible_tokens = self.config.max_model_len - len(prompt_ids) if self.config.max_model_len else 2048
        if max_possible_tokens < 0:
            raise ValueError(f"Prompt length ({len(prompt_ids)}) exceeds available context space")

        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = self.config.response_length

        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", 1.0)
        sampling_params_obj = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)

        # Add LoRA request if enabled
        lora_request = None
        if self.config.lora_rank > 0 or self.config.enable_lora:
            VLLM_LORA_INT_ID = 123
            VLLM_LORA_NAME = "123"
            VLLM_LORA_PATH = "simon_lora_path"

            if hasattr(self.engine, 'list_loras'):
                lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
                if lora_loaded:
                    lora_request = LoRARequest(
                        lora_name=VLLM_LORA_NAME,
                        lora_int_id=VLLM_LORA_INT_ID,
                        lora_path=VLLM_LORA_PATH
                    )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params_obj,
            request_id=request_id,
            lora_request=lora_request
        )

        final_res = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params_obj.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay and hasattr(final_res.outputs[0], 'routed_experts'):
            routed_experts = final_res.outputs[0].routed_experts

        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        return {
            "token_ids": token_ids,
            "log_probs": log_probs,
            "routed_experts": routed_experts,
            "stop_reason": stop_reason
        }


async def main():
    """Main entry point for the standalone vLLM server."""
    parser = argparse.ArgumentParser(description="Standalone vLLM server based on verl implementation")
    parser.add_argument("model", help="Path to the model directory or Hugging Face model ID")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--max-model-len", type=int, help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--enable-lora", action="store_true", help="Enable LoRA support")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--quantization", choices=["fp8", "torchao"], help="Quantization method")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create configuration
    config = VLLMServerConfig(
        dtype=args.dtype,
        tensor_model_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=args.enable_lora,
        lora_rank=args.lora_rank,
        quantization=args.quantization,
        load_format="auto",
        response_length=512,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
    )

    # Create and launch server
    server = StandaloneVLLMServer(config, args.model)
    await server.launch_server()

    # Keep server running
    server_address, server_port = server.get_server_address()
    logger.info(f"vLLM server is running at http://{server_address}:{server_port}")
    logger.info("Press Ctrl+C to stop the server")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        if hasattr(server, '_server_task') and server._server_task:
            server._server_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())