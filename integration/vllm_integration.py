#!/usr/bin/env python3
"""
vLLM Integration for mini-SWE-agent
This module provides a custom model class that integrates with your standalone vLLM server.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from minisweagent.models import Model

logger = logging.getLogger(__name__)


class VLLMModel(Model):
    """
    Custom model class for integrating with standalone vLLM server.
    This class wraps the vLLM OpenAI-compatible API endpoints.
    """

    def __init__(
        self,
        model_name: str,
        vllm_server_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize VLLMModel.

        Args:
            model_name: The model name to use (e.g., "Qwen/Qwen3-Coder-30B-A3B-Instruct")
            vllm_server_url: URL of your vLLM server (default: http://localhost:8000)
            api_key: API key if required (usually not needed for local vLLM)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        self.model_name = model_name
        self.vllm_server_url = vllm_server_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs

        # Create async HTTP client
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {}
        )

        logger.info(f"Initialized VLLMModel with server: {self.vllm_server_url}")
        logger.info(f"Model name: {self.model_name}")

    async def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate completion using vLLM server.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        # Convert messages to prompt (for non-chat models)
        prompt = self._messages_to_prompt(messages)

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stop": kwargs.get("stop", []),
            "logprobs": kwargs.get("logprobs", None),
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature", "top_p", "stop", "logprobs"]:
                payload[key] = value

        # Make request to vLLM server
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.vllm_server_url}/v1/completions",
                    json=payload
                )
                response.raise_for_status()

                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["text"]
                else:
                    raise ValueError(f"Unexpected response format: {result}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    raise e
                await asyncio.sleep(1 * (2 ** attempt))  # Exponential backoff

        raise RuntimeError("Failed to get response from vLLM server")

    async def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion tokens from vLLM server.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters

        Yields:
            Generated tokens as they arrive
        """
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stop": kwargs.get("stop", []),
            "stream": True,
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature", "top_p", "stop", "stream"]:
                payload[key] = value

        # Make streaming request
        async with self._client.stream(
            "POST",
            f"{self.vllm_server_url}/v1/completions",
            json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("text", "")
                            if delta:
                                yield delta
                    except json.JSONDecodeError:
                        continue

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message list to a single prompt string.
        This is a simple implementation - you may want to customize based on your model's requirements.
        """
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n".join(prompt_parts) + "\nAssistant: "

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def __del__(self):
        """Cleanup resources."""
        try:
            asyncio.get_event_loop().create_task(self.close())
        except RuntimeError:
            # Event loop not running
            pass


# Register the model class for easy access
_MODEL_CLASS_MAPPING = {
    "vllm": "integration.vllm_integration.VLLMModel",
}