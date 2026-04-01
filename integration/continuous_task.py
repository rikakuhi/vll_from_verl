#!/usr/bin/env python3
"""
Continuous task runner for mini-SWE-agent with vLLM integration.
This script will continuously run tasks against your deployed vLLM model.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

from integration.vllm_integration import VLLMModel
from minisweagent.agents import get_agent
from minisweagent.environments import get_environment
from minisweagent.run.utilities.config import configure_if_first_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


class ContinuousTaskRunner:
    """Continuous task runner for testing vLLM model stability."""

    def __init__(
        self,
        model_name: str,
        vllm_server_url: str = "http://localhost:8000",
        test_tasks_file: str = None,
        max_iterations: int = 100,
        delay_between_tasks: float = 1.0,
        output_dir: str = "continuous_test_results"
    ):
        self.model_name = model_name
        self.vllm_server_url = vllm_server_url
        self.max_iterations = max_iterations
        self.delay_between_tasks = delay_between_tasks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load test tasks
        if test_tasks_file and os.path.exists(test_tasks_file):
            with open(test_tasks_file, 'r', encoding='utf-8') as f:
                self.test_tasks = json.load(f)
        else:
            # Default test tasks focused on code generation (relevant for Qwen3-Coder)
            self.test_tasks = [
                "Write a Python function to implement binary search.",
                "Create a JavaScript function that validates an email address.",
                "Implement a quick sort algorithm in Python.",
                "Write a function to reverse a linked list in Python.",
                "Create a Python class for a simple calculator with add, subtract, multiply, and divide methods.",
                "Implement a function to check if a string is a palindrome.",
                "Write a Python function to find the factorial of a number using recursion.",
                "Create a function to merge two sorted arrays into one sorted array.",
                "Implement a stack data structure in Python with push, pop, and peek methods.",
                "Write a function to find the longest common prefix among an array of strings."
            ]

        logger.info(f"Loaded {len(self.test_tasks)} test tasks")
        logger.info(f"Output directory: {self.output_dir}")

    async def run_single_task(self, task: str, iteration: int) -> dict:
        """Run a single task and return results."""
        start_time = time.time()
        result = {
            "iteration": iteration,
            "task": task,
            "success": False,
            "response": "",
            "error": "",
            "duration": 0.0,
            "has_exclamation_marks_only": False,
            "has_nan_logprobs": False
        }

        try:
            # Initialize model
            model = VLLMModel(
                model_name=self.model_name,
                vllm_server_url=self.vllm_server_url,
                timeout=300
            )

            # Create environment
            env = get_environment({}, default_type="local")

            # Create agent configuration
            agent_config = {
                "agent_class": "interactive",
                "mode": "yolo",
                "cost_limit": 0,
                "confirm_exit": False
            }

            # Create agent
            agent = get_agent(model, env, agent_config, default_type="interactive")

            # Run task
            response = await model([{"role": "user", "content": task}])

            # Analyze response
            result["response"] = response
            result["success"] = True

            # Check for exclamation marks only issue
            stripped_response = response.strip().replace('!', '')
            if stripped_response == '':
                result["has_exclamation_marks_only"] = True
                logger.warning(f"Iteration {iteration}: Response contains only exclamation marks!")

            # Note: Logprobs would need to be checked in the actual vLLM response
            # For now, we'll just log if we detect suspicious patterns

            duration = time.time() - start_time
            result["duration"] = duration

            logger.info(f"Iteration {iteration}: Task completed in {duration:.2f}s")

        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg
            result["duration"] = time.time() - start_time
            logger.error(f"Iteration {iteration}: Error occurred - {error_msg}")

        finally:
            # Cleanup
            if 'model' in locals():
                await model.close()

        return result

    async def run_continuous_tests(self):
        """Run continuous tests against the vLLM server."""
        logger.info("Starting continuous task runner...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"vLLM Server: {self.vllm_server_url}")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"Delay between tasks: {self.delay_between_tasks}s")

        results = []
        exclamation_mark_issues = 0
        errors = 0

        for i in range(self.max_iterations):
            # Select task (cycle through available tasks)
            task = self.test_tasks[i % len(self.test_tasks)]

            # Run task
            result = await self.run_single_task(task, i + 1)
            results.append(result)

            # Track issues
            if result["has_exclamation_marks_only"]:
                exclamation_mark_issues += 1
            if not result["success"]:
                errors += 1

            # Save results periodically
            if (i + 1) % 10 == 0:
                self.save_results(results, f"results_batch_{(i + 1) // 10}.json")

            # Log summary
            if (i + 1) % 10 == 0:
                success_rate = (i + 1 - errors) / (i + 1) * 100
                exclamation_rate = exclamation_mark_issues / (i + 1) * 100
                logger.info(f"Progress: {i + 1}/{self.max_iterations} | "
                           f"Success: {success_rate:.1f}% | "
                           f"Exclamation issues: {exclamation_rate:.1f}%")

            # Delay between tasks
            if self.delay_between_tasks > 0:
                await asyncio.sleep(self.delay_between_tasks)

        # Final save
        self.save_results(results, "final_results.json")

        # Print final summary
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r["success"])
        exclamation_issues = sum(1 for r in results if r["has_exclamation_marks_only"])
        avg_duration = sum(r["duration"] for r in results) / total_tasks if total_tasks > 0 else 0

        console.print("\n[bold green]Continuous Testing Complete![/bold green]")
        console.print(f"Total tasks: {total_tasks}")
        console.print(f"Successful: {successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
        console.print(f"Exclamation mark issues: {exclamation_issues} ({exclamation_issues/total_tasks*100:.1f}%)")
        console.print(f"Average duration: {avg_duration:.2f}s")
        console.print(f"Results saved to: {self.output_dir}")

        # Alert if exclamation mark issues found
        if exclamation_issues > 0:
            console.print(f"\n[bold red]⚠️  EXCLAMATION MARK ISSUES DETECTED![/bold red]")
            console.print(f"This suggests the same issue you're experiencing in VERL!")
            console.print(f"Check the detailed results in {self.output_dir}")

    def save_results(self, results: List[dict], filename: str):
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")


def main(
    model_name: str = typer.Option("Qwen/Qwen3-Coder-30B-A3B-Instruct", "--model", "-m", help="Model name"),
    vllm_url: str = typer.Option("http://localhost:8000", "--vllm-url", help="vLLM server URL"),
    tasks_file: str = typer.Option(None, "--tasks-file", help="JSON file with test tasks"),
    iterations: int = typer.Option(100, "--iterations", "-n", help="Number of iterations to run"),
    delay: float = typer.Option(1.0, "--delay", "-d", help="Delay between tasks in seconds"),
    output_dir: str = typer.Option("continuous_test_results", "--output-dir", help="Output directory for results")
):
    """Run continuous tests against your vLLM server to reproduce the exclamation mark issue."""

    # Ensure first-time configuration
    configure_if_first_time()

    # Create and run continuous task runner
    runner = ContinuousTaskRunner(
        model_name=model_name,
        vllm_server_url=vllm_url,
        test_tasks_file=tasks_file,
        max_iterations=iterations,
        delay_between_tasks=delay,
        output_dir=output_dir
    )

    asyncio.run(runner.run_continuous_tests())


if __name__ == "__main__":
    typer.run(main)