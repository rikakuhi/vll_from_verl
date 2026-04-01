# vLLM Integration with mini-SWE-agent

This directory contains the integration code to connect mini-SWE-agent with your standalone vLLM server.

## Files Overview

- `vllm_integration.py`: Custom model class that wraps your vLLM server's OpenAI-compatible API
- `vllm_config.yaml`: Configuration file for mini-SWE-agent to use your vLLM server
- `continuous_task.py`: Script to run continuous tests against your vLLM server
- `test_tasks.json`: Sample test tasks focused on code generation (relevant for Qwen3-Coder)

## Setup Instructions

### 1. Install Dependencies

```bash
cd E:\code\vllm_from_verl
pip install -r requirements.txt
```

### 2. Start Your vLLM Server

```bash
# In one terminal
python launch_vllm.py Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --tensor-parallel-size <your_tp_size> \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --enable-chunked-prefill \
    --enable-prefix-caching
```

### 3. Run Continuous Tests

```bash
# In another terminal
python integration/continuous_task.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --vllm-url "http://localhost:8000" \
    --iterations 100 \
    --delay 1.0 \
    --output-dir "test_results_qwen3"
```

### 4. Use with mini-SWE-agent CLI

You can also use the standard mini-SWE-agent CLI with your custom configuration:

```bash
# Install mini-swe-agent in development mode
pip install -e mini-swe-agent

# Run with custom vLLM configuration
mini -c integration/vllm_config.yaml -t "Write a Python function to implement binary search."
```

## Testing for Exclamation Mark Issues

The `continuous_task.py` script is specifically designed to help you reproduce and debug the exclamation mark issue you're experiencing in VERL:

- **Automatically detects responses that contain only exclamation marks**
- **Logs detailed results including response content, errors, and timing**
- **Runs multiple iterations to increase chances of reproducing intermittent issues**
- **Saves results to JSON files for analysis**

If the continuous testing reveals exclamation mark issues with your standalone vLLM server, this indicates that the problem is likely in the vLLM configuration or model itself, rather than VERL's integration logic.

If no issues are found with the standalone server, the problem is likely in VERL's weight synchronization or training loop integration.

## Configuration Options

### VLLMModel Parameters

- `model_name`: The model name to use (required)
- `vllm_server_url`: URL of your vLLM server (default: http://localhost:8000)
- `api_key`: API key if required (usually not needed for local vLLM)
- `timeout`: Request timeout in seconds (default: 300)
- `max_retries`: Maximum number of retry attempts (default: 3)

### Continuous Task Runner Options

- `--model`: Model name to test
- `--vllm-url`: vLLM server URL
- `--tasks-file`: JSON file with custom test tasks
- `--iterations`: Number of test iterations to run
- `--delay`: Delay between tasks in seconds
- `--output-dir`: Directory to save test results

## Troubleshooting

### Common Issues

1. **Connection refused**: Make sure your vLLM server is running and accessible
2. **Model not found**: Verify the model name matches what's loaded in your vLLM server
3. **Timeout errors**: Increase the timeout parameter or check GPU memory usage

### Debugging Tips

- Check the vLLM server logs for any error messages
- Monitor GPU memory usage during testing
- Try with smaller batch sizes or reduced context lengths if running out of memory
- Test with different temperature and top_p values to see if the issue is related to sampling parameters

## License

This integration code follows the same Apache License 2.0 as the main repository.