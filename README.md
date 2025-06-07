# local-llm

Scripts to run LLMs locally using HuggingFace models.

## Requirements

- Python 3.8+
- macOS, Linux, or Windows
- Recommended: a machine with a GPU for faster inference

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

The main script is `main.py`. You can run it from the terminal to generate responses from a local LLM.

### Basic Command

```bash
python main.py "<your prompt here>"
```

### Optional Arguments

- `--model_name`: HuggingFace model name (default: deepseek-ai/deepseek-llm-7b-chat)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 1000)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Nucleus sampling probability (default: 0.9)

#### Example

```bash
python main.py "What is the capital of France?" --model_name deepseek-ai/deepseek-llm-7b-chat --max_new_tokens 200
```

## File Structure

- `main.py`: Entry point for running LLM inference from the terminal
- `utils.py`: Helper functions for loading models and generating responses
- `requirements.txt`: Python dependencies
- `notebooks/llm_inference.ipynb`: Example Jupyter notebook for LLM inference

## Notes

- The first run for a new model will download weights from HuggingFace.
- For best performance, use a machine with a compatible GPU and CUDA drivers.
