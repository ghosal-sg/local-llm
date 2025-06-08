import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default model name
DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"

# Cache for loaded models and tokenizers
_model_cache = {}

def get_model_and_tokenizer(model_name):
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", offload_folder="./offload"
        )
        _model_cache[model_name] = (model, tokenizer)
    return _model_cache[model_name]

def generate_response(prompt, model_name=DEFAULT_MODEL_NAME, max_new_tokens=1024, temperature=0.7, top_p=0.9):
    """
    Generate a response from a language model given a prompt.
    """
    model, tokenizer = get_model_and_tokenizer(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
