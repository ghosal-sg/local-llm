import argparse
from utils import generate_response, DEFAULT_MODEL_NAME

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference from the terminal.")
    parser.add_argument("prompt", type=str, help="Prompt to send to the model.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability.")
    args = parser.parse_args()

    response = generate_response(
        prompt=args.prompt,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    print(response)

if __name__ == "__main__":
    main()
