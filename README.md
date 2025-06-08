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
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 1024)
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

## Acknowledgements

- This project uses the [HuggingFace Transformers](https://github.com/huggingface/transformers) library for model loading and inference.
- Model weights are downloaded from [HuggingFace Hub](https://huggingface.co/), specifically the [deepseek-ai/deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) model.
- Built with [PyTorch](https://pytorch.org/) as the backend for deep learning.

## Citations

If you use this project, please cite the following works:

### HuggingFace Transformers

```
@inproceedings{wolf-etal-2020-transformers,
  title = {Transformers: State-of-the-Art Natural Language Processing},
  author = {Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R\'emi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year = {2020},
  pages = {38--45},
  url = {https://www.aclweb.org/anthology/2020.emnlp-demos.6}
}
```

### PyTorch

```
@article{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019},
  url={https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html}
}
```

### DeepSeek LLM

- [deepseek-ai/deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) (Please refer to the model card for license and citation details.)

---

Please ensure you comply with the licenses of all dependencies and models used in this project.
