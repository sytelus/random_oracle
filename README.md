

<div align="center">
<h1>Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</h1>

[![PyPI](https://img.shields.io/pypi/v/verbalized-sampling?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/verbalized-sampling/) [![Python](https://img.shields.io/pypi/pyversions/verbalized-sampling?style=for-the-badge&logo=python&logoColor=white&label=)](https://pypi.org/project/verbalized-sampling/) [![Homepage](https://img.shields.io/badge/Homepage-4d8cd8?style=for-the-badge&logo=google-chrome&logoColor=white)](https://www.verbalized-sampling.com/) [![Paper](https://img.shields.io/badge/Paper-2510.01171-red?style=for-the-badge)](https://arxiv.org/abs/2510.01171)  [![Blog](https://img.shields.io/badge/Blog-4d8cd8?style=for-the-badge&logo=notion&logoColor=white)](https://simonucl.notion.site/verbalized-sampling)
</div>

---

<p align="center">
  <a href="#quickstart">Quickstart</a> | <a href="#colab-notebooks">Colab</a> | <a href="#researchers">Researchers</a> | <a href="https://arxiv.org/abs/2510.01171">Paper</a> | <a href="https://simonucl.notion.site/verbalized-sampling">Blog</a> | <a href="#citation">Citation</a>
</p>

**Verbalized Sampling (VS)** is a simple prompting strategy that improves LLM diversity by 2-3x. It works by asking the model to generate multiple responses with their probabilities, then sampling from this distribution. VS is **training-free** (works with any LLM via prompting), **model-agnostic** (GPT, Claude, Gemini, Llama, etc.), **orthogonal to temperature**, and effective across tasks like **creative writing**, **social simulation**, **synthetic data generation**, and **open-ended QA**.

## Quickstart

To try Verbalized Sampling, just copy and paste this into any chatbot (ChatGPT, Claude, Gemini, etc.):

```
Generate 10 responses to the user query, each within a separate <response> tag. Each response should be 50-100 words.
Each <response> must include a <text> and a numeric <probability>. Randomly sample the responses from the full distribution.

<user_query>Tell me a joke.</user_query>
```

If you want more jokes, just respond and ask "Tell me 10 more jokes" in the same conversation. For even better results, paste this into a system prompt instead:

```
You are a helpful assistant. For each query, please generate a set of five possible responses, each within a separate <response> tag. Responses should each include a <text> and a numeric <probability>. Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10.
```

For all of the above in a single function call, the ability to automatically sample from the verbalized responses, and LangChain integration, run `pip install verbalized-sampling` and use it as follows:

```python
# Set OPENAI_API_KEY or OPENROUTER_API_KEY in bash
from verbalized_sampling import verbalize

# Generate distribution of responses
dist = verbalize("Tell me a joke", k=5, tau=0.10, temperature=0.9)

# Sample from the distribution
joke = dist.sample(seed=42)
print(joke.text)
```

## Colab Notebooks

Here are some examples of how to use verbalized sampling for generating more diverse stories, ideas, images, and how to use our package:

| Notebook                           | Description                                                                                                                                  | Code                                             | Run it Yourself!                                                                                                                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Direct vs. Verbalized Sampling** | Head-to-head comparison demonstrating VS effectiveness: 2-3x diversity improvement in creative tasks while maintaining quality               | [View on GitHub](notebooks/vs_base.ipynb)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UDk4W5w6gF0dQ9Tpu0sPQethEht51GXL#offline=true&sandboxMode=true) |
| **Image Generation with VS**       | Visual comparison of Direct Prompting vs. Verbalized Sampling for text-to-image generation, showcasing creative diversity in artistic styles | [View on GitHub](notebooks/vs_with_image.ipynb)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J18VJRnrCjIb6sTivY-znb8C3JsLQCIz#offline=true&sandboxMode=true) |
| **Complete Framework Tutorial**    | Step-by-step guide to using verbalized sampling: API basics, transforms, selection methods, recipes, and advanced features                   | [View on GitHub](notebooks/framework_demo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eC0nIUVC1kyANxxzhNib44qmPphdWy9o#offline=true&sandboxMode=true) |

## Researchers

Our library includes everything you need to reproduce the results from our paper. For example:

```bash
# Run creative writing experiments
python scripts/tasks/run_poem.py --model gpt-4.1 --methods direct vs_standard --num-responses 50

# Evaluate bias mitigation on geographic data
python scripts/tasks/run_state_name.py --model anthropic/claude-sonnet-4 --methods direct vs_standard

# Compare diversity metrics across methods
python scripts/tasks/run_story.py --model gpt-4.1 --methods direct vs_standard vs_cot --metrics diversity ngram
```

For complete experiment instructions with exact commands, parameter settings, and expected outputs, see **[EXPERIMENTS.md](scripts/EXPERIMENTS.md)** which provides 1-to-1 mapping between paper sections and experiment scripts.

## Citation

If you use Verbalized Sampling in your research, please cite our paper:

```bibtex
@misc{zhang2025verbalizedsamplingmitigatemode,
  title={Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity},
  author={Jiayi Zhang and Simon Yu and Derek Chong and Anthony Sicilia and Michael R. Tomz and Christopher D. Manning and Weiyan Shi},
  year={2025},
  eprint={2510.01171},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.01171}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
