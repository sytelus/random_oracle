# Prompt Entropy vs. Verbalized Sampling (Joke Task)

## Experiment Setup
- **Model:** `anthropic/claude-3.5-sonnet` served through the OpenRouter backend with sampling enabled (`temperature=0.8`, `top_p=0.95`).
- **Tasks & prompts:** 6 prompts drawn from `data/joke.txt`; each configuration generates 20 responses per prompt (120 total outputs).
- **Methods compared:**
  - `direct_baseline` – original direct prompting.
  - `vs_standard` – verbalized sampling baseline from the paper.
  - `prompt_entropy` – new entropy-indexed prompt strategy added in this work.
- **Metrics:** `distinct_n` (token-level diversity proxy) and `length` (token count) from the evaluation suite. Pairwise ROUGE-L is logged for methods that reuse identical prompts; the entropy variant yields undefined ROUGE-L because every prompt is unique.
- **Reproduction command:** `python -m scripts.tasks.run_prompt_entropy`
- **Artifacts:** Generation logs, evaluation JSON, plots, and HTML summary reside in `experiments/prompt_entropy_jokes/`.

## Reference Results from the Verbalized Sampling Paper

| Method (GPT-4.1) | Diversity ↑ | Rouge-L ↓ | Human Quality ↑ |
|------------------|-------------|-----------|-----------------|
| Direct           | 27.0 ± 13.1 | 61.2 ± 31.7 | 84.3 ± 12.9 |
| VS-Standard      | 60.2 ± 9.0  | 18.7 ± 20.6 | 83.4 ± 12.6 |

*Table values reproduced from Table 19 of the Verbalized Sampling paper (verbalized_sampling.pdf, p.47).*

## Claude-3.5-Sonnet Results (This Run)

| Method | Distinct-N ↑ (mean ± std) | Pairwise ROUGE-L ↓ (mean ± std) | Avg. token length | Notes |
|--------|---------------------------|---------------------------------|--------------------|-------|
| Direct baseline | 0.986 ± 0.015 | 0.554 ± 0.332 | 18.16 ± 3.77 | Single response per prompt |
| VS-Standard     | **0.987 ± 0.021** | **0.153 ± 0.194** | 17.21 ± 4.64 | Five samples verbalized then post-sampled |
| Prompt Entropy  | 0.954 ± 0.048 | n/a¹ | **38.80 ± 24.25** | Unique prompt IDs break pairwise cohorts |

## Observations
- VS-Standard continues to post the strongest pairwise ROUGE-L separation versus direct prompting (0.153 vs. 0.554), echoing the paper’s diversity improvements even though we evaluate on a different Claude-3.5-Sonnet backend.
- Prompt Entropy responses remain highly diverse (0.954 distinct-n) but trail VS-Standard by ~3 percentage points, suggesting the indexed prompt still repeats phrasing under Claude despite unique numbering.
- The entropy prompts encourage noticeably longer generations (≈39 tokens vs. ≈18 for the baselines), which may be helpful for richer jokes but complicates length-matched comparisons to the paper.
- Pairwise ROUGE-L remains undefined for Prompt Entropy because every request string is unique; Distinct-N therefore serves as the primary diversity proxy for this method.
- Compared with Table 19 of the paper (GPT-4.1: 27.0 → 60.2 diversity), our Claude-3.5 run shows a smaller absolute gap. Model differences and the reduced prompt set likely explain the smaller effect size.

## Sample Generations (Truncated)

```
Prompt Entropy (#1825):
A guy walks into a bar and orders a beer. The bartender serves it in a filthy, crusty mug. The customer says, "This glass is dirty!" The bartender leans in close and whispers, "Speak quietly, sir... that's our strongest brew."

VS-Standard (sample 1):
Why did the coffee file a police report? Because it got mugged!
```

¹ Pairwise ROUGE-L is undefined for `prompt_entropy` because no two prompts share identical wording; the evaluator therefore records `0.0`.

## Artifacts
- Generation outputs: `experiments/prompt_entropy_jokes/generation/`
- Evaluation summaries: `experiments/prompt_entropy_jokes/evaluation/`
- Plots & HTML report: `experiments/prompt_entropy_jokes/plots/`, `experiments/prompt_entropy_jokes/pipeline_report.html`
