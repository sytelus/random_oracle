# Adding a New Sampling Method

This project treats *methods* as the primary abstraction for trying alternative prompting or decoding strategies. A method touches prompt construction, response parsing, schemas for strict JSON mode, and end-user surfaces like the CLI. This guide walks through every place that needs to be updated, explains the knobs you can tune, and finishes with a worked example that adds an entropy-injected joke sampler.

---

## Where Methods Plug In

- `verbalized_sampling/methods/factory.py` – owns the `Method` enum, helper classifiers (e.g., `is_method_multi_turn`), and the `PromptFactory` that builds system/user messages.
- `verbalized_sampling/methods/prompt.py` – prompt templates per `TaskType`; also defines format prompts used in strict JSON mode.
- `verbalized_sampling/methods/schema.py` – JSON/tool schemas used when `strict_json=True`.
- `verbalized_sampling/methods/parser.py` – normalizes raw model outputs into the `{ "text": ..., "probability": ... }` shape that the pipeline expects.
- `verbalized_sampling/tasks/base.py` – runtime glue that selects the right execution path (single turn, multi-turn, base-model, combined) based on helper predicates.
- `verbalized_sampling/cli.py` and `scripts/tasks/*.py` – user entry points; keep method descriptions and example configs discoverable.

You rarely need to touch evaluation code: once the parser emits the canonical structure, the rest of the pipeline, metrics, and report generation work automatically.

---

## Step-by-Step Checklist

1. **Name the method**
   - In `verbalized_sampling/methods/factory.py`, add a new member to `class Method(str, Enum)`.
   - If the method deserves a paper-friendly display string, extend `Method.paper_name`.
   - Update helper predicates when relevant:
     - `is_method_structured` if the model must return JSON.
     - `is_method_multi_turn` if it needs iterative prompts.
     - `is_method_combined` when you plan to reuse the VS-Multi continuation loop.
     - `is_method_base_model` for raw completion (no chat template).
   - Add a human-readable entry to `METHOD_DESCRIPTIONS` in `verbalized_sampling/cli.py`.

2. **Wire the prompt generation**
   - Decide which prompt *type* the method should reuse. Map it inside `PromptFactory._get_prompt_type_from_method`. If you need a brand-new prompt flavor, add a new string key and implement the corresponding `get_*_prompt` method in every relevant template class in `verbalized_sampling/methods/prompt.py`.
   - Structured outputs append a format string defined by `PromptTemplateFactory.get_template(...).get_format_prompt(...)`. To opt-in, extend `PromptFactory.METHOD_TO_FORMAT`.
   - If your prompts depend on per-task context (for example, you want a task-specific instruction or continuation), add helpers in the template subclasses (e.g., `CreativityPromptTemplate`) rather than sprinkling conditionals into `PromptFactory.pack_prompt`.

3. **Describe the JSON or tool schema**
   - When `strict_json=True`, models rely on schemas from `verbalized_sampling/methods/schema.py`.
   - Add a new branch in `get_schema` and, if the schema should work with Claude tool-calling, add a matching branch in `get_tool_schema`. Reuse `_create_structured_response_with_field_schema` or create a bespoke helper if the payload differs.
   - If the method introduces a new `probability_definition`, register it in both `PromptFactory.PROBABILITY_DEFINITIONS` and `_get_probability_field_info`.

4. **Parse the model output**
   - Extend the `match` statement in `ResponseParser.parse_response`.
   - Implement a parser that returns a `List[Dict[str, Any]]`. Each dict *must* at least contain `"text"`. Include `"probability"` (or the alternate field name produced by your schema) when available. Reuse `maybe_rename_response` if the LLM emits an alias such as `"confidence"`.
   - Keep the parser forgiving: handle dicts, lists, and raw strings to support non-strict runs and partial failures.

5. **Ensure tasks know how to run it**
   - If the method flips one of the helper predicates (multi-turn, combined, base model), confirm that `BaseTask.run` already covers that code path. Otherwise, add a new branch alongside `_run_multi_turn`, `_run_combined`, or `_run_base_model`.
   - For methods that change how responses are chunked (e.g., different `num_samples_per_prompt` semantics), update `Pipeline.run_generation` or the task constructor kwargs accordingly.

6. **Surface it to users**
   - Document the method in `NEW_METHOD.md` (this file) or `README.md` as appropriate.
   - Optionally add the method to a convenience script under `scripts/tasks/` so experiments can be launched with a single command.
   - If the method needs environment variables or assets, record them in `docs/configuration.md` and ensure `.gitignore` protects secrets.

7. **Test end-to-end**
   - CLI sanity check: `python -m verbalized_sampling.cli list_methods` should show the new entry.
   - Dry run using a lightweight dataset (e.g., jokes): `python scripts/tasks/run_jokes.py --model openai/gpt-4.1 --methods direct <NEW_METHOD> --num-responses 2 --num-prompts 1`.
   - Inspect `pipeline_report.html` in the output directory to confirm parsing and metrics succeed.

---

## Configuration Reference

Most knobs you tweak while experimenting live in two places: `ExperimentConfig` (generation) and the per-method dicts passed to helper scripts like `scripts/tasks/run_jokes.py`. The table below summarizes the important ones.

| Name | Where it lives | Meaning |
|------|----------------|---------|
| `method` | `Method` enum value | Selects the prompting/parsing strategy. |
| `strict_json` | Script-level method dict & `ExperimentConfig` | Enables schemas when the provider supports JSON/tool calling. |
| `num_responses` | `ExperimentConfig` | Total number of outputs you want to keep per prompt. |
| `num_samples` | Script-level method dict & `BaseTask` | How many candidates the LLM should verbalize per call (for VS methods). |
| `num_prompts` | `ExperimentConfig` / task kwargs | Number of dataset prompts to draw from disk (or length of `custom_prompts`). |
| `num_samples_per_prompt` | Method dict | For VS-Multi and similar methods, controls the per-turn burst size. |
| `target_words` | `ExperimentConfig` / templates | Soft length hint forwarded to the prompt template. |
| `temperature`, `top_p`, `min_p` | `ExperimentConfig` | Standard sampling controls passed directly to the model client (`min_p` requires `use_vllm=True`). |
| `probability_definition` | Method dict & schema builder | Selects the field name and wording injected into the format prompt. Options come from `PromptFactory.PROBABILITY_DEFINITIONS`. |
| `probability_tuning` | Method dict | Adds a distribution shaping sentence (e.g., “probability below 0.08”). Leave at `-1` to disable. |
| `random_seed` | `ExperimentConfig` | Used for sampling prompts from the dataset and any internal random choices. |
| `use_vllm` | `ExperimentConfig` | Forces the vLLM backend and unlocks `min_p`. |
| `all_possible` | `ExperimentConfig` | Special mode for `STANDARD_ALL_POSSIBLE` prompts. |
| `custom_prompts` | `ExperimentConfig` | Override dataset-backed prompts with explicit strings. |
| `num_workers` | `PipelineConfig` | Thread-pool size for both generation and evaluation. |

**Probability definitions.** The built-in keys are: `implicit`, `explicit`, `relative`, `percentage`, `confidence`, `perplexity`, and `nll`. These strings determine both the wording presented to the model and the field name the parser expects.

---

## Worked Example: Entropy-Indexed Jokes

Suppose we want to reproduce the “prompt entropy” idea from `PROJECT.md`. We'll add a method that appends `Tell me joke #{k}` to each user prompt, where `k` is sampled uniformly for every request.

1. **Declare the method**
   ```python
   # verbalized_sampling/methods/factory.py
   class Method(str, Enum):
       ...
       ENTROPY_INDEXED = "entropy_indexed"

   def is_method_structured(method: Method) -> bool:
       return method in [
           ...,
           Method.ENTROPY_INDEXED,  # returns JSON with probabilities
       ]
   ```
   Add `Method.ENTROPY_INDEXED` to `METHOD_TO_FORMAT` (reusing `"vs_standard"` formatting) and optional display text inside `METHOD_DESCRIPTIONS` in `verbalized_sampling/cli.py`.

2. **Prompt wiring**
   ```python
   # verbalized_sampling/methods/factory.py
   @staticmethod
   def _get_prompt_type_from_method(method: Method, all_possible: bool = False) -> str:
       if method == Method.ENTROPY_INDEXED:
           return "standard"
       ...
   ```
   Inside `PromptFactory.pack_prompt`, inject the index:
   ```python
   if method == Method.ENTROPY_INDEXED:
       index = random.randint(1, 10_000)
       prompt = f"{prompt}\nUse the joke numbered {index}."
   ```
   Because we mapped the prompt type to `"standard"` and the format type to `"vs_standard"`, every task automatically receives the correct system instructions and JSON formatting.

3. **Schema and parser**
   - `get_schema` can reuse `_create_structured_response_with_field_schema("explicit")`.
   - In `ResponseParser.parse_response`, call `parse_structure_with_probability` for the new enum branch to decode the `{text, probability}` pairs.

4. **Surface it**
   - Update `METHOD_DESCRIPTIONS` with a short explanation: “Verbalized sampling + indexed prompt entropy.”
   - Optionally add a convenience entry inside `scripts/tasks/run_jokes.py`:
     ```python
     methods = [
         {
             "method": Method.ENTROPY_INDEXED,
             "strict_json": True,
             "num_samples": 5,
             "probability_definition": "explicit",
         },
         ...
     ]
     ```

5. **Run a quick check**
   ```bash
   python scripts/tasks/run_jokes.py --model openai/gpt-4.1 --methods entropy_indexed --num-responses 4 --num-prompts 1
   ```
   Inspect the output directory (default `joke_experiments_test/...`) to verify the recorded probabilities and the injected index in prompts.

---

## Troubleshooting Tips
- **Schema errors** (“tool arguments failed validation”): confirm the schema and the parser agree on field names (`probability` vs. `confidence`).
- **Empty outputs**: disable `strict_json` temporarily to see the raw string and adjust the parser.
- **Multi-turn loops stalling**: check `num_samples_per_prompt` and the helper predicates so the correct branch in `BaseTask.run` fires.
- **Unknown method in CLI**: confirm the enum import in `scripts/tasks/...` pulls the new member (restart your REPL if an earlier session reused the old enum).

With these steps, adding a method becomes an incremental exercise: wire the enum, teach the prompt factory, tell the parser what to expect, and the experiments pipeline will do the rest.

