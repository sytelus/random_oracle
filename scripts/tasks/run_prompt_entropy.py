#!/usr/bin/env python3
"""Run local experiments comparing Prompt Entropy to existing baselines."""

from pathlib import Path

from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import (
    EvaluationConfig,
    ExperimentConfig,
    Pipeline,
    PipelineConfig,
)
from verbalized_sampling.tasks import Task


def build_experiments(model_name: str) -> list[ExperimentConfig]:
    """Create experiment configurations for the comparison study."""
    shared = dict(
        task=Task.JOKE,
        model_name=model_name,
        num_prompts=6,
        num_responses=20,
        temperature=0.8,
        top_p=0.95,
        target_words=0,
        random_seed=42,
    )

    return [
        ExperimentConfig(
            name="direct_baseline",
            method=Method.DIRECT,
            num_samples=1,
            strict_json=False,
            **shared,
        ),
        ExperimentConfig(
            name="vs_standard",
            method=Method.VS_STANDARD,
            num_samples=5,
            strict_json=True,
            probability_definition="explicit",
            **shared,
        ),
        ExperimentConfig(
            name="prompt_entropy",
            method=Method.PROMPT_ENTROPY,
            num_samples=1,
            strict_json=False,
            **shared,
        ),
    ]


def main() -> None:
    model_name = "claude-3.5-sonnet"
    experiments = build_experiments(model_name)

    pipeline = Pipeline(
        PipelineConfig(
            experiments=experiments,
            evaluation=EvaluationConfig(metrics=["ngram", "length"], num_workers=2),
            output_base_dir=Path("experiments/prompt_entropy_jokes"),
            num_workers=2,
            skip_existing=False,
            rerun=False,
            title="Prompt Entropy vs Baselines (Tiny GPT-2, Joke Task)",
        )
    )

    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
