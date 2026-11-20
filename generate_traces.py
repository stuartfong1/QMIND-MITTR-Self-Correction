"""Generate DeepSeek reasoning traces on a sample of GSM-8K training problems.

For each sampled problem, the model reasons inside <think> ... </think>. Only the text
inside the thinking block is kept, and the traces are saved as JSON Lines to
{checkpoint_dir}/{exp_name}/reasoning_traces_train.jsonl for use in train_probe.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.gsm8k_dataset import GSM8KDataset, format_prompt
from model.model_registry import ModelRegistry

# Number of training problems to generate reasoning traces for
NUMBER_OF_PROBLEMS = 200


def main(config: dict) -> None:
    """Generate and save reasoning traces for a random sample of training problems.

    Parameters
    ----------
    config : dict
        Settings from the experiment YAML config

    """
    ModelRegistry.set_config(config)

    retrieved_model = ModelRegistry.get_model(config["model_name"])
    model = retrieved_model.model
    tokenizer = retrieved_model.tokenizer

    train_dataset = GSM8KDataset(Path(config["dataset_path"]), split="train")

    # Sample problems without replacement, with a fixed seed so the sample is reproducible
    random_generator = np.random.default_rng(seed=0)
    sampled_indices = random_generator.choice(len(train_dataset), size=NUMBER_OF_PROBLEMS, replace=False)

    trace_path = Path(config["checkpoint_dir"]) / config["exp_name"] / "reasoning_traces_train.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with trace_path.open("w") as trace_file:
        for problem_number, dataset_index in enumerate(sampled_indices):
            question, correct_answer = train_dataset[int(dataset_index)]

            # The prompt ends with "<think>\n", so everything generated starts inside the thinking block.
            # The chat template already contains the BOS token, so don't add another one.
            prompt_text = format_prompt(tokenizer, question)
            model_inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)

            # Generate with the sampling settings recommended for the R1 distills
            with torch.inference_mode():
                generated_token_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                )

            # Decode only the newly generated tokens, skipping the prompt
            prompt_length = model_inputs.input_ids.shape[1]
            generated_text = tokenizer.decode(generated_token_ids[0, prompt_length:], skip_special_tokens=True)

            # Keep only the reasoning before "</think>". If generation hit max_new_tokens before
            # closing the thinking block, the whole (truncated) output is reasoning.
            reasoning_trace = generated_text.split("</think>")[0]

            # One JSON object per line
            trace_record = {"question": question, "correct_answer": correct_answer, "reasoning_trace": reasoning_trace}
            trace_file.write(json.dumps(trace_record) + "\n")

            print(f"Generated trace {problem_number + 1}/{NUMBER_OF_PROBLEMS}")

    print(f"Saved reasoning traces to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning traces on GSM-8K training problems.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML config")
    args = parser.parse_args()

    with args.config.open() as config_file:
        config = yaml.safe_load(config_file)

    main(config)
