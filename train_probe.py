"""Train a linear probe for self-correction on the residual stream of a single layer.

Each reasoning trace is passed through the model, and the hidden state of every
token is paired with a boolean is_self_correction label.
A logistic regression model is trained on these (hidden state, label) pairs, and the
self-correction direction is saved to a file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from data.gsm8k_dataset import IGNORE_LABEL, ReasoningTraceDataset
from model.model_registry import ModelRegistry

# Fraction of label-0 (non-self-correction) tokens kept for training, to reduce memory.
# All label-1 tokens are kept, since they are rare.
NEGATIVE_KEEP_FRACTION = 0.1


def main(config: dict) -> None:
    """Collect the activations from each layer and use them to train the probe.

    Parameters
    ----------
    config : dict
        Settings from the experiment YAML config

    """
    ModelRegistry.set_config(config)

    retrieved_model = ModelRegistry.get_model(config["model_name"])
    model = retrieved_model.model
    tokenizer = retrieved_model.tokenizer

    probe_layer = config["probe_layer"]

    # Activations and labels from every reasoning trace
    all_hidden_states = []
    all_token_labels = []

    # Reasoning traces saved by generate_traces.py on the GSM-8K training split
    trace_path = Path(config["checkpoint_dir"]) / config["exp_name"] / "reasoning_traces_train.jsonl"
    trace_dataset = ReasoningTraceDataset(trace_path, tokenizer)

    # Fixed seed so the subsample of label-0 tokens is reproducible
    random_generator = np.random.default_rng(seed=0)

    for token_ids, token_labels in trace_dataset:
        # Add a batch dimension: (1, num_tokens)
        input_ids = token_ids.unsqueeze(0).to(model.device)

        # Index 0 is the embedding output, so the we use index probe_layer + 1.
        with torch.inference_mode():
            model_outputs = model(input_ids=input_ids, output_hidden_states=True)
        probe_layer_hidden_states = model_outputs.hidden_states[probe_layer + 1]

        # (num_tokens, residual_dim)
        probe_layer_hidden_states = probe_layer_hidden_states[0].float().cpu().numpy()

        # Keep only the reasoning trace tokens. Prompt tokens (including the BOS
        # attention sink) are labelled IGNORE_LABEL and are not used by the probe.
        token_labels = token_labels.numpy()
        is_trace_token = token_labels != IGNORE_LABEL

        # Keep every label-1 token, and each label-0 token independently with probability
        # NEGATIVE_KEEP_FRACTION. Subsampling happens per trace, before storing, so the
        # discarded activations never accumulate in memory.
        is_kept_token = (token_labels == 1) | (random_generator.random(len(token_labels)) < NEGATIVE_KEEP_FRACTION)
        is_selected_token = is_trace_token & is_kept_token

        all_hidden_states.append(probe_layer_hidden_states[is_selected_token])
        all_token_labels.append(token_labels[is_selected_token])

    # Hidden states: (total_tokens, residual_dim)
    # Labels: (total_tokens,)
    hidden_state_matrix = np.concatenate(all_hidden_states, axis=0)
    label_vector = np.concatenate(all_token_labels, axis=0)

    # Fit a logistic regression model on the hidden states
    # Use balanced class weights since self-correction tokens are a minority class
    probe = LogisticRegression(max_iter=1000, class_weight="balanced")
    probe.fit(hidden_state_matrix, label_vector)

    # The normal w of the hyperplane points towards the self-correction class.
    # Normalise the vector to unit length
    probe_normal = probe.coef_[0]
    self_correction_direction = probe_normal / np.linalg.norm(probe_normal)

    # Save the direction for use in intervention.py
    probe_direction_path = (
        Path(config["checkpoint_dir"])
        / config["exp_name"]
        / f"self_correction_direction_layer_{probe_layer}.npy"
    )
    probe_direction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(probe_direction_path, self_correction_direction.astype(np.float32))
    print(f"Saved self-correction direction to {probe_direction_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear probe for self-correction.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML config")
    args = parser.parse_args()

    with args.config.open() as config_file:
        config = yaml.safe_load(config_file)

    main(config)
