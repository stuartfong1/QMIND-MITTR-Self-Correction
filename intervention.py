"""Steer the model towards self-correction by intervening in one layer.

A forward hook replaces the layer's output residual stream h with
h + intervention_factor * d, where d is self-correction direction saved by
train_probe.py.
The steered model then generates reasoning traces for a set of math problems,
which are printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

from data.gsm8k_dataset import GSM8KDataset, format_prompt
from model.model_registry import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


def make_steering_hook(
    self_correction_direction: torch.Tensor,
    intervention_factor: float,
) -> Callable:
    """Build a forward hook that adds a scaled direction to a layer's output.

    Parameters
    ----------
    self_correction_direction : torch.Tensor
        Unit-norm direction of shape (residual_dim,).
    intervention_factor : float
        Steering strength alpha, so the output becomes h + alpha * direction.

    Returns
    -------
    Callable
        Steering hook

    """

    def steering_hook(
        module: nn.Module,
        input: tuple,
        output: torch.Tensor | tuple,
    ) -> torch.Tensor | tuple:
        """Add the scaled direction to every token position of the layer output."""
        # Depending on the `transformers` version, a decoder layer returns either the
        # hidden states directly or a tuple whose first element is the hidden states
        if isinstance(output, tuple):
            steered_hidden_states = output[0] + intervention_factor * self_correction_direction
            return (steered_hidden_states, *output[1:])

        # Broadcasting adds the (residual_dim,) direction to every (batch, position) entry
        return output + intervention_factor * self_correction_direction

    return steering_hook


def main(config: dict) -> None:
    """Register the steering hook and print the steered reasoning traces.

    Parameters
    ----------
    config : dict
        Settings from the experiment YAML config

    """
    # Give the config to the registry so it can build the model
    ModelRegistry.set_config(config)

    retrieved_model = ModelRegistry.get_model(config["model_name"])
    model = retrieved_model.model
    tokenizer = retrieved_model.tokenizer

    # Load the unit-norm direction from train_probe.py
    probe_direction_path = (
        Path(config["checkpoint_dir"])
        / config["exp_name"]
        / f"self_correction_direction_layer_{config['probe_layer']}.npy"
    )
    self_correction_direction = torch.from_numpy(
        np.load(probe_direction_path),
    ).to(device=model.device, dtype=model.dtype)

    # Attach the hook to the same layer the probe was trained on:
    # model.model.layers[probe_layer] produces hidden_states[probe_layer + 1]
    steering_hook = make_steering_hook(
        self_correction_direction,
        config["intervention_factor"],
    )
    probe_layer_module = model.model.layers[config["probe_layer"]]
    hook_handle = probe_layer_module.register_forward_hook(steering_hook)

    test_dataset = GSM8KDataset(Path(config["dataset_dir"]), split="test")

    for question, correct_answer in test_dataset:
        # Format the problem with the same prompt used to generate the probe's training traces.
        # The chat template already contains the BOS token, so don't add another one.
        prompt_text = format_prompt(tokenizer, question)
        model_inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)

        # Generate with the sampling settings recommended for the R1 distills
        # (temperature 0.6, top-p 0.95), since greedy decoding tends to repeat itself.
        # The hook is active for every forward pass, i.e. the prompt and each new token.
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
        steered_reasoning_trace = tokenizer.decode(generated_token_ids[0, prompt_length:], skip_special_tokens=True)

        print(f"=== Problem: {question} ===")
        print(steered_reasoning_trace)
        print(f"=== Correct answer: {correct_answer} ===")
        print()

    # Detach the hook so the cached model is left unmodified
    hook_handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steer the model towards self-correction.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML config")
    args = parser.parse_args()

    with args.config.open() as config_file:
        config = yaml.safe_load(config_file)

    main(config)
