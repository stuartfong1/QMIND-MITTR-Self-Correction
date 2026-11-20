from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class RetrievedModel:
    """Result of retrieval from the model registry.

    Parameters
    ----------
    model : nn.Module
        The causal language model.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer that converts text into the token IDs expected by the model.

    """

    model: nn.Module
    tokenizer: PreTrainedTokenizerBase


class ModelRegistry:
    """Load models and their tokenizers by name, caching each one after first use.

    Call set_config once with the experiment config before calling get_model.
    """

    _models: ClassVar[dict[str, RetrievedModel]] = {}
    _config: ClassVar[dict | None] = None

    @classmethod
    def set_config(cls, config: dict) -> None:
        """Set the config used to build the models.

        Parameters
        ----------
        config : dict
            Settings from the experiment YAML config

        """
        cls._config = config

    @classmethod
    def get_model(cls, model_name: str) -> RetrievedModel:
        """Retrieve the model if it has already been built, otherwise build it.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve

        Returns
        -------
        RetrievedModel
            Model and tokenizer

        """
        if model_name in cls._models:
            return cls._models[model_name]

        match model_name:
            case "deepseek-r1-distill-llama-8b":
                model = AutoModelForCausalLM.from_pretrained(
                    cls._config["weight_path"],
                    device_map="auto",
                    torch_dtype="auto",
                )

                model.eval()

                tokenizer = AutoTokenizer.from_pretrained(cls._config["weight_path"])

                cls._models[model_name] = RetrievedModel(
                    model=model,
                    tokenizer=tokenizer,
                )

            case _:
                msg = f"Unknown model name: {model_name}"
                raise ValueError(msg)

        return cls._models[model_name]
