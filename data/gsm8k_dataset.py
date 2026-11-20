"""Datasets for GSM-8K problems and for DeepSeek reasoning traces generated on them.

GSM8KDataset reads the main_{split}.csv files and yields (question, correct_answer).
ReasoningTraceDataset reads the traces saved by generate_traces.py and yields token IDs
together with token-level self-correction labels obtained by keyword matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# Phrases that are likely to mark a self-correction in a reasoning trace.
# Matching is case-sensitive so that e.g. "actually" in the middle of a sentence is not counted.
SELF_CORRECTION_KEYWORDS = [
    "Wait",
    "hold on",
    "any other way",
    "Hmm",
    "Actually",
    "Alternatively",
    "But wait",
    "Let me check",
    "Let me double-check",
    "Let me verify",
    "I made a mistake",
    "That's not right",
    "re-examine",
]

# A single regex matching any keyword as a whole word. Longer keywords are tried first so that
# e.g. "Let me double-check" is matched as a whole rather than stopping at a shorter prefix.
SELF_CORRECTION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(keyword) for keyword in sorted(SELF_CORRECTION_KEYWORDS, key=len, reverse=True)) + r")\b",
)

# Label for tokens that should be ignored by the probe (the prompt), as opposed to 0 or 1
IGNORE_LABEL = -1


def format_prompt(tokenizer: PreTrainedTokenizerBase, question: str) -> str:
    """Wrap a math question in the chat template, ending inside the thinking block.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Tokenizer whose chat template is applied.
    question : str
        The GSM-8K question.

    Returns
    -------
    str
        Prompt text, including the BOS token, that ends with "<think>\n" so the
        model's next tokens are its reasoning.

    """
    # DeepSeek's recommended instruction for math problems
    user_message = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True,
        tokenize=False,
    )

    # Some versions of the chat template already open the thinking block; force it otherwise
    # so that generation and probing always see the same prompt
    if not prompt_text.endswith("<think>\n"):
        prompt_text += "<think>\n"

    return prompt_text


def extract_final_answer(gsm8k_answer: str) -> str:
    """Extract the final numerical answer from a GSM-8K solution.

    Parameters
    ----------
    gsm8k_answer : str
        Worked solution, ending with "#### <answer>".

    Returns
    -------
    str
        The text after "####", with surrounding whitespace removed.

    """
    return gsm8k_answer.split("####")[-1].strip()


class GSM8KDataset(Dataset):
    """GSM-8K problems loaded from CSV files.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing main_train.csv and main_test.csv.
    split : str
        Either "train" or "test".

    """

    def __init__(self, dataset_dir: Path, split: str) -> None:
        # Columns are "question" and "answer" (worked solution ending in "#### <answer>")
        self.problems = pd.read_csv(Path(dataset_dir) / f"main_{split}.csv")

    def __len__(self) -> int:
        """Return the number of problems in the split."""
        return len(self.problems)

    def __getitem__(self, index: int) -> tuple[str, str]:
        """Return the question and its final answer.

        Parameters
        ----------
        index : int
            Index of the problem.

        Returns
        -------
        question : str
            The math word problem.
        correct_answer : str
            The final numerical answer.

        """
        problem = self.problems.iloc[index]
        return problem["question"], extract_final_answer(problem["answer"])


class ReasoningTraceDataset(Dataset):
    """Reasoning traces saved by generate_traces.py, with token-level self-correction labels.

    Each item is the full model input (prompt followed by the reasoning trace), so that
    the probe sees the same context as during generation. Prompt tokens are labelled
    IGNORE_LABEL; trace tokens are labelled 1 if they overlap a self-correction keyword
    and 0 otherwise.

    Parameters
    ----------
    trace_path : Path
        JSON Lines file with "question" and "reasoning_trace" fields.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used to tokenize the traces and align the labels with the tokens.

    """

    def __init__(self, trace_path: Path, tokenizer: PreTrainedTokenizerBase) -> None:
        with Path(trace_path).open() as trace_file:
            self.traces = [json.loads(line) for line in trace_file]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the number of reasoning traces."""
        return len(self.traces)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a prompt and reasoning trace and label each token.

        Parameters
        ----------
        index : int
            Index of the reasoning trace.

        Returns
        -------
        token_ids : torch.Tensor
            Token IDs of shape (number_of_tokens,).
        token_labels : torch.Tensor
            Labels of shape (number_of_tokens,): IGNORE_LABEL for prompt tokens,
            1 for self-correction keyword tokens, 0 for all other trace tokens.

        """
        trace = self.traces[index]
        prompt_text = format_prompt(self.tokenizer, trace["question"])
        full_text = prompt_text + trace["reasoning_trace"]

        # The chat template already contains the BOS token, so don't add another one.
        # offset_mapping gives the (start, end) character span of each token in full_text.
        tokenized = self.tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = tokenized.input_ids
        token_offsets = tokenized.offset_mapping

        # Character spans of every keyword match, searching only within the reasoning trace
        trace_start = len(prompt_text)
        keyword_spans = [match.span() for match in SELF_CORRECTION_PATTERN.finditer(full_text, pos=trace_start)]

        token_labels = []
        for token_start, token_end in token_offsets:
            if token_start < trace_start:
                # Token belongs to the prompt
                token_labels.append(IGNORE_LABEL)
            else:
                # Label 1 if the token's characters overlap any keyword span
                overlaps_keyword = any(token_start < span_end and token_end > span_start for span_start, span_end in keyword_spans)
                token_labels.append(int(overlaps_keyword))

        return torch.tensor(token_ids), torch.tensor(token_labels)
