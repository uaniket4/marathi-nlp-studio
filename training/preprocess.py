"""Tokenization and subword label alignment for token classification.

Turns word-level IOB annotations into model-ready features. The key concern is
aligning labels with WordPiece subwords: only the first subword of each word
keeps its label; continuation subwords and special tokens get ``-100`` so they
are ignored by the loss and by evaluation.
"""

from __future__ import annotations

from typing import Dict, List

from transformers import PreTrainedTokenizerBase


def tokenize_and_align_labels(
    tokens: List[List[str]],
    tags: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    max_length: int,
) -> Dict[str, list]:
    """Tokenize pre-split words and align a label to each subword.

    Args:
        tokens: list of sentences, each a list of word strings.
        tags: parallel list of IOB tag strings.
        tokenizer: a fast tokenizer (needs ``word_ids``).
        label2id: mapping from tag string to integer id.
        max_length: truncation length.

    Returns:
        A dict with ``input_ids``, ``attention_mask`` and ``labels``.
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    aligned_labels: List[List[int]] = []
    for i, tag_seq in enumerate(tags):
        word_ids = encoding.word_ids(batch_index=i)
        label_ids: List[int] = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[tag_seq[word_id]])
            else:
                # Continuation subword of the same word -> ignore in loss.
                label_ids.append(-100)
            prev_word_id = word_id
        aligned_labels.append(label_ids)

    encoding["labels"] = aligned_labels
    return dict(encoding)
