from collections import defaultdict
from typing import Literal

import torch

ContextMode = Literal["sentence", "local-bidirectional", "local-history"]
LOCAL_CONTEXT = {"local-bidirectional": (4, 4), "local-history": (8, 0)}


def context_window_indices(sentence_uids, left: int, right: int, device=None):
    """Build one isolated, padded window per target in the original batch order.

    Input keystrokes must be chronological within each sentence. Missing
    neighbors at a sentence boundary are masked rather than borrowed from
    another sentence. The target always occupies column ``left``.
    """
    if left < 0 or right < 0:
        raise ValueError("Context extents must be nonnegative")
    if not sentence_uids:
        raise ValueError("A context batch must contain at least one keystroke")
    groups = defaultdict(list)
    for index, uid in enumerate(sentence_uids):
        groups[uid].append(index)
    width = left + right + 1
    indices = [[0] * width for _ in sentence_uids]
    valid = [[False] * width for _ in sentence_uids]
    for group in groups.values():
        for position, target in enumerate(group):
            for column, offset in enumerate(range(-left, right + 1)):
                neighbor = position + offset
                if 0 <= neighbor < len(group):
                    indices[target][column] = group[neighbor]
                    valid[target][column] = True
    return (
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )
