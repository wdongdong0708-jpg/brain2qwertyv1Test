from __future__ import annotations

import gc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .transforms import SPANISH_FUNCTION_WORDS

DEFAULT_SHIFT_SECONDS: tuple[float, ...] = (
    -15.0,
    -30.0,
    -45.0,
    -60.0,
    -90.0,
    -120.0,
)

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@dataclass
class SentenceWordGroup:
    sentence_uid: str
    split: str
    timeline: str
    subject_id: int
    sentence_text: str
    words: list[str]
    word_indices: np.ndarray
    onsets: np.ndarray
    durations: np.ndarray
    delta_taus: np.ndarray
    char_lengths: np.ndarray
    is_function_word: np.ndarray
    row_indices: np.ndarray


class DuplicateAwareSigLipLoss(nn.Module):
    """SigLIP loss that treats identical word targets in a batch as positives."""

    def __init__(
        self,
        initial_temperature: float = 10.0,
        initial_bias: float = -10.0,
        identical_candidates_threshold: float = 0.999,
    ):
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )
        self.bias = nn.Parameter(torch.tensor(float(initial_bias)))
        self.identical_candidates_threshold = float(identical_candidates_threshold)

    def forward(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        estimate = F.normalize(estimate.float(), dim=-1, eps=1e-12)
        candidate = F.normalize(candidate.float(), dim=-1, eps=1e-12)
        scores = estimate @ candidate.T
        logits = self.log_temperature.exp() * scores + self.bias
        sim = candidate @ candidate.T
        positives = sim >= self.identical_candidates_threshold
        targets = positives.to(logits.dtype)
        weights = (~positives).to(logits.dtype) + torch.eye(
            len(logits), device=logits.device, dtype=logits.dtype
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction="sum"
        ) / max(len(logits), 1)


def compute_shifted_word_onsets(
    word_events: pd.DataFrame,
    timeline_bounds: dict[str, tuple[float, float]],
    shift_seconds: float,
    *,
    window_start_offset: float = 0.0,
    window_duration: float = 1.0,
) -> np.ndarray:
    """Shift each sentence's word onset grid circularly within the same recording.

    Preserves the intra-sentence word interval grid (tau_{i+1} - tau_i == delta_tau_i)
    with exact floating-point identity while mapping the neural windows to historical
    offsets (e.g., -15s, -30s, -45s, -60s, -90s, -120s) in the same timeline.
    """
    if abs(float(shift_seconds)) <= 1e-9:
        return word_events["start"].to_numpy(dtype=np.float64).copy()

    shifted_starts = np.zeros(len(word_events), dtype=np.float64)
    starts = word_events["start"].to_numpy(dtype=np.float64)
    timelines = word_events["timeline"].astype(str).to_numpy()
    suids = word_events["sentence_UID"].astype(str).to_numpy()

    groups: dict[str, list[int]] = defaultdict(list)
    for pos, suid in enumerate(suids):
        groups[suid].append(pos)

    margin_left = max(0.0, -float(window_start_offset)) + 0.20
    margin_right = max(0.0, float(window_start_offset) + float(window_duration)) + 0.20

    for suid, indices in groups.items():
        idx_arr = np.asarray(indices, dtype=np.int64)
        sent_starts = starts[idx_arr]
        order = np.argsort(sent_starts, kind="stable")
        idx_sorted = idx_arr[order]
        sent_sorted = sent_starts[order]

        first_onset = float(sent_sorted[0])
        relative_offsets = sent_sorted - first_onset
        sent_span = float(relative_offsets[-1])

        tl = timelines[idx_sorted[0]]
        rec_min, rec_max = timeline_bounds[tl]
        valid_min = float(rec_min) + margin_left
        valid_max = float(rec_max) - margin_right - sent_span
        span = valid_max - valid_min
        if span <= 1e-3:
            new_first = max(float(rec_min), min(first_onset, float(rec_max) - sent_span))
        else:
            new_first = valid_min + ((first_onset + float(shift_seconds) - valid_min) % span)
        shifted_starts[idx_sorted] = new_first + relative_offsets

    return shifted_starts


def audit_word_timing_statistics(
    word_events: pd.DataFrame,
    window_durations: Sequence[float] = (0.5, 1.0),
) -> dict:
    """Compute empirical word typing duration, inter-word interval, and window overlap stats."""
    df = word_events.sort_values(["sentence_UID", "word_order"]).copy()
    delta_tau = df["delta_tau"].to_numpy(dtype=np.float64)
    durations = df["duration"].to_numpy(dtype=np.float64)
    char_lengths = df["char_length"].to_numpy(dtype=np.float64)
    is_func = df["is_function_word"].to_numpy(dtype=bool)

    non_final_mask = (df["word_order"] + 1) < df["sentence_word_count"]
    inter_word_dt = delta_tau[non_final_mask] if non_final_mask.any() else delta_tau
    inter_word_chars = (
        char_lengths[non_final_mask] if non_final_mask.any() else char_lengths
    )

    if len(inter_word_dt) > 1 and np.std(inter_word_dt) > 0 and np.std(inter_word_chars) > 0:
        corr_dt_len = float(np.corrcoef(inter_word_dt, inter_word_chars)[0, 1])
    else:
        corr_dt_len = 0.0

    overlaps = {}
    for w_dur in window_durations:
        w_f = float(w_dur)
        overlap_sec = np.clip(w_f - inter_word_dt, 0.0, w_f)
        overlap_ratio = overlap_sec / max(w_f, 1e-6)
        overlaps[f"T={w_f:.1f}s"] = {
            "window_seconds": w_f,
            "mean_overlap_ratio": float(np.mean(overlap_ratio)),
            "median_overlap_ratio": float(np.median(overlap_ratio)),
            "fraction_overlapping_pairs": float(np.mean(overlap_sec > 0.0)),
            "function_word_mean_overlap_ratio": float(
                np.mean(
                    np.clip(w_f - delta_tau[is_func & non_final_mask], 0.0, w_f) / w_f
                )
            )
            if (is_func & non_final_mask).any()
            else 0.0,
            "content_word_mean_overlap_ratio": float(
                np.mean(
                    np.clip(w_f - delta_tau[(~is_func) & non_final_mask], 0.0, w_f) / w_f
                )
            )
            if ((~is_func) & non_final_mask).any()
            else 0.0,
        }

    return {
        "total_words": int(len(df)),
        "total_sentences": int(df["sentence_UID"].nunique()),
        "unique_words": int(df["text"].nunique()),
        "delta_tau_mean_seconds": float(np.mean(inter_word_dt)),
        "delta_tau_median_seconds": float(np.median(inter_word_dt)),
        "delta_tau_p10_seconds": float(np.percentile(inter_word_dt, 10)),
        "delta_tau_p90_seconds": float(np.percentile(inter_word_dt, 90)),
        "word_duration_mean_seconds": float(np.mean(durations)),
        "function_word_delta_tau_mean_seconds": float(np.mean(delta_tau[is_func]))
        if is_func.any()
        else 0.0,
        "content_word_delta_tau_mean_seconds": float(np.mean(delta_tau[~is_func]))
        if (~is_func).any()
        else 0.0,
        "pearson_r_delta_tau_vs_char_length": corr_dt_len,
        "window_overlap_by_duration": overlaps,
    }


def _hash_word_embedding(word: str, dim: int = 1024) -> np.ndarray:
    """Deterministic length-invariant character/syllable/motor embedding for Spanish words."""
    import hashlib

    w = str(word).strip().lower()
    if not w:
        w = " "
    left_keys = set("qwertasdfgzxcvb")
    tokens: list[tuple[str, float]] = [
        (f"init1:{w[0]}", 1.5),
        (f"init2:{w[:2]}", 1.3),
        (f"end1:{w[-1]}", 1.2),
        (f"end2:{w[-2:]}", 1.2),
        (f"hand0:{'L' if w[0] in left_keys else 'R'}", 0.8),
        (f"hand_end:{'L' if w[-1] in left_keys else 'R'}", 0.8),
    ]
    padded = f"<{w}>"
    unique_grams = sorted(
        {padded[i : i + n] for n in (2, 3) for i in range(len(padded) - n + 1)}
    )
    gram_weight = 1.0 / max(np.sqrt(len(unique_grams)), 1.0)
    for gram in unique_grams:
        tokens.append((f"gram:{gram}", float(gram_weight)))

    vec = np.zeros(dim, dtype=np.float32)
    for tok, weight in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        unit = rng.standard_normal(dim, dtype=np.float32)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
        vec += float(weight) * unit
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-12)


def load_or_compute_bge_m3_embeddings(
    words: Sequence[str],
    cache_paths: Sequence[Path],
    *,
    backend: str = "bge-m3",
    device: str = "auto",
    embedding_dim: int = 1024,
) -> tuple[list[str], np.ndarray]:
    """Load cached BGE-M3 1024-d L2-normalized Spanish word embeddings or compute in Colab."""
    unique_words = sorted({str(w).strip().lower() for w in words if str(w).strip()})
    if not unique_words:
        raise ValueError("Vocabulary word list cannot be empty")

    if backend == "hash":
        raw_emb = np.stack(
            [_hash_word_embedding(w, embedding_dim) for w in unique_words]
        ).astype(np.float32)
        vocab_mean = raw_emb.mean(axis=0, keepdims=True)
        centered_emb = raw_emb - vocab_mean
        embedding_matrix_np = centered_emb / np.clip(
            np.linalg.norm(centered_emb, axis=-1, keepdims=True), 1e-12, None
        )
        return unique_words, embedding_matrix_np.astype(np.float32)

    cached_map: dict[str, np.ndarray] = {}
    for path in cache_paths:
        if path is not None and Path(path).is_file():
            try:
                data = np.load(path, allow_pickle=False)
                c_words = [str(w) for w in data["words"].tolist()]
                c_vecs = np.asarray(data["embeddings"], dtype=np.float32)
                if c_vecs.ndim == 2 and c_vecs.shape[1] == embedding_dim:
                    for w, v in zip(c_words, c_vecs):
                        cached_map[w] = v
            except Exception:
                continue

    missing = [w for w in unique_words if w not in cached_map]
    if missing:
        from transformers import AutoModel, AutoTokenizer

        run_device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        print(
            f"Encoding {len(missing)} Spanish words with {BGE_M3_MODEL_ID} on {run_device}...",
            flush=True,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                BGE_M3_MODEL_ID, revision=BGE_M3_REVISION
            )
            model = AutoModel.from_pretrained(
                BGE_M3_MODEL_ID, revision=BGE_M3_REVISION
            ).to(run_device)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(BGE_M3_MODEL_ID)
            model = AutoModel.from_pretrained(BGE_M3_MODEL_ID).to(run_device)
        model.eval()
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(missing), batch_size):
                batch_words = missing[start : start + batch_size]
                encoded = tokenizer(
                    batch_words,
                    padding=True,
                    truncation=True,
                    max_length=32,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(run_device)
                outputs = model(**encoded)
                cls_vecs = F.normalize(outputs.last_hidden_state[:, 0].float(), dim=-1)
                for w, vec in zip(batch_words, cls_vecs.cpu().numpy()):
                    cached_map[w] = vec.astype(np.float32)

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_words = sorted(cached_map.keys())
        all_vecs = np.stack([cached_map[w] for w in all_words]).astype(np.float32)
        for path in cache_paths:
            if path is not None:
                try:
                    path = Path(path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(
                        tmp, words=np.array(all_words), embeddings=all_vecs
                    )
                    tmp.replace(path)
                except Exception:
                    continue

    raw_emb = np.stack([cached_map[w] for w in unique_words]).astype(np.float32)
    vocab_mean = raw_emb.mean(axis=0, keepdims=True)
    centered_emb = raw_emb - vocab_mean
    embedding_matrix_np = centered_emb / np.clip(
        np.linalg.norm(centered_emb, axis=-1, keepdims=True), 1e-12, None
    )
    return unique_words, embedding_matrix_np.astype(np.float32)


def build_vocabulary_banks(
    word_events: pd.DataFrame,
    vocab_words: Sequence[str],
    top_k_50: int = 50,
) -> dict:
    """Build the stratified 50-word diagnostic bank (V_50, Function + Content) and full vocabulary (V_full)."""
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    all_counts = Counter(word_events["text"].astype(str).tolist())
    train_df = word_events[word_events["split"] == "train"]
    val_df = word_events[word_events["split"] == "val"]
    test_df = word_events[word_events["split"] == "test"]
    if train_df.empty:
        train_df = word_events

    train_counts = Counter(train_df["text"].astype(str).tolist())
    val_counts = Counter(val_df["text"].astype(str).tolist())
    test_counts = Counter(test_df["text"].astype(str).tolist())

    target_k = min(int(top_k_50), len(vocab_words))
    func_candidates = [
        w
        for w, _ in sorted(all_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if w in word_to_idx and w in SPANISH_FUNCTION_WORDS
    ]
    max_func = min(len(func_candidates), max(12, target_k // 3))
    selected_words: list[str] = list(func_candidates[:max_func])
    selected_set: set[str] = set(selected_words)

    rem_budget = max(0, target_k - len(selected_words))
    test_content = [
        w
        for w, _ in sorted(test_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if w in word_to_idx and w not in SPANISH_FUNCTION_WORDS and w not in selected_set
    ]
    val_content = [
        w
        for w, _ in sorted(val_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if w in word_to_idx and w not in SPANISH_FUNCTION_WORDS and w not in selected_set
    ]
    train_content = [
        w
        for w, _ in sorted(train_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if w in word_to_idx and w not in SPANISH_FUNCTION_WORDS and w not in selected_set
    ]
    all_content = [
        w
        for w, _ in sorted(all_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if w in word_to_idx and w not in SPANISH_FUNCTION_WORDS and w not in selected_set
    ]

    n_test_c = min(len(test_content), int(round(rem_budget * 0.60)))
    for w in test_content[:n_test_c]:
        if len(selected_words) < target_k and w not in selected_set:
            selected_words.append(w)
            selected_set.add(w)

    n_val_c = min(len(val_content), int(round(rem_budget * 0.25)))
    for w in val_content[:n_val_c]:
        if len(selected_words) < target_k and w not in selected_set:
            selected_words.append(w)
            selected_set.add(w)

    for pool in (train_content, test_content, val_content, all_content, func_candidates):
        for w in pool:
            if len(selected_words) >= target_k:
                break
            if w in word_to_idx and w not in selected_set:
                selected_words.append(w)
                selected_set.add(w)

    vocab50_words = selected_words[:target_k]
    vocab50_indices = np.asarray([word_to_idx[w] for w in vocab50_words], dtype=np.int64)
    full_vocab_words = [w for w in vocab_words if w in word_to_idx]
    full_vocab_indices = np.asarray(
        [word_to_idx[w] for w in full_vocab_words],
        dtype=np.int64,
    )
    return {
        "word_to_idx": word_to_idx,
        "vocab50_words": vocab50_words,
        "vocab50_indices": vocab50_indices,
        "full_vocab_words": full_vocab_words,
        "full_vocab_indices": full_vocab_indices,
        "train_word_counts": dict(train_counts),
    }


def apply_leave_one_out_cms(
    clean_predictions: torch.Tensor,
    shifted_predictions: dict[float, torch.Tensor],
    eta: float = 0.50,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Leave-One-Out Common-Mode Subtraction (LOO-CMS) across historical shifts.

    - Clean (0s) subtracts eta * mean of all M shifted predictions at the same duration grid.
    - Each Shifted (delta_k) subtracts eta * mean of the other M-1 shifted predictions,
      guaranteeing zero self-subtraction leakage and never flipping vector sign along the common mode.
    """
    clean_norm = F.normalize(clean_predictions.float(), dim=-1, eps=1e-12)
    shift_keys = sorted(shifted_predictions.keys())
    shifted_norm = {
        k: F.normalize(shifted_predictions[k].float(), dim=-1, eps=1e-12)
        for k in shift_keys
    }
    if abs(float(eta)) <= 1e-9 or len(shift_keys) < 2:
        return clean_norm, shifted_norm

    stacked = torch.stack([shifted_norm[k] for k in shift_keys], dim=0)
    sum_all = stacked.sum(dim=0)
    m = float(len(shift_keys))

    def _debias(
        vec: torch.Tensor, mean_shift: torch.Tensor, m_eff: float
    ) -> torch.Tensor:
        r = torch.norm(mean_shift, dim=-1, keepdim=True)
        mu_hat = mean_shift / torch.clamp(r, min=1e-12)
        r0 = 1.0 / max(float(np.sqrt(max(m_eff, 1.0))), 1.0)
        excess = torch.clamp((r - 0.5 * r0) / max(1.0 - 0.5 * r0, 1e-6), 0.0, 1.0)
        proj_coeff = torch.clamp(
            (vec * mu_hat).sum(dim=-1, keepdim=True), min=0.0
        )
        updated = vec - float(eta) * (
            0.5 * mean_shift + 0.5 * excess * proj_coeff * mu_hat
        )
        return F.normalize(updated, dim=-1, eps=1e-12)

    mean_all = sum_all / m
    debiased_clean = _debias(clean_norm, mean_all, m)
    debiased_shifted: dict[float, torch.Tensor] = {}
    for k in shift_keys:
        mean_other = (sum_all - shifted_norm[k]) / max(m - 1.0, 1.0)
        debiased_shifted[k] = _debias(
            shifted_norm[k], mean_other, max(m - 1.0, 1.0)
        )
    return debiased_clean, debiased_shifted


def evaluate_word_retrieval(
    predictions: torch.Tensor | np.ndarray,
    target_word_indices: np.ndarray,
    is_function_word: np.ndarray,
    embedding_bank: np.ndarray,
    vocab50_indices: np.ndarray,
    full_vocab_indices: np.ndarray,
) -> dict:
    """Compute 50-word and full-vocabulary Top-10 and Top-1 retrieval accuracies."""
    if isinstance(predictions, torch.Tensor):
        pred_np = F.normalize(predictions.detach().cpu().float(), dim=-1).numpy()
    else:
        pred_np = np.asarray(predictions, dtype=np.float32)
        pred_np = pred_np / np.maximum(
            np.linalg.norm(pred_np, axis=-1, keepdims=True), 1e-12
        )

    targets = np.asarray(target_word_indices, dtype=np.int64)
    func_mask = np.asarray(is_function_word, dtype=bool)
    bank = np.asarray(embedding_bank, dtype=np.float32)

    def _bank_metrics(candidate_indices: np.ndarray, restrict_queries: bool) -> dict:
        cand_idx = np.asarray(candidate_indices, dtype=np.int64)
        cand_pos_map = {int(w_id): pos for pos, w_id in enumerate(cand_idx.tolist())}
        in_bank = np.array([int(t) in cand_pos_map for t in targets], dtype=bool)
        query_mask = in_bank if restrict_queries else np.ones(len(targets), dtype=bool)
        if not query_mask.any():
            return {
                "queries": 0,
                "candidates": int(len(cand_idx)),
                "chance_top10": 100.0 * min(10, len(cand_idx)) / max(len(cand_idx), 1),
                "chance_top1": 100.0 / max(len(cand_idx), 1),
                "micro_top10": 0.0,
                "macro_top10": 0.0,
                "micro_top1": 0.0,
                "macro_top1": 0.0,
                "function_top10": 0.0,
                "content_top10": 0.0,
            }

        q_preds = pred_np[query_mask]
        q_targets = targets[query_mask]
        q_func = func_mask[query_mask]
        cand_vecs = bank[cand_idx]
        scores = q_preds @ cand_vecs.T

        k10 = min(10, len(cand_idx))
        top10_cols = np.argpartition(-scores, kth=k10 - 1, axis=1)[:, :k10]
        top1_col = np.argmax(scores, axis=1)

        hit10 = np.zeros(len(q_targets), dtype=bool)
        hit1 = np.zeros(len(q_targets), dtype=bool)
        for i, t_id in enumerate(q_targets.tolist()):
            pos = cand_pos_map.get(int(t_id), -1)
            if pos >= 0:
                hit10[i] = pos in top10_cols[i]
                hit1[i] = int(top1_col[i]) == pos

        by_type10: dict[int, list[bool]] = defaultdict(list)
        by_type1: dict[int, list[bool]] = defaultdict(list)
        by_func_type10: dict[int, list[bool]] = defaultdict(list)
        by_cont_type10: dict[int, list[bool]] = defaultdict(list)
        for t_id, is_f, h10, h1 in zip(
            q_targets.tolist(), q_func.tolist(), hit10.tolist(), hit1.tolist()
        ):
            if int(t_id) in cand_pos_map:
                by_type10[int(t_id)].append(h10)
                by_type1[int(t_id)].append(h1)
                if is_f:
                    by_func_type10[int(t_id)].append(h10)
                else:
                    by_cont_type10[int(t_id)].append(h10)

        macro10 = (
            100.0 * float(np.mean([np.mean(v) for v in by_type10.values()]))
            if by_type10
            else 0.0
        )
        macro1 = (
            100.0 * float(np.mean([np.mean(v) for v in by_type1.values()]))
            if by_type1
            else 0.0
        )
        func_macro10 = (
            100.0 * float(np.mean([np.mean(v) for v in by_func_type10.values()]))
            if by_func_type10
            else (100.0 * float(np.mean(hit10[q_func])) if q_func.any() else 0.0)
        )
        cont_macro10 = (
            100.0 * float(np.mean([np.mean(v) for v in by_cont_type10.values()]))
            if by_cont_type10
            else (100.0 * float(np.mean(hit10[~q_func])) if (~q_func).any() else 0.0)
        )
        return {
            "queries": int(query_mask.sum()),
            "candidates": int(len(cand_idx)),
            "chance_top10": 100.0 * k10 / max(len(cand_idx), 1),
            "chance_top1": 100.0 / max(len(cand_idx), 1),
            "micro_top10": 100.0 * float(np.mean(hit10)),
            "macro_top10": macro10,
            "micro_top1": 100.0 * float(np.mean(hit1)),
            "macro_top1": macro1,
            "function_top10": 100.0 * float(np.mean(hit10[q_func]))
            if q_func.any()
            else 0.0,
            "content_top10": 100.0 * float(np.mean(hit10[~q_func]))
            if (~q_func).any()
            else 0.0,
            "function_macro_top10": func_macro10,
            "content_macro_top10": cont_macro10,
        }

    return {
        "vocab50": _bank_metrics(vocab50_indices, restrict_queries=True),
        "full_vocab": _bank_metrics(full_vocab_indices, restrict_queries=False),
    }


def summarize_clean_vs_shifted(
    clean_preds: torch.Tensor,
    shifted_preds_map: dict[float, torch.Tensor],
    target_word_indices: np.ndarray,
    is_function_word: np.ndarray,
    embedding_bank: np.ndarray,
    vocab50_indices: np.ndarray,
    full_vocab_indices: np.ndarray,
) -> dict:
    """Evaluate Clean (0s), each Shifted condition, their aggregate mean, and Net Gain."""
    clean_metrics = evaluate_word_retrieval(
        clean_preds,
        target_word_indices,
        is_function_word,
        embedding_bank,
        vocab50_indices,
        full_vocab_indices,
    )
    per_shift = {
        f"{shift:+.0f}s": evaluate_word_retrieval(
            s_preds,
            target_word_indices,
            is_function_word,
            embedding_bank,
            vocab50_indices,
            full_vocab_indices,
        )
        for shift, s_preds in sorted(shifted_preds_map.items())
    }

    def _agg(bank_key: str, metric_key: str) -> tuple[float, float]:
        vals = [m[bank_key][metric_key] for m in per_shift.values()]
        return (
            (float(np.mean(vals)), float(np.std(vals)))
            if vals
            else (0.0, 0.0)
        )

    v50_macro10_mean, v50_macro10_std = _agg("vocab50", "macro_top10")
    v50_micro10_mean, v50_micro10_std = _agg("vocab50", "micro_top10")
    v50_macro1_mean, _ = _agg("vocab50", "macro_top1")
    fv_macro10_mean, _ = _agg("full_vocab", "macro_top10")
    fv_micro10_mean, _ = _agg("full_vocab", "micro_top10")
    func10_mean, _ = _agg("vocab50", "function_top10")
    cont10_mean, _ = _agg("vocab50", "content_top10")
    fv_func10_mean, _ = _agg("full_vocab", "function_top10")
    fv_cont10_mean, _ = _agg("full_vocab", "content_top10")

    c_v50 = clean_metrics["vocab50"]
    c_fv = clean_metrics["full_vocab"]
    return {
        "clean": clean_metrics,
        "shifted_per_anchor": per_shift,
        "shifted_mean": {
            "vocab50_macro_top10": v50_macro10_mean,
            "vocab50_macro_top10_std": v50_macro10_std,
            "vocab50_micro_top10": v50_micro10_mean,
            "vocab50_micro_top10_std": v50_micro10_std,
            "vocab50_macro_top1": v50_macro1_mean,
            "full_vocab_macro_top10": fv_macro10_mean,
            "full_vocab_micro_top10": fv_micro10_mean,
            "function_top10": func10_mean,
            "content_top10": cont10_mean,
            "full_vocab_function_top10": fv_func10_mean,
            "full_vocab_content_top10": fv_cont10_mean,
        },
        "net_gain": {
            "vocab50_macro_top10": c_v50["macro_top10"] - v50_macro10_mean,
            "vocab50_micro_top10": c_v50["micro_top10"] - v50_micro10_mean,
            "vocab50_macro_top1": c_v50["macro_top1"] - v50_macro1_mean,
            "full_vocab_macro_top10": c_fv["macro_top10"] - fv_macro10_mean,
            "full_vocab_micro_top10": c_fv["micro_top10"] - fv_micro10_mean,
            "function_top10": c_v50["function_top10"] - func10_mean,
            "content_top10": c_v50["content_top10"] - cont10_mean,
            "full_vocab_function_top10": c_fv["function_top10"] - fv_func10_mean,
            "full_vocab_content_top10": c_fv["content_top10"] - fv_cont10_mean,
        },
        "shortcut_inflation_over_chance_v50_top10": v50_macro10_mean
        - c_v50["chance_top10"],
    }


def compute_streaming_latency_stats(
    onsets: np.ndarray,
    durations: np.ndarray,
    sentence_uids: Sequence[str],
    window_offset: float,
    window_duration: float,
) -> dict[str, dict[str, float]]:
    """Compute empirical post-onset and post-word-completion wait latencies (s) per streaming horizon."""
    onsets_arr = np.asarray(onsets, dtype=np.float64)
    durs_arr = np.asarray(durations, dtype=np.float64)
    w_end = float(window_offset) + float(window_duration)

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, suid in enumerate(sentence_uids):
        groups[str(suid)].append(idx)

    horizons_map: dict[str, int | None] = {
        "K0": 0,
        "K1": 1,
        "K2": 2,
        "Offline": None,
    }
    stats: dict[str, dict[str, float]] = {}
    for key, r in horizons_map.items():
        onset_waits: list[float] = []
        completion_waits: list[float] = []
        for rows in groups.values():
            seq_len = len(rows)
            for pos, i_idx in enumerate(rows):
                target_pos = (seq_len - 1) if r is None else min(seq_len - 1, pos + int(r))
                j_idx = rows[target_pos]
                emit_time = float(onsets_arr[j_idx]) + w_end
                onset_wait = max(0.0, emit_time - float(onsets_arr[i_idx]))
                comp_wait = max(
                    0.0, emit_time - (float(onsets_arr[i_idx]) + float(durs_arr[i_idx]))
                )
                onset_waits.append(onset_wait)
                completion_waits.append(comp_wait)
        stats[key] = {
            "window_start_s": float(window_offset),
            "window_end_s": float(w_end),
            "post_onset_mean_s": float(np.mean(onset_waits)) if onset_waits else 0.0,
            "post_onset_median_s": float(np.median(onset_waits)) if onset_waits else 0.0,
            "post_completion_mean_s": float(np.mean(completion_waits))
            if completion_waits
            else 0.0,
            "post_completion_median_s": float(np.median(completion_waits))
            if completion_waits
            else 0.0,
        }
    return {**stats, "horizons": dict(stats)}


class TimingOnlyShortcutModel(nn.Module):
    """Pure timing-grid model (zero neural input) predicting word embeddings from delta_tau."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self, timing_features: torch.Tensor, sentence_uids: Sequence[str]
    ) -> torch.Tensor:
        h = self.input_proj(timing_features.float())
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(sentence_uids):
            groups[str(uid)].append(idx)
        max_len = max(len(g) for g in groups.values())
        batch_h = h.new_zeros((len(groups), max_len, h.shape[-1]))
        pad_mask = torch.ones((len(groups), max_len), dtype=torch.bool, device=h.device)
        group_list = list(groups.values())
        for g_idx, rows in enumerate(group_list):
            batch_h[g_idx, : len(rows)] = h[rows]
            pad_mask[g_idx, : len(rows)] = False
        ctx = self.transformer(batch_h, src_key_padding_mask=pad_mask)
        out = h.new_zeros((len(h), self.out_proj.out_features))
        for g_idx, rows in enumerate(group_list):
            out[rows] = self.out_proj(ctx[g_idx, : len(rows)])
        return F.normalize(out, dim=-1, eps=1e-12)


class WordContextModel(nn.Module):
    """Word-level neural encoder + optional sentence/rolling Transformer with zero-init gated residual."""

    def __init__(
        self,
        brain_encoder: nn.Module,
        encoder_out_dim: int,
        embedding_dim: int = 1024,
        use_transformer: bool = False,
        transformer_dim: int = 256,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        *,
        gated_residual: bool = False,
        history_radius: int = 2,
    ):
        super().__init__()
        self.brain_encoder = brain_encoder
        self.word_proj = nn.Linear(encoder_out_dim, embedding_dim)
        self.use_transformer = bool(use_transformer)
        self.gated_residual = bool(gated_residual)
        self.history_radius = int(history_radius)
        if self.use_transformer:
            self.in_proj = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, transformer_dim, bias=False),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=transformer_depth, enable_nested_tensor=False
            )
            self.out_proj = nn.Linear(
                transformer_dim, embedding_dim, bias=not self.gated_residual
            )
            if self.gated_residual:
                nn.init.zeros_(self.out_proj.weight)
            self.context_alpha = nn.Parameter(torch.full((1,), 0.25))
            # Plain Linear without LayerNorm so neighbor sin(theta) norm attenuation is preserved
            self.rolling_in_proj = nn.ModuleDict(
                {
                    str(r): nn.Linear(embedding_dim, transformer_dim, bias=False)
                    for r in (0, 1, 2)
                }
            )
            self.rolling_transformers = nn.ModuleDict(
                {
                    str(r): nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=transformer_dim,
                            nhead=transformer_heads,
                            dim_feedforward=transformer_dim * 2,
                            dropout=0.1,
                            activation="gelu",
                            batch_first=True,
                            norm_first=False,
                        ),
                        num_layers=max(1, transformer_depth - 1),
                        enable_nested_tensor=False,
                    )
                    for r in (0, 1, 2)
                }
            )
            # Non-zero alpha_r with zero-initialized projection weights breaks the (alpha=0, W=0) saddle point
            self.rolling_alpha = nn.ParameterDict(
                {str(r): nn.Parameter(torch.full((1,), 0.25)) for r in (0, 1, 2)}
            )
            self.phase_alpha = nn.ParameterDict(
                {str(r): nn.Parameter(torch.full((1,), 0.20)) for r in (0, 1, 2)}
            )
            self.gate_proj = nn.ModuleDict(
                {
                    str(r): nn.Linear(embedding_dim, 1, bias=True)
                    for r in (0, 1, 2)
                }
            )
            self.phase_proj = nn.ModuleDict(
                {
                    str(r): nn.Linear(embedding_dim, embedding_dim, bias=False)
                    for r in (0, 1, 2)
                }
            )
            self.rel_pos_scale = nn.ParameterDict(
                {
                    str(r): nn.Parameter(torch.zeros(9, transformer_dim))
                    for r in (0, 1, 2)
                }
            )
            self.rel_pos_emb = nn.ParameterDict(
                {
                    str(r): nn.Parameter(torch.zeros(9, transformer_dim))
                    for r in (0, 1, 2)
                }
            )
            self.rolling_out_proj = nn.ModuleDict(
                {
                    str(r): nn.Linear(transformer_dim, embedding_dim, bias=False)
                    for r in (0, 1, 2)
                }
            )
            self.rolling_unmix_proj = nn.ModuleDict(
                {
                    str(r): nn.Linear(
                        embedding_dim * (self.history_radius + r + 1),
                        embedding_dim,
                        bias=False,
                    )
                    for r in (0, 1, 2)
                }
            )
            for r in (0, 1, 2):
                r_key = str(r)
                nn.init.zeros_(self.rolling_out_proj[r_key].weight)
                nn.init.zeros_(self.rolling_unmix_proj[r_key].weight)
                nn.init.zeros_(self.phase_proj[r_key].weight)
                nn.init.zeros_(self.gate_proj[r_key].weight)
                nn.init.constant_(self.gate_proj[r_key].bias, 1.0)
        else:
            self.in_proj = None
            self.transformer = None
            self.out_proj = None
            self.context_alpha = None
            self.rolling_in_proj = None
            self.rolling_transformers = None
            self.rolling_alpha = None
            self.phase_alpha = None
            self.gate_proj = None
            self.phase_proj = None
            self.rel_pos_scale = None
            self.rel_pos_emb = None
            self.rolling_out_proj = None
            self.rolling_unmix_proj = None

    def encode_words(
        self,
        neuro: torch.Tensor,
        subject_id: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self.brain_encoder(neuro, subject_id, channel_positions)
        return F.normalize(self.word_proj(raw.float()), dim=-1, eps=1e-12)

    def forward_context_from_embeddings(
        self,
        word_embeddings: torch.Tensor,
        sentence_uids: Sequence[str],
        *,
        return_raw_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(word_embeddings.float(), dim=-1, eps=1e-12)
        if not self.use_transformer or self.transformer is None:
            return (z, torch.zeros_like(z)) if return_raw_delta else z
        h = self.in_proj(z)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(sentence_uids):
            groups[str(uid)].append(idx)
        group_list = list(groups.values())
        max_len = max(len(g) for g in group_list)
        batch_h = h.new_zeros((len(group_list), max_len, h.shape[-1]))
        pad_mask = torch.ones((len(group_list), max_len), dtype=torch.bool, device=h.device)
        for g_idx, rows in enumerate(group_list):
            batch_h[g_idx, : len(rows)] = h[rows]
            pad_mask[g_idx, : len(rows)] = False
        ctx = self.transformer(batch_h, src_key_padding_mask=pad_mask)
        out = z.new_zeros((len(z), self.out_proj.out_features))
        for g_idx, rows in enumerate(group_list):
            out[rows] = self.out_proj(ctx[g_idx, : len(rows)])
        if self.gated_residual and self.context_alpha is not None:
            if len(out) > 1:
                out = out - out.mean(dim=0, keepdim=True)
            gate = torch.tanh(self.context_alpha)
            pred = F.normalize(z + gate * out, dim=-1, eps=1e-12)
            return (pred, out) if return_raw_delta else pred
        pred = F.normalize(out + z, dim=-1, eps=1e-12)
        return (pred, out) if return_raw_delta else pred

    def forward_rolling_from_embeddings(
        self,
        word_embeddings: torch.Tensor,
        sentence_uids: Sequence[str],
        horizons: Sequence[int] = (0, 1, 2),
        history_radius: int = 2,
        *,
        lag_embeddings: Sequence[torch.Tensor] | None = None,
        return_raw_deltas: bool = False,
    ) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Progressive local-window rolling revision (z -> K0 -> K1 -> K2) with sin(theta) ortho norm & dynamic gate."""
        z = F.normalize(word_embeddings.float(), dim=-1, eps=1e-12)
        if not self.use_transformer or self.rolling_transformers is None:
            out_dict = {int(r): z for r in horizons}
            return (out_dict, out_dict) if return_raw_deltas else out_dict

        lag_normed: list[torch.Tensor] = []
        if lag_embeddings is not None:
            if isinstance(lag_embeddings, torch.Tensor) and lag_embeddings.ndim == 3:
                lag_normed = [
                    F.normalize(lag_embeddings[:, i].to(z.device).float(), dim=-1, eps=1e-12)
                    for i in range(lag_embeddings.shape[1])
                ]
            else:
                lag_normed = [
                    F.normalize(le.to(z.device).float(), dim=-1, eps=1e-12)
                    for le in lag_embeddings
                ]

        groups: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(sentence_uids):
            groups[str(uid)].append(idx)

        req_horizons = [int(r) for r in horizons]
        max_h = min(max(req_horizons, default=0), 2)
        all_results: dict[int, torch.Tensor] = {}
        all_deltas: dict[int, torch.Tensor] = {}
        n_total, emb_dim = z.shape
        hist_r = int(history_radius)
        prev_state = z

        for r_int in range(max_h + 1):
            r_key = str(r_int)
            base_z = prev_state
            if (
                r_int < len(lag_normed)
                and self.phase_alpha is not None
                and self.phase_proj is not None
            ):
                p_gate = torch.tanh(self.phase_alpha[r_key])
                lag_vec = lag_normed[r_int]
                # Preserve natural sin(theta) orthogonal residual without F.normalize noise amplification
                lag_ortho = lag_vec - (lag_vec * base_z).sum(dim=-1, keepdim=True) * base_z
                phase_delta = self.phase_proj[r_key](lag_vec + 0.5 * lag_ortho)
            else:
                p_gate = z.new_zeros(1)
                phase_delta = torch.zeros_like(base_z)

            col_tensors: list[torch.Tensor] = []
            sin_thetas: list[torch.Tensor] = []
            scales: list[torch.Tensor] = []
            biases: list[torch.Tensor] = []
            for offset in range(-hist_r, r_int + 1):
                rel_idx = min(max(offset + 4, 0), 8)
                scales.append(1.0 + self.rel_pos_scale[r_key][rel_idx])
                biases.append(self.rel_pos_emb[r_key][rel_idx])
                if offset == 0:
                    col_tensors.append(base_z)
                else:
                    nbr_row_idx = list(range(n_total))
                    is_self = [False] * n_total
                    for rows in groups.values():
                        seq_len = len(rows)
                        for pos, target_row in enumerate(rows):
                            avail_len = min(seq_len, pos + r_int + 1)
                            if avail_len <= 1:
                                nbr_row_idx[target_row] = target_row
                                is_self[target_row] = True
                            else:
                                cand_pos = pos + offset
                                if 0 <= cand_pos < avail_len:
                                    nbr_pos = cand_pos
                                else:
                                    nbr_pos = cand_pos % avail_len
                                    if nbr_pos == pos:
                                        nbr_pos = (pos + 1) % avail_len
                                nbr_row_idx[target_row] = rows[nbr_pos]
                    nbr_idx_t = torch.tensor(nbr_row_idx, dtype=torch.long, device=z.device)
                    nbr_vecs = z[nbr_idx_t]
                    collinear = (nbr_vecs * base_z).sum(dim=-1, keepdim=True) * base_z
                    # Natural sin(theta) norm: overlapping short function words auto-attenuate noise!
                    ortho_vecs = nbr_vecs - collinear
                    if any(is_self):
                        self_mask = torch.tensor(
                            is_self, dtype=torch.bool, device=z.device
                        ).unsqueeze(-1)
                        ortho_vecs = torch.where(self_mask, torch.zeros_like(base_z), ortho_vecs)
                    sin_thetas.append(
                        torch.norm(ortho_vecs, dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
                    )
                    col_tensors.append(ortho_vecs)

            windows_emb = torch.stack(col_tensors, dim=1)
            scale_t = torch.stack(scales, dim=0).unsqueeze(0)
            bias_t = torch.stack(biases, dim=0).unsqueeze(0)
            h_win = self.rolling_in_proj[r_key](windows_emb) * scale_t + bias_t

            ctx = self.rolling_transformers[r_key](h_win)
            attn_delta = self.rolling_out_proj[r_key](ctx[:, hist_r])
            if windows_emb.shape[1] == self.history_radius + r_int + 1:
                unmix_delta = self.rolling_unmix_proj[r_key](
                    windows_emb.reshape(n_total, -1)
                )
                raw_delta = attn_delta + 0.5 * unmix_delta + p_gate * phase_delta
            else:
                raw_delta = attn_delta + p_gate * phase_delta

            if n_total > 1:
                raw_delta = raw_delta - raw_delta.mean(dim=0, keepdim=True)

            mean_sin_theta = (
                torch.stack(sin_thetas, dim=0).mean(dim=0)
                if sin_thetas
                else base_z.new_ones((n_total, 1))
            )
            subspace_gate = torch.sigmoid(self.gate_proj[r_key](base_z))
            gate = (
                (0.35 + 0.65 * mean_sin_theta)
                * subspace_gate
                * torch.tanh(self.rolling_alpha[r_key])
            )
            prev_state = F.normalize(
                base_z + gate * raw_delta, dim=-1, eps=1e-12
            )
            all_results[r_int] = prev_state
            all_deltas[r_int] = raw_delta

        results = {r: all_results[min(max(r, 0), max_h)] for r in req_horizons}
        raw_deltas = {r: all_deltas[min(max(r, 0), max_h)] for r in req_horizons}
        if return_raw_deltas:
            return results, raw_deltas
        return results

    def forward(
        self,
        neuro: torch.Tensor,
        subject_id: torch.Tensor,
        sentence_uids: Sequence[str],
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        word_emb = self.encode_words(neuro, subject_id, channel_positions)
        return self.forward_context_from_embeddings(word_emb, sentence_uids)


class _ContinuousConvBlock(nn.Module):
    """Dilated + depthwise temporal conv block matching Brain2Qwerty v2 ConvConformer."""

    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.GroupNorm(num_groups=min(8, dim), num_channels=dim),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=15,
                padding=7,
                groups=dim,
            ),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.conv(x)


class SpaceChunkedV2Model(nn.Module):
    """Brain2Qwerty v2 Space-Chunked ([tau_i, tau_{i+1}]) model for Experiment 2.

    Separates and tests the two duration-shortcut channels in space-delimited chunking:
    - Channel A (`length_aware_pool=True`): variable frame count L_i = round(delta_tau_i * fs)
      between two spaces directly exposes word duration / character count to the pooler/aligner.
    - Channel B (`continuous_encoder=True`): continuous sentence-level wide-receptive-field
      convolution + frame-level Conformer/Transformer mixes boundary timing across [tau_i, tau_{i+1}]
      before space-boundary slicing, even when intra-word pooling is a strict mean (`mean(dim=0)`).
    """

    def __init__(
        self,
        in_channels: int,
        n_subjects: int = 20,
        hidden_dim: int = 256,
        embedding_dim: int = 1024,
        *,
        continuous_encoder: bool = True,
        length_aware_pool: bool = True,
        use_aligner: bool = True,
        aligner_depth: int = 3,
        aligner_heads: int = 4,
        frequency: float = 50.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.continuous_encoder = bool(continuous_encoder)
        self.length_aware_pool = bool(length_aware_pool)
        self.use_aligner = bool(use_aligner)
        self.frequency = float(frequency)

        self.subject_emb = nn.Embedding(max(int(n_subjects) + 4, 8), hidden_dim)
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
        )

        if self.continuous_encoder:
            # Channel B: wide-receptive-field dilated conv + frame-level self-attention
            # operating across the continuous sentence BEFORE slicing at [tau_i, tau_{i+1}]
            self.conv_blocks = nn.ModuleList(
                [
                    _ContinuousConvBlock(hidden_dim, kernel_size=7, dilation=1),
                    _ContinuousConvBlock(hidden_dim, kernel_size=7, dilation=2),
                    _ContinuousConvBlock(hidden_dim, kernel_size=7, dilation=4),
                ]
            )
            frame_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=aligner_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.frame_transformer = nn.TransformerEncoder(
                frame_layer, num_layers=2, enable_nested_tensor=False
            )
            self.frame_pos_emb = nn.Parameter(
                torch.randn(1, 2048, hidden_dim) * 0.02
            )
        else:
            # Isolated per-chunk frame encoder: zero cross-boundary receptive field
            self.isolated_conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            )
            self.conv_blocks = None
            self.frame_transformer = None
            self.frame_pos_emb = None

        # Intra-word frame MLP matching brain2qwerty_v2.ctc_segmenter._IntraWordMLPPooler
        self.intra_word_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.length_aware_pool:
            # Channel A: pooler retains variable frame count L_i between spaces
            self.len_proj = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.sum_scale = nn.Parameter(torch.tensor(0.5))
        else:
            self.len_proj = None
            self.sum_scale = None

        self.word_proj = nn.Linear(hidden_dim, embedding_dim)

        if self.use_aligner:
            self.aligner_in = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, hidden_dim),
            )
            aligner_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=aligner_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.aligner = nn.TransformerEncoder(
                aligner_layer, num_layers=aligner_depth, enable_nested_tensor=False
            )
            self.aligner_out = nn.Linear(hidden_dim, embedding_dim)
        else:
            self.aligner_in = None
            self.aligner = None
            self.aligner_out = None

    def _compute_chunk_frame_lengths(
        self, delta_taus: torch.Tensor, max_frames: int
    ) -> torch.Tensor:
        frames = torch.round(delta_taus.float() * self.frequency).long()
        return torch.clamp(frames, min=2, max=max(int(max_frames), 2))

    def _pool_chunk(
        self,
        frames_hd: torch.Tensor,
        chunk_len: int,
        rel_pos: float,
    ) -> torch.Tensor:
        """Pool variable-length space-delimited word chunk (L_i, D) -> (D,)."""
        h = self.intra_word_mlp(frames_hd[:chunk_len])
        mean_vec = h.mean(dim=0)
        if not self.length_aware_pool or self.len_proj is None:
            return mean_vec
        l_f = float(chunk_len)
        len_feat = frames_hd.new_tensor(
            [l_f / 20.0, float(np.log1p(l_f)), float(rel_pos)]
        )
        sum_vec = mean_vec * (l_f / 20.0)
        return mean_vec + self.sum_scale * sum_vec + self.len_proj(len_feat)

    def encode_space_chunks(
        self,
        neuro: torch.Tensor,
        subject_id: torch.Tensor,
        delta_taus: torch.Tensor,
        sentence_uids: Sequence[str],
    ) -> torch.Tensor:
        """Encode variable-length space-delimited chunks [tau_i, tau_{i+1}] per word."""
        device = neuro.device
        n_words, _, max_frames = neuro.shape
        subj_idx = subject_id.view(-1).long().clamp(
            min=0, max=self.subject_emb.num_embeddings - 1
        )
        subj_vec = self.subject_emb(subj_idx).unsqueeze(-1)
        chunk_lens = self._compute_chunk_frame_lengths(delta_taus.to(device), max_frames)

        groups: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(sentence_uids):
            groups[str(uid)].append(idx)
        group_list = list(groups.values())

        word_vecs = neuro.new_zeros((n_words, self.hidden_dim))

        if not self.continuous_encoder:
            # Isolated per-chunk path (Channel B = OFF): mask frames strictly to [0, L_i)
            t_grid = torch.arange(max_frames, device=device).unsqueeze(0)
            valid_mask = (t_grid < chunk_lens.unsqueeze(1)).unsqueeze(1).float()
            x_masked = neuro.float() * valid_mask
            h = (self.spatial_proj(x_masked) + subj_vec) * valid_mask
            h = (h + 0.2 * self.isolated_conv(h)) * valid_mask
            h_td = h.transpose(1, 2)  # (N, T, D)
            for rows in group_list:
                seq_len = max(len(rows), 1)
                for pos, r_idx in enumerate(rows):
                    l_i = int(chunk_lens[r_idx].item())
                    word_vecs[r_idx] = self._pool_chunk(
                        h_td[r_idx], l_i, rel_pos=pos / seq_len
                    )
        else:
            # Continuous sentence path (Channel B = ON): stitch continuous sentence
            # waveform from contiguous [tau_i, tau_{i+1}] chunks, run wide Conv + Conformer,
            # then slice at exact space boundaries [s_i, e_i].
            sent_tensors: list[torch.Tensor] = []
            sent_subjs: list[torch.Tensor] = []
            sent_spans: list[list[tuple[int, int, int, float]]] = []
            max_sent_len = 0
            for rows in group_list:
                pieces: list[torch.Tensor] = []
                spans: list[tuple[int, int, int, float]] = []
                cursor = 0
                seq_len = max(len(rows), 1)
                for pos, r_idx in enumerate(rows):
                    l_i = int(chunk_lens[r_idx].item())
                    pieces.append(neuro[r_idx, :, :l_i].float())
                    spans.append((r_idx, cursor, cursor + l_i, pos / seq_len))
                    cursor += l_i
                sent_cat = torch.cat(pieces, dim=-1)
                sent_tensors.append(sent_cat)
                sent_subjs.append(subj_vec[rows[0]])
                sent_spans.append(spans)
                max_sent_len = max(max_sent_len, cursor)

            g_count = len(group_list)
            padded_sent = neuro.new_zeros((g_count, self.in_channels, max_sent_len))
            sent_pad_mask = torch.ones(
                (g_count, max_sent_len), dtype=torch.bool, device=device
            )
            subj_sent = torch.stack(sent_subjs, dim=0)
            for g_idx, s_t in enumerate(sent_tensors):
                t_len = s_t.shape[-1]
                padded_sent[g_idx, :, :t_len] = s_t
                sent_pad_mask[g_idx, :t_len] = False

            h_sent = self.spatial_proj(padded_sent) + subj_sent
            for block in self.conv_blocks:
                h_sent = block(h_sent)
            h_frames = h_sent.transpose(1, 2)  # (G, T_sent, D)
            pos_len = min(max_sent_len, self.frame_pos_emb.shape[1])
            pos_pad = torch.zeros_like(h_frames)
            pos_pad[:, :pos_len] = self.frame_pos_emb[:, :pos_len]
            h_frames = h_frames + pos_pad
            z_final = self.frame_transformer(
                h_frames, src_key_padding_mask=sent_pad_mask
            )

            # CTCSpaceSegmenter-style slicing by space boundaries [s_i, e_i]
            for g_idx, spans in enumerate(sent_spans):
                for r_idx, s_i, e_i, rel_pos in spans:
                    chunk_frames = z_final[g_idx, s_i:e_i]
                    word_vecs[r_idx] = self._pool_chunk(
                        chunk_frames, e_i - s_i, rel_pos=rel_pos
                    )

        return F.normalize(self.word_proj(word_vecs), dim=-1, eps=1e-12)

    def forward_aligner_from_embeddings(
        self,
        word_embeddings: torch.Tensor,
        sentence_uids: Sequence[str],
    ) -> torch.Tensor:
        if not self.use_aligner or self.aligner is None:
            return F.normalize(word_embeddings.float(), dim=-1, eps=1e-12)
        h = self.aligner_in(word_embeddings.float())
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(sentence_uids):
            groups[str(uid)].append(idx)
        group_list = list(groups.values())
        max_len = max(len(g) for g in group_list)
        batch_h = h.new_zeros((len(group_list), max_len, h.shape[-1]))
        pad_mask = torch.ones(
            (len(group_list), max_len), dtype=torch.bool, device=h.device
        )
        for g_idx, rows in enumerate(group_list):
            batch_h[g_idx, : len(rows)] = h[rows]
            pad_mask[g_idx, : len(rows)] = False
        ctx = self.aligner(batch_h, src_key_padding_mask=pad_mask)
        out = word_embeddings.new_zeros((len(word_embeddings), self.embedding_dim))
        for g_idx, rows in enumerate(group_list):
            out[rows] = self.aligner_out(ctx[g_idx, : len(rows)])
        return F.normalize(out + word_embeddings.float(), dim=-1, eps=1e-12)

    def forward(
        self,
        neuro: torch.Tensor,
        subject_id: torch.Tensor,
        delta_taus: torch.Tensor,
        sentence_uids: Sequence[str],
    ) -> torch.Tensor:
        chunk_embs = self.encode_space_chunks(
            neuro, subject_id, delta_taus, sentence_uids
        )
        return self.forward_aligner_from_embeddings(chunk_embs, sentence_uids)

