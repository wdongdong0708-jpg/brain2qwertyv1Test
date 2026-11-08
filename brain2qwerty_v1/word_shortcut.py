from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

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
    """Deterministic character n-gram embedding fallback for offline unit tests."""
    import hashlib

    vec = np.zeros(dim, dtype=np.float32)
    padded = f"<{word.lower()}>"
    grams = [padded[i : i + n] for n in (1, 2, 3) for i in range(len(padded) - n + 1)]
    for gram in grams or [word]:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec += rng.standard_normal(dim, dtype=np.float32)
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
        mat = np.stack([_hash_word_embedding(w, embedding_dim) for w in unique_words])
        return unique_words, mat.astype(np.float32)

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

    matrix = np.stack([cached_map[w] for w in unique_words]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return unique_words, matrix


def build_vocabulary_banks(
    word_events: pd.DataFrame,
    vocab_words: Sequence[str],
    top_k_50: int = 50,
) -> dict:
    """Build the frozen 50-word diagnostic bank (V_50) and full training vocabulary (V_full)."""
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    train_df = word_events[word_events["split"] == "train"]
    if train_df.empty:
        train_df = word_events
    counts = Counter(train_df["text"].astype(str).tolist())
    ranked_train_words = [
        w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if w in word_to_idx
    ]
    vocab50_words = ranked_train_words[: min(int(top_k_50), len(ranked_train_words))]
    vocab50_indices = np.asarray([word_to_idx[w] for w in vocab50_words], dtype=np.int64)
    full_vocab_indices = np.asarray(
        [word_to_idx[w] for w in sorted(counts.keys()) if w in word_to_idx],
        dtype=np.int64,
    )
    return {
        "word_to_idx": word_to_idx,
        "vocab50_words": vocab50_words,
        "vocab50_indices": vocab50_indices,
        "full_vocab_words": [vocab_words[i] for i in full_vocab_indices],
        "full_vocab_indices": full_vocab_indices,
        "train_word_counts": dict(counts),
    }


def apply_leave_one_out_cms(
    clean_predictions: torch.Tensor,
    shifted_predictions: dict[float, torch.Tensor],
    eta: float = 0.50,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Leave-One-Out Common-Mode Subtraction (LOO-CMS) on unit-normalized predictions.

    - Clean (0s) subtracts the mean of all M historical shifted predictions.
    - Each Shifted (delta_k) subtracts only the mean of the other M-1 shifted predictions,
      guaranteeing zero self-subtraction leakage on the negative control.
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
    debiased_clean = F.normalize(
        clean_norm - float(eta) * (sum_all / m), dim=-1, eps=1e-12
    )
    debiased_shifted: dict[float, torch.Tensor] = {}
    for k in shift_keys:
        mean_other = (sum_all - shifted_norm[k]) / max(m - 1.0, 1.0)
        debiased_shifted[k] = F.normalize(
            shifted_norm[k] - float(eta) * mean_other, dim=-1, eps=1e-12
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
        for t_id, h10, h1 in zip(q_targets.tolist(), hit10.tolist(), hit1.tolist()):
            if int(t_id) in cand_pos_map:
                by_type10[int(t_id)].append(h10)
                by_type1[int(t_id)].append(h1)

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
        },
        "net_gain": {
            "vocab50_macro_top10": c_v50["macro_top10"] - v50_macro10_mean,
            "vocab50_micro_top10": c_v50["micro_top10"] - v50_micro10_mean,
            "vocab50_macro_top1": c_v50["macro_top1"] - v50_macro1_mean,
            "full_vocab_macro_top10": c_fv["macro_top10"] - fv_macro10_mean,
            "full_vocab_micro_top10": c_fv["micro_top10"] - fv_micro10_mean,
            "function_top10": c_v50["function_top10"] - func10_mean,
            "content_top10": c_v50["content_top10"] - cont10_mean,
        },
        "shortcut_inflation_over_chance_v50_top10": v50_macro10_mean
        - c_v50["chance_top10"],
    }


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
    """Word-level neural encoder + optional sentence/rolling Transformer."""

    def __init__(
        self,
        brain_encoder: nn.Module,
        encoder_out_dim: int,
        embedding_dim: int = 1024,
        use_transformer: bool = False,
        transformer_dim: int = 512,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
    ):
        super().__init__()
        self.brain_encoder = brain_encoder
        self.word_proj = nn.Linear(encoder_out_dim, embedding_dim)
        self.use_transformer = bool(use_transformer)
        if self.use_transformer:
            self.in_proj = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, transformer_dim),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=transformer_depth, enable_nested_tensor=False
            )
            self.out_proj = nn.Linear(transformer_dim, embedding_dim)
        else:
            self.in_proj = None
            self.transformer = None
            self.out_proj = None

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
    ) -> torch.Tensor:
        if not self.use_transformer or self.transformer is None:
            return F.normalize(word_embeddings.float(), dim=-1, eps=1e-12)
        h = self.in_proj(word_embeddings.float())
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
        out = word_embeddings.new_zeros((len(word_embeddings), self.out_proj.out_features))
        for g_idx, rows in enumerate(group_list):
            out[rows] = self.out_proj(ctx[g_idx, : len(rows)])
        return F.normalize(out + word_embeddings.float(), dim=-1, eps=1e-12)

    def forward_rolling_from_embeddings(
        self,
        word_embeddings: torch.Tensor,
        sentence_uids: Sequence[str],
        horizons: Sequence[int] = (0, 1, 2),
        history_radius: int = 2,
    ) -> dict[int, torch.Tensor]:
        """Single-pass local-window rolling revision (K0, K1, K2) on frozen word embeddings."""
        if not self.use_transformer or self.transformer is None:
            normed = F.normalize(word_embeddings.float(), dim=-1, eps=1e-12)
            return {int(r): normed for r in horizons}

        h = self.in_proj(word_embeddings.float())
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(sentence_uids):
            groups[str(uid)].append(idx)

        results: dict[int, torch.Tensor] = {}
        n_total = len(word_embeddings)
        width_max = int(history_radius) + max(int(r) for r in horizons) + 1

        for r in horizons:
            r_int = int(r)
            width = int(history_radius) + r_int + 1
            windows = h.new_zeros((n_total, width, h.shape[-1]))
            pad_mask = torch.ones((n_total, width), dtype=torch.bool, device=h.device)
            for rows in groups.values():
                seq_len = len(rows)
                for pos, target_row in enumerate(rows):
                    eff_right = min(r_int, seq_len - 1 - pos)
                    for col, offset in enumerate(range(-int(history_radius), eff_right + 1)):
                        nbr = pos + offset
                        if 0 <= nbr < seq_len:
                            windows[target_row, col] = h[rows[nbr]]
                            pad_mask[target_row, col] = False
            ctx = self.transformer(windows, src_key_padding_mask=pad_mask)
            delta = self.out_proj(ctx[:, int(history_radius)])
            results[r_int] = F.normalize(
                word_embeddings.float() + delta, dim=-1, eps=1e-12
            )
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
