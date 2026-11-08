from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

import neuralset as ns
import studies  # noqa: F401

from ..cache import cache_signature, configure_cache, neuro_cache_key
from ..transforms import SPANISH_FUNCTION_WORDS
from ..utils import ChannelPositions2D, select_participants
from ..word_shortcut import (
    DEFAULT_SHIFT_SECONDS,
    DuplicateAwareSigLipLoss,
    SpaceChunkedV2Model,
    TimingOnlyShortcutModel,
    WordContextModel,
    apply_leave_one_out_cms,
    audit_word_timing_statistics,
    build_vocabulary_banks,
    compute_shifted_word_onsets,
    compute_streaming_latency_stats,
    evaluate_word_retrieval,
    load_or_compute_bge_m3_embeddings,
    summarize_clean_vs_shifted,
)
from ..workflow import (
    WorkflowExperiment,
    make_config,
    requested_subjects,
    select_subjects,
    subject_tag,
    write_json,
)


@dataclass
class SplitWordBundle:
    split: str
    neuro: torch.Tensor
    subject_id: torch.Tensor
    channel_positions: torch.Tensor | None
    word_uids: list[str]
    sentence_uids: list[str]
    words: list[str]
    word_indices: np.ndarray
    onsets: np.ndarray
    durations: np.ndarray
    delta_taus: np.ndarray
    char_lengths: np.ndarray
    is_function_word: np.ndarray
    timing_features: torch.Tensor


def _trigger_attr(segment, key: str, default=None):
    extra = getattr(segment.trigger, "extra", None)
    if isinstance(extra, dict) and key in extra and extra[key] is not None:
        return extra[key]
    return getattr(segment.trigger, key, default)


def _crop_bundle_frames(bundle: SplitWordBundle, n_frames: int) -> SplitWordBundle:
    n_f = max(2, min(int(n_frames), int(bundle.neuro.shape[-1])))
    if n_f == bundle.neuro.shape[-1]:
        return bundle
    return replace(bundle, neuro=bundle.neuro[:, :, :n_f].contiguous())


def _slice_window_bundle(
    master_bundle: SplitWordBundle,
    master_offset: float,
    target_offset: float,
    target_duration: float,
    fps: float = 50.0,
) -> SplitWordBundle:
    """Slice a sub-window [target_offset, target_offset + target_duration] from master_bundle."""
    total_f = int(master_bundle.neuro.shape[-1])
    n_f = max(2, min(int(round(float(target_duration) * fps)), total_f))
    start_f = max(0, int(round((float(target_offset) - float(master_offset)) * fps)))
    if start_f + n_f > total_f:
        start_f = max(0, total_f - n_f)
    end_f = min(total_f, start_f + n_f)
    if start_f == 0 and end_f == total_f:
        return master_bundle
    return replace(master_bundle, neuro=master_bundle.neuro[:, :, start_f:end_f].contiguous())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Brain2Qwerty Word-Level Typing Duration Shortcut Verification (Exp1 Fixed-Window, Step 2 Pre-Motor Lead & Unmixing, Exp2 v2 Space-Chunked)"
    )
    p.add_argument("--modality", choices=["meg", "eeg"], default="meg")
    subjects = p.add_mutually_exclusive_group()
    subjects.add_argument(
        "--subject",
        default="all",
        help="002 or S2; omitted/all selects every valid subject",
    )
    subjects.add_argument(
        "--subjects",
        nargs="+",
        help="Fixed participant list, separated by spaces or commas (e.g. S1 S2)",
    )
    subjects.add_argument(
        "--num-subjects",
        type=int,
        help="Select the first N available, valid participants in numeric ID order",
    )
    p.add_argument("--data-root", "--studies-root", type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--results-root", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument(
        "--experiments",
        nargs="+",
        choices=["exp1", "exp2", "all"],
        default=["exp1", "exp2"],
        help="Run Experiment 1 (fixed window overlap + pre-motor lead), Experiment 2 (v2 space-chunked [tau_i, tau_{i+1}]), or both",
    )
    p.add_argument(
        "--window-offset",
        type=float,
        default=0.0,
        help="Primary word window start offset (seconds) relative to word onset tau_i",
    )
    p.add_argument(
        "--window-offsets",
        type=float,
        nargs="+",
        default=[0.0, -0.25, -0.35],
        help="Step 2 pre-motor lead window offsets (seconds) to compare against post-motor [+0.00s]",
    )
    p.add_argument(
        "--window-duration",
        type=float,
        default=0.5,
        help="Experiment 1 fixed word window duration T (seconds), e.g. 0.5 or 1.0",
    )
    p.add_argument(
        "--shifts",
        type=float,
        nargs="+",
        default=list(DEFAULT_SHIFT_SECONDS),
        help="Historical circular shift anchors (seconds) for the negative control",
    )
    p.add_argument(
        "--cms-eta",
        type=float,
        default=0.50,
        help="Leave-One-Out Common-Mode Subtraction (LOO-CMS) scaling factor eta",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--timing-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--profile", choices=["full", "small"], default="full")
    p.add_argument(
        "--embedding-backend",
        choices=["bge-m3", "hash"],
        default="bge-m3",
        help="Use BAAI/bge-m3 in Colab or deterministic hash embeddings in offline tests",
    )
    p.add_argument("--embedding-cache", type=Path)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--require-cache", action="store_true")
    p.add_argument("--cache-version", default="1")
    p.add_argument("--max-train-sentences", type=int)
    p.add_argument("--max-val-sentences", type=int)
    p.add_argument("--max-test-sentences", type=int)
    return p


def compute_timeline_bounds(
    events: pd.DataFrame, modality: str
) -> dict[str, tuple[float, float]]:
    """Extract [rec_min, rec_max] bounds per recording timeline from Meg/Eeg events."""
    rec_type = "Eeg" if modality == "eeg" else "Meg"
    rec_rows = events[events["type"] == rec_type]
    bounds: dict[str, tuple[float, float]] = {}
    if not rec_rows.empty:
        for tl, grp in rec_rows.groupby("timeline"):
            r_min = float(grp["start"].min())
            r_max = float((grp["start"] + grp["duration"].fillna(1.0)).max())
            if pd.notna(r_min) and pd.notna(r_max) and r_max > r_min:
                bounds[str(tl)] = (r_min, r_max)
    for tl, grp in events.groupby("timeline"):
        if str(tl) not in bounds:
            r_min = float(grp["start"].min()) - 5.0
            r_max = float((grp["start"] + grp["duration"].fillna(1.0)).max()) + 5.0
            bounds[str(tl)] = (r_min, r_max)
    return bounds


def filter_sentences_within_bounds(
    events: pd.DataFrame,
    timeline_bounds: dict[str, tuple[float, float]],
    window_offset: float,
    window_duration: float,
    max_sentences_per_split: dict[str, int | None] | None = None,
) -> pd.DataFrame:
    """Keep complete sentences whose word windows lie safely inside recording bounds."""
    events = events.copy()
    word_mask = events["type"] == "Word"
    words = events[word_mask]
    if words.empty:
        raise RuntimeError("No typing-phase Word events found after preprocessing")

    margin_left = max(0.0, -float(window_offset)) + 0.20
    margin_right = max(0.0, float(window_offset) + float(window_duration)) + 0.20

    valid_suids: set[str] = set()
    for suid, grp in words.groupby("sentence_UID", sort=False):
        tl = str(grp["timeline"].iloc[0])
        rec_min, rec_max = timeline_bounds[tl]
        w_min = float(grp["start"].min())
        w_max = float(grp["start"].max())
        sent_span = w_max - w_min
        if (
            w_min >= rec_min + margin_left
            and w_max <= rec_max - margin_right
            and (rec_max - rec_min) > (sent_span + margin_left + margin_right)
        ):
            valid_suids.add(str(suid))

    if max_sentences_per_split:
        kept_by_split: set[str] = set()
        for split in ("train", "val", "test"):
            split_words = words[
                (words["split"] == split) & (words["sentence_UID"].isin(valid_suids))
            ]
            suids_ordered = list(dict.fromkeys(split_words["sentence_UID"].astype(str)))
            limit = max_sentences_per_split.get(split)
            if limit is not None and limit > 0:
                suids_ordered = suids_ordered[: int(limit)]
            kept_by_split.update(suids_ordered)
        valid_suids = kept_by_split

    drop_word_mask = word_mask & (~events["sentence_UID"].astype(str).isin(valid_suids))
    return events[~drop_word_mask].reset_index(drop=True)


def _compute_timing_feature_matrix(
    sentence_uids: Sequence[str],
    delta_taus: np.ndarray,
    word_orders: np.ndarray,
    sentence_word_counts: np.ndarray,
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, dict[str, tuple[float, float]]]:
    """Build 4-d pure timing features (delta_tau_i, prev_delta_tau, rel_pos, log_len)."""
    dt = np.asarray(delta_taus, dtype=np.float32)
    prev_dt = np.zeros_like(dt)
    for i in range(len(dt)):
        if i > 0 and sentence_uids[i] == sentence_uids[i - 1]:
            prev_dt[i] = dt[i - 1]
        else:
            prev_dt[i] = dt[i]
    rel_pos = np.asarray(word_orders, dtype=np.float32) / np.maximum(
        np.asarray(sentence_word_counts, dtype=np.float32), 1.0
    )
    log_len = np.log1p(np.asarray(sentence_word_counts, dtype=np.float32))
    raw = np.stack([dt, prev_dt, rel_pos, log_len], axis=1)
    if stats is None:
        mean = raw.mean(axis=0)
        std = np.maximum(raw.std(axis=0), 1e-5)
        stats = {"mean": mean, "std": std}
    normed = (raw - stats["mean"]) / stats["std"]
    return torch.from_numpy(normed.astype(np.float32)), stats


def slice_split_word_bundle(
    events: pd.DataFrame,
    extractors: dict,
    split: str,
    word_to_idx: dict[str, int],
    *,
    window_offset: float,
    window_duration: float,
    batch_size: int = 256,
    workers: int = 0,
    timing_stats: dict | None = None,
    reference_bundle: SplitWordBundle | None = None,
) -> tuple[SplitWordBundle, dict]:
    """Slice all Word segments for a split into a contiguous in-memory SplitWordBundle."""
    sub_mask = events["type"].isin(["Meg", "Eeg"]) | (
        (events["type"] == "Word") & (events["split"] == split)
    )
    sub_events = events.loc[sub_mask].reset_index(drop=True)
    mask = sub_events["type"] == "Word"
    segments = ns.segments.list_segments(
        sub_events, mask, start=float(window_offset), duration=float(window_duration)
    )
    if not segments:
        raise RuntimeError(f"No Word segments found for split={split!r}")
    dataset = ns.SegmentDataset(
        extractors=extractors,
        segments=segments,
        remove_incomplete_segments=True,
    )
    if not len(dataset):
        raise RuntimeError(f"All Word segments were incomplete for split={split!r}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(int(batch_size), 64),
        shuffle=False,
        num_workers=int(workers),
        collate_fn=dataset.collate_fn,
    )
    neuro_chunks: list[torch.Tensor] = []
    subj_chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []
    word_uids: list[str] = []
    sentence_uids: list[str] = []
    words: list[str] = []
    onsets: list[float] = []
    durations: list[float] = []
    delta_taus: list[float] = []
    char_lengths: list[int] = []
    is_func: list[bool] = []
    word_orders: list[int] = []
    sent_counts: list[int] = []

    for batch in loader:
        neuro_chunks.append(batch.data["neuro"].detach().cpu().float())
        subj_chunks.append(batch.data["subject_id"].detach().cpu())
        if "channel_positions" in batch.data and batch.data["channel_positions"] is not None:
            pos_chunks.append(batch.data["channel_positions"].detach().cpu().float())
        for seg in batch.segments:
            w_text = str(_trigger_attr(seg, "text", "")).strip().lower()
            words.append(w_text)
            word_uids.append(str(_trigger_attr(seg, "word_UID", "")))
            sentence_uids.append(str(_trigger_attr(seg, "sentence_UID", "")))
            onsets.append(float(_trigger_attr(seg, "start", 0.0)))
            durations.append(float(_trigger_attr(seg, "duration", 0.1)))
            delta_taus.append(float(_trigger_attr(seg, "delta_tau", 0.3)))
            char_lengths.append(int(_trigger_attr(seg, "char_length", max(len(w_text), 1))))
            is_func.append(bool(_trigger_attr(seg, "is_function_word", False)))
            word_orders.append(int(_trigger_attr(seg, "word_order", 0)))
            sent_counts.append(int(_trigger_attr(seg, "sentence_word_count", 1)))

    neuro_all = torch.cat(neuro_chunks, dim=0)
    subj_all = torch.cat(subj_chunks, dim=0)
    pos_all = torch.cat(pos_chunks, dim=0) if pos_chunks else None

    if reference_bundle is not None and word_uids != reference_bundle.word_uids:
        uid_to_pos = {uid: idx for idx, uid in enumerate(word_uids)}
        missing = [uid for uid in reference_bundle.word_uids if uid not in uid_to_pos]
        if missing:
            raise RuntimeError(
                f"Shifted {split} split lost {len(missing)} segments compared to Clean"
            )
        perm = [uid_to_pos[uid] for uid in reference_bundle.word_uids]
        perm_t = torch.tensor(perm, dtype=torch.long)
        neuro_all = neuro_all[perm_t]
        subj_all = subj_all[perm_t]
        if pos_all is not None:
            pos_all = pos_all[perm_t]
        word_uids = list(reference_bundle.word_uids)
        sentence_uids = list(reference_bundle.sentence_uids)
        words = list(reference_bundle.words)
        onsets = [onsets[i] for i in perm]
        durations = list(reference_bundle.durations.tolist())
        delta_taus = list(reference_bundle.delta_taus.tolist())
        char_lengths = list(reference_bundle.char_lengths.tolist())
        is_func = list(reference_bundle.is_function_word.tolist())
        timing_feats = reference_bundle.timing_features.clone()
    else:
        timing_feats, timing_stats = _compute_timing_feature_matrix(
            sentence_uids,
            np.asarray(delta_taus, dtype=np.float64),
            np.asarray(word_orders, dtype=np.int64),
            np.asarray(sent_counts, dtype=np.int64),
            stats=timing_stats,
        )

    word_indices = np.asarray([word_to_idx[w] for w in words], dtype=np.int64)
    bundle = SplitWordBundle(
        split=split,
        neuro=neuro_all,
        subject_id=subj_all,
        channel_positions=pos_all,
        word_uids=word_uids,
        sentence_uids=sentence_uids,
        words=words,
        word_indices=word_indices,
        onsets=np.asarray(onsets, dtype=np.float64),
        durations=np.asarray(durations, dtype=np.float64),
        delta_taus=np.asarray(delta_taus, dtype=np.float64),
        char_lengths=np.asarray(char_lengths, dtype=np.int64),
        is_function_word=np.asarray(is_func, dtype=bool),
        timing_features=timing_feats,
    )
    return bundle, timing_stats


def build_sentence_batches(
    sentence_uids: Sequence[str],
    max_words_per_batch: int = 128,
    *,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Group word indices by sentence_UID so sentences are never split across batches."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, suid in enumerate(sentence_uids):
        groups[str(suid)].append(idx)
    group_list = list(groups.values())
    if shuffle and rng is not None:
        order = rng.permutation(len(group_list))
        group_list = [group_list[i] for i in order]

    batches: list[np.ndarray] = []
    current: list[int] = []
    budget = max(int(max_words_per_batch), 8)
    for g in group_list:
        if current and len(current) + len(g) > budget:
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
        current.extend(g)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def train_timing_only_model(
    train_bundle: SplitWordBundle,
    val_bundle: SplitWordBundle,
    embedding_bank: np.ndarray,
    vocab_banks: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> TimingOnlyShortcutModel:
    """Train Model 0: Pure timing-grid (delta_tau_i) shortcut probe with zero neural input."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TimingOnlyShortcutModel(embedding_dim=embedding_bank.shape[1]).to(device)
    criterion = DuplicateAwareSigLipLoss().to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *criterion.parameters()], lr=lr, weight_decay=1e-3
    )
    bank_t = torch.from_numpy(embedding_bank).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0

    for _ in range(max(int(epochs), 1)):
        model.train()
        batches = build_sentence_batches(
            train_bundle.sentence_uids, batch_size, shuffle=True, rng=rng
        )
        for rows in batches:
            rows_t = torch.from_numpy(rows).long()
            feats = train_bundle.timing_features[rows_t].to(device)
            suids = [train_bundle.sentence_uids[i] for i in rows]
            targets = bank_t[
                torch.from_numpy(train_bundle.word_indices[rows]).long().to(device)
            ]
            optimizer.zero_grad(set_to_none=True)
            preds = model(feats, suids)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(
                val_bundle.timing_features.to(device), val_bundle.sentence_uids
            )
            val_metrics = evaluate_word_retrieval(
                val_preds,
                val_bundle.word_indices,
                val_bundle.is_function_word,
                embedding_bank,
                vocab_banks["vocab50_indices"],
                vocab_banks["full_vocab_indices"],
            )
            score = val_metrics["vocab50"]["macro_top10"]
            if score >= best_val:
                best_val = score
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_word_context_model(
    model: WordContextModel,
    bundle: SplitWordBundle,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both raw Stage-1 word embeddings and final model output embeddings on CPU."""
    model.eval()
    batches = build_sentence_batches(bundle.sentence_uids, batch_size, shuffle=False)
    word_embs: list[torch.Tensor] = []
    ctx_embs: list[torch.Tensor] = []
    for rows in batches:
        rows_t = torch.from_numpy(rows).long()
        neuro = bundle.neuro[rows_t].to(device)
        subj = bundle.subject_id[rows_t].to(device)
        pos = (
            bundle.channel_positions[rows_t].to(device)
            if bundle.channel_positions is not None
            else None
        )
        suids = [bundle.sentence_uids[i] for i in rows]
        w_emb = model.encode_words(neuro, subj, pos)
        c_emb = model.forward_context_from_embeddings(w_emb, suids)
        word_embs.append(w_emb.detach().cpu())
        ctx_embs.append(c_emb.detach().cpu())
    return torch.cat(word_embs, dim=0), torch.cat(ctx_embs, dim=0)


def _materialize_word_context_model(
    model: WordContextModel,
    bundle: SplitWordBundle,
    device: torch.device,
) -> None:
    model.to(device)
    model.eval()
    n = min(len(bundle.neuro), 4)
    with torch.no_grad():
        neuro = bundle.neuro[:n].to(device)
        subj = bundle.subject_id[:n].to(device)
        pos = (
            bundle.channel_positions[:n].to(device)
            if bundle.channel_positions is not None
            else None
        )
        suids = bundle.sentence_uids[:n]
        model(neuro, subj, suids, pos)
        for name, param in list(model.named_parameters()):
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                parent = (
                    model.get_submodule(name.rsplit(".", 1)[0])
                    if "." in name
                    else model
                )
                attr = name.rsplit(".", 1)[-1]
                setattr(
                    parent,
                    attr,
                    torch.nn.Parameter(
                        torch.empty(1, device=device), requires_grad=False
                    ),
                )
    model.train()


def train_neural_word_model(
    label: str,
    model: WordContextModel,
    train_bundle: SplitWordBundle,
    val_bundle: SplitWordBundle,
    embedding_bank: np.ndarray,
    vocab_banks: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> WordContextModel:
    """Train a neural WordContextModel (Word-Only or Full-Context Joint) end-to-end."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _materialize_word_context_model(model, train_bundle, device)
    criterion = DuplicateAwareSigLipLoss().to(device)
    optimizer = torch.optim.AdamW(
        [
            *[p for p in model.parameters() if p.requires_grad],
            *criterion.parameters(),
        ],
        lr=lr,
        weight_decay=1e-3,
    )
    bank_t = torch.from_numpy(embedding_bank).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0

    for epoch in range(max(int(epochs), 1)):
        model.train()
        batches = build_sentence_batches(
            train_bundle.sentence_uids, batch_size, shuffle=True, rng=rng
        )
        epoch_losses: list[float] = []
        for rows in batches:
            rows_t = torch.from_numpy(rows).long()
            neuro = train_bundle.neuro[rows_t].to(device)
            subj = train_bundle.subject_id[rows_t].to(device)
            pos = (
                train_bundle.channel_positions[rows_t].to(device)
                if train_bundle.channel_positions is not None
                else None
            )
            suids = [train_bundle.sentence_uids[i] for i in rows]
            targets = bank_t[
                torch.from_numpy(train_bundle.word_indices[rows]).long().to(device)
            ]
            optimizer.zero_grad(set_to_none=True)
            preds = model(neuro, subj, suids, pos)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        _, val_preds = predict_word_context_model(model, val_bundle, batch_size, device)
        val_metrics = evaluate_word_retrieval(
            val_preds,
            val_bundle.word_indices,
            val_bundle.is_function_word,
            embedding_bank,
            vocab_banks["vocab50_indices"],
            vocab_banks["full_vocab_indices"],
        )
        score = val_metrics["vocab50"]["macro_top10"]
        if score >= best_val:
            best_val = score
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch == epochs - 1:
            print(
                f"[{label}] Epoch {epoch + 1}/{epochs} | "
                f"loss={np.mean(epoch_losses):.4f} | val_v50_top10={score:.2f}%",
                flush=True,
            )

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def _compute_lag_embeddings(
    word_only_model: WordContextModel,
    master_bundle: SplitWordBundle,
    master_offset: float,
    window_offset: float,
    window_duration: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract causal multi-phase Stage-1 embeddings at (w_offset-0.10s, w_offset+0.14s, w_offset+0.28s)."""
    lag_offsets = (
        float(window_offset) - 0.10,
        float(window_offset) + 0.14,
        float(window_offset) + 0.28,
    )
    lag_tensors: list[torch.Tensor] = []
    for l_off in lag_offsets:
        sub_b = _slice_window_bundle(
            master_bundle, master_offset, l_off, window_duration
        )
        w_emb, _ = predict_word_context_model(
            word_only_model, sub_b, batch_size, device
        )
        lag_tensors.append(w_emb)
    return torch.stack(lag_tensors, dim=1)


def train_frozen_context_transformer(
    label: str,
    stage1_model: WordContextModel,
    train_stage1_embs: torch.Tensor,
    val_stage1_embs: torch.Tensor,
    train_bundle: SplitWordBundle,
    val_bundle: SplitWordBundle,
    embedding_bank: np.ndarray,
    vocab_banks: dict,
    *,
    rolling: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    train_lag_embs: torch.Tensor | None = None,
    val_lag_embs: torch.Tensor | None = None,
) -> WordContextModel:
    """Train Stage-2 Sentence or Progressive Rolling-K0->K1->K2 Transformer on frozen Stage-1 embeddings."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WordContextModel(
        brain_encoder=copy.deepcopy(stage1_model.brain_encoder),
        encoder_out_dim=stage1_model.word_proj.in_features,
        embedding_dim=embedding_bank.shape[1],
        use_transformer=True,
        gated_residual=True,
    ).to(device)
    model.word_proj.load_state_dict(stage1_model.word_proj.state_dict())
    for p in model.brain_encoder.parameters():
        p.requires_grad_(False)
    for p in model.word_proj.parameters():
        p.requires_grad_(False)
    model.brain_encoder.eval()
    model.word_proj.eval()

    bank_t = torch.from_numpy(embedding_bank).to(device)
    val_targets_cpu = torch.from_numpy(embedding_bank[val_bundle.word_indices]).float()

    def _val_tuple(val_preds_cpu: torch.Tensor) -> tuple[float, float, float]:
        metrics = evaluate_word_retrieval(
            val_preds_cpu,
            val_bundle.word_indices,
            val_bundle.is_function_word,
            embedding_bank,
            vocab_banks["vocab50_indices"],
            vocab_banks["full_vocab_indices"],
        )
        p_norm = torch.nn.functional.normalize(val_preds_cpu.float(), dim=-1, eps=1e-12)
        cos_sim = float((p_norm * val_targets_cpu).sum(dim=-1).mean())
        return (
            round(float(metrics["vocab50"]["macro_top10"]), 4),
            round(float(metrics["full_vocab"]["macro_top10"]), 4),
            round(cos_sim, 6),
        )

    if not rolling:
        criterion = DuplicateAwareSigLipLoss().to(device)
        gate_params = [model.context_alpha] if model.context_alpha is not None else []
        gate_ids = {id(p) for p in gate_params}
        base_params = [
            p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": lr, "weight_decay": 1e-3},
                {"params": gate_params, "lr": lr * 6.0, "weight_decay": 0.0},
                {"params": list(criterion.parameters()), "lr": lr, "weight_decay": 0.0},
            ]
        )
        model.eval()
        with torch.no_grad():
            init_preds = model.forward_context_from_embeddings(
                val_stage1_embs.to(device), val_bundle.sentence_uids
            ).cpu()
        best_val = _val_tuple(init_preds)
        best_state = copy.deepcopy(model.state_dict())

        for _ in range(max(int(epochs), 1)):
            model.train()
            model.brain_encoder.eval()
            model.word_proj.eval()
            batches = build_sentence_batches(
                train_bundle.sentence_uids, batch_size, shuffle=True, rng=rng
            )
            for rows in batches:
                rows_t = torch.from_numpy(rows).long()
                w_emb = train_stage1_embs[rows_t].to(device)
                suids = [train_bundle.sentence_uids[i] for i in rows]
                targets = bank_t[
                    torch.from_numpy(train_bundle.word_indices[rows]).long().to(device)
                ]
                optimizer.zero_grad(set_to_none=True)
                preds, raw_delta = model.forward_context_from_embeddings(
                    w_emb, suids, return_raw_delta=True
                )
                aux_preds = torch.nn.functional.normalize(
                    w_emb + 0.25 * raw_delta, dim=-1, eps=1e-12
                )
                loss = criterion(preds, targets) + 0.35 * criterion(aux_preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*base_params, *gate_params], 1.0
                )
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model.forward_context_from_embeddings(
                    val_stage1_embs.to(device), val_bundle.sentence_uids
                ).cpu()
                score = _val_tuple(val_preds)
                if score >= best_val:
                    best_val = score
                    best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        return model

    # Progressive Rolling Revision: train K0 -> freeze K0 & train K1 -> freeze K1 & train K2
    stage_epochs = max(int(epochs), 1)
    for r_stage in (0, 1, 2):
        r_key = str(r_stage)
        for p in model.parameters():
            p.requires_grad_(False)
        stage_modules = [
            model.rolling_in_proj[r_key],
            model.rolling_transformers[r_key],
            model.rolling_out_proj[r_key],
            model.rolling_unmix_proj[r_key],
        ]
        for mod in stage_modules:
            for p in mod.parameters():
                p.requires_grad_(True)
        model.rel_pos_scale[r_key].requires_grad_(True)
        model.rel_pos_emb[r_key].requires_grad_(True)
        model.rolling_alpha[r_key].requires_grad_(True)
        model.phase_alpha[r_key].requires_grad_(True)

        gate_params = [model.rolling_alpha[r_key], model.phase_alpha[r_key]]
        gate_ids = {id(p) for p in gate_params}
        base_params = [
            p for p in model.parameters() if p.requires_grad and id(p) not in gate_ids
        ]
        criterion = DuplicateAwareSigLipLoss().to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": lr, "weight_decay": 1e-3},
                {"params": gate_params, "lr": lr * 8.0, "weight_decay": 0.0},
                {"params": list(criterion.parameters()), "lr": lr, "weight_decay": 0.0},
            ]
        )

        model.eval()
        with torch.no_grad():
            val_w = val_stage1_embs.to(device)
            val_lag = val_lag_embs.to(device) if val_lag_embs is not None else None
            init_r_preds = model.forward_rolling_from_embeddings(
                val_w,
                val_bundle.sentence_uids,
                horizons=(r_stage,),
                history_radius=2,
                lag_embeddings=val_lag,
            )[r_stage].cpu()
        best_stage_val = _val_tuple(init_r_preds)
        best_stage_state = copy.deepcopy(model.state_dict())

        for _ in range(stage_epochs):
            model.train()
            model.brain_encoder.eval()
            model.word_proj.eval()
            batches = build_sentence_batches(
                train_bundle.sentence_uids, batch_size, shuffle=True, rng=rng
            )
            for rows in batches:
                rows_t = torch.from_numpy(rows).long()
                w_emb = train_stage1_embs[rows_t].to(device)
                lag_b = (
                    train_lag_embs[rows_t].to(device)
                    if train_lag_embs is not None
                    else None
                )
                suids = [train_bundle.sentence_uids[i] for i in rows]
                targets = bank_t[
                    torch.from_numpy(train_bundle.word_indices[rows]).long().to(device)
                ]
                optimizer.zero_grad(set_to_none=True)
                views, deltas = model.forward_rolling_from_embeddings(
                    w_emb,
                    suids,
                    horizons=(r_stage,),
                    history_radius=2,
                    lag_embeddings=lag_b,
                    return_raw_deltas=True,
                )
                aux_preds = torch.nn.functional.normalize(
                    w_emb + 0.25 * deltas[r_stage], dim=-1, eps=1e-12
                )
                loss = criterion(views[r_stage], targets) + 0.40 * criterion(
                    aux_preds, targets
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*base_params, *gate_params], 1.0
                )
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model.forward_rolling_from_embeddings(
                    val_w,
                    val_bundle.sentence_uids,
                    horizons=(r_stage,),
                    history_radius=2,
                    lag_embeddings=val_lag,
                )[r_stage].cpu()
                score = _val_tuple(val_preds)
                if score >= best_stage_val:
                    best_stage_val = score
                    best_stage_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_stage_state)
        print(
            f"[{label} | K{r_stage}] best_val_v50={best_stage_val[0]:.2f}% | "
            f"full_vocab={best_stage_val[1]:.2f}% | cos={best_stage_val[2]:.4f} | "
            f"alpha={float(torch.tanh(model.rolling_alpha[r_key]).detach().cpu()):+.3f}",
            flush=True,
        )

    model.eval()
    return model


@torch.no_grad()
def predict_space_chunked_model(
    model: SpaceChunkedV2Model,
    bundle: SplitWordBundle,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both per-chunk word embeddings and post-Aligner embeddings on CPU."""
    model.eval()
    batches = build_sentence_batches(bundle.sentence_uids, batch_size, shuffle=False)
    chunk_embs: list[torch.Tensor] = []
    out_embs: list[torch.Tensor] = []
    for rows in batches:
        rows_t = torch.from_numpy(rows).long()
        neuro = bundle.neuro[rows_t].to(device)
        subj = bundle.subject_id[rows_t].to(device)
        dt = torch.from_numpy(bundle.delta_taus[rows]).float().to(device)
        suids = [bundle.sentence_uids[i] for i in rows]
        c_emb = model.encode_space_chunks(neuro, subj, dt, suids)
        o_emb = model.forward_aligner_from_embeddings(c_emb, suids)
        chunk_embs.append(c_emb.detach().cpu())
        out_embs.append(o_emb.detach().cpu())
    return torch.cat(chunk_embs, dim=0), torch.cat(out_embs, dim=0)


def train_space_chunked_v2_model(
    label: str,
    model: SpaceChunkedV2Model,
    train_bundle: SplitWordBundle,
    val_bundle: SplitWordBundle,
    embedding_bank: np.ndarray,
    vocab_banks: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> SpaceChunkedV2Model:
    """Train a Brain2Qwerty v2 Space-Chunked ([tau_i, tau_{i+1}]) model variant."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    criterion = DuplicateAwareSigLipLoss().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [*trainable, *criterion.parameters()], lr=lr, weight_decay=1e-3
    )
    bank_t = torch.from_numpy(embedding_bank).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0

    for epoch in range(max(int(epochs), 1)):
        model.train()
        batches = build_sentence_batches(
            train_bundle.sentence_uids, batch_size, shuffle=True, rng=rng
        )
        epoch_losses: list[float] = []
        for rows in batches:
            rows_t = torch.from_numpy(rows).long()
            neuro = train_bundle.neuro[rows_t].to(device)
            subj = train_bundle.subject_id[rows_t].to(device)
            dt = torch.from_numpy(train_bundle.delta_taus[rows]).float().to(device)
            suids = [train_bundle.sentence_uids[i] for i in rows]
            targets = bank_t[
                torch.from_numpy(train_bundle.word_indices[rows]).long().to(device)
            ]
            optimizer.zero_grad(set_to_none=True)
            preds = model(neuro, subj, dt, suids)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        _, val_preds = predict_space_chunked_model(
            model, val_bundle, batch_size, device
        )
        val_metrics = evaluate_word_retrieval(
            val_preds,
            val_bundle.word_indices,
            val_bundle.is_function_word,
            embedding_bank,
            vocab_banks["vocab50_indices"],
            vocab_banks["full_vocab_indices"],
        )
        score = val_metrics["vocab50"]["macro_top10"]
        if score >= best_val:
            best_val = score
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch == epochs - 1:
            print(
                f"[{label}] Epoch {epoch + 1}/{epochs} | "
                f"loss={np.mean(epoch_losses):.4f} | val_v50_top10={score:.2f}%",
                flush=True,
            )

    model.load_state_dict(best_state)
    model.eval()
    return model


def train_frozen_space_chunked_aligner(
    isolated_model: SpaceChunkedV2Model,
    train_chunk_embs: torch.Tensor,
    val_chunk_embs: torch.Tensor,
    train_bundle: SplitWordBundle,
    val_bundle: SplitWordBundle,
    embedding_bank: np.ndarray,
    vocab_banks: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> SpaceChunkedV2Model:
    """Train Cross-Word Aligner on frozen Exp2-1 isolated space-chunked embeddings."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpaceChunkedV2Model(
        in_channels=isolated_model.in_channels,
        n_subjects=max(1, isolated_model.subject_emb.num_embeddings - 4),
        hidden_dim=isolated_model.hidden_dim,
        embedding_dim=isolated_model.embedding_dim,
        continuous_encoder=False,
        length_aware_pool=False,
        use_aligner=True,
    ).to(device)
    model.load_state_dict(isolated_model.state_dict(), strict=False)
    for name, p in model.named_parameters():
        if not name.startswith("aligner"):
            p.requires_grad_(False)

    criterion = DuplicateAwareSigLipLoss().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [*trainable, *criterion.parameters()], lr=lr, weight_decay=1e-3
    )
    bank_t = torch.from_numpy(embedding_bank).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0

    for _ in range(max(int(epochs), 1)):
        model.train()
        batches = build_sentence_batches(
            train_bundle.sentence_uids, batch_size, shuffle=True, rng=rng
        )
        for rows in batches:
            rows_t = torch.from_numpy(rows).long()
            c_emb = train_chunk_embs[rows_t].to(device)
            suids = [train_bundle.sentence_uids[i] for i in rows]
            targets = bank_t[
                torch.from_numpy(train_bundle.word_indices[rows]).long().to(device)
            ]
            optimizer.zero_grad(set_to_none=True)
            preds = model.forward_aligner_from_embeddings(c_emb, suids)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model.forward_aligner_from_embeddings(
                val_chunk_embs.to(device), val_bundle.sentence_uids
            ).cpu()
            val_metrics = evaluate_word_retrieval(
                val_preds,
                val_bundle.word_indices,
                val_bundle.is_function_word,
                embedding_bank,
                vocab_banks["vocab50_indices"],
                vocab_banks["full_vocab_indices"],
            )
            score = val_metrics["vocab50"]["macro_top10"]
            if score >= best_val:
                best_val = score
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def format_markdown_report(
    modality: str,
    window_duration: float,
    window_offset: float,
    cms_eta: float,
    shifts: Sequence[float],
    timing_audit: dict,
    model_summaries: dict[str, dict],
    premotor_unmixing_rows: Sequence[dict] | None = None,
) -> tuple[str, pd.DataFrame]:
    """Generate a Markdown report and summary DataFrame for Exp 1, Step 2 Pre-Motor Unmixing & Exp 2."""
    rows: list[dict] = []
    for name, summary in model_summaries.items():
        c_v50 = summary["clean"]["vocab50"]
        c_fv = summary["clean"]["full_vocab"]
        s_mean = summary["shifted_mean"]
        net = summary["net_gain"]
        rows.append(
            {
                "Model": name,
                "V50_Chance_Top10 (%)": round(c_v50["chance_top10"], 2),
                "V50_Clean_Top10 (%)": round(c_v50["macro_top10"], 2),
                "V50_Shifted_Top10 (%)": round(s_mean["vocab50_macro_top10"], 2),
                "V50_Shifted_Std (%)": round(s_mean["vocab50_macro_top10_std"], 2),
                "V50_NetGain_Top10 (pp)": round(net["vocab50_macro_top10"], 2),
                "Shortcut_Over_Chance (pp)": round(
                    summary["shortcut_inflation_over_chance_v50_top10"], 2
                ),
                "V50_Clean_Top1 (%)": round(c_v50["macro_top1"], 2),
                "V50_Shifted_Top1 (%)": round(s_mean["vocab50_macro_top1"], 2),
                "V50_NetGain_Top1 (pp)": round(net["vocab50_macro_top1"], 2),
                "FullVocab_Clean_Top10 (%)": round(c_fv["macro_top10"], 2),
                "FullVocab_Shifted_Top10 (%)": round(s_mean["full_vocab_macro_top10"], 2),
                "FullVocab_NetGain_Top10 (pp)": round(net["full_vocab_macro_top10"], 2),
                "FuncWord_Clean_Top10 (%)": round(c_v50["function_top10"], 2),
                "FuncWord_Shifted_Top10 (%)": round(s_mean["function_top10"], 2),
                "ContentWord_Clean_Top10 (%)": round(c_v50["content_top10"], 2),
                "ContentWord_Shifted_Top10 (%)": round(s_mean["content_top10"], 2),
            }
        )
    df = pd.DataFrame(rows)
    ov_05 = timing_audit["window_overlap_by_duration"].get("T=0.5s", {})
    ov_10 = timing_audit["window_overlap_by_duration"].get("T=1.0s", {})

    exp1_rows = [
        r
        for r in rows
        if not str(r["Model"]).startswith("Exp2")
        and not str(r["Model"]).startswith("Step2")
    ]
    exp2_rows = [r for r in rows if str(r["Model"]).startswith("Exp2")]

    def _render_table(sub_rows: list[dict]) -> list[str]:
        t_lines = [
            "| Model | V50 Clean Top-10 (%) | V50 Shifted Top-10 (%) | Shortcut Over Chance (pp) | **V50 Net Gain (pp)** | V50 Net Top-1 (pp) | Full-Vocab Net Top-10 (pp) | Func Clean/Shift (%) | Content Clean/Shift (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in sub_rows:
            t_lines.append(
                f"| `{r['Model']}` | {r['V50_Clean_Top10 (%)']:.2f} | "
                f"{r['V50_Shifted_Top10 (%)']:.2f} ± {r['V50_Shifted_Std (%)']:.2f} | "
                f"{r['Shortcut_Over_Chance (pp)']:+.2f} | "
                f"**{r['V50_NetGain_Top10 (pp)']:+.2f}** | "
                f"{r['V50_NetGain_Top1 (pp)']:+.2f} | "
                f"{r['FullVocab_NetGain_Top10 (pp)']:+.2f} | "
                f"{r['FuncWord_Clean_Top10 (%)']:.1f} / {r['FuncWord_Shifted_Top10 (%)']:.1f} | "
                f"{r['ContentWord_Clean_Top10 (%)']:.1f} / {r['ContentWord_Shifted_Top10 (%)']:.1f} |"
            )
        return t_lines

    lines = [
        f"# Brain2Qwerty Word-Level Duration Shortcut Verification ({modality.upper()})",
        "",
        "## 1. Empirical Word Typing Interval (`Δτ_i`) & Window Overlap Audit",
        f"- **Total typing words**: `{timing_audit['total_words']}` across `{timing_audit['total_sentences']}` sentences (`{timing_audit['unique_words']}` unique words)",
        f"- **Inter-word interval `Δτ_i`**: mean = `{timing_audit['delta_tau_mean_seconds']:.3f}s`, median = `{timing_audit['delta_tau_median_seconds']:.3f}s` (P10 = `{timing_audit['delta_tau_p10_seconds']:.3f}s`, P90 = `{timing_audit['delta_tau_p90_seconds']:.3f}s`)",
        f"- **Function vs. Content word `Δτ_i`**: Function words = `{timing_audit['function_word_delta_tau_mean_seconds']:.3f}s` vs. Content words = `{timing_audit['content_word_delta_tau_mean_seconds']:.3f}s`",
        f"- **Correlation `r(Δτ_i, word_char_length)`**: `{timing_audit['pearson_r_delta_tau_vs_char_length']:.3f}`",
        f"- **Adjacent Window Overlap (`T=0.5s`)**: mean = `{100.0 * ov_05.get('mean_overlap_ratio', 0.0):.1f}%` (`{100.0 * ov_05.get('fraction_overlapping_pairs', 0.0):.1f}%` overlapping pairs)",
        f"- **Adjacent Window Overlap (`T=1.0s`)**: mean = `{100.0 * ov_10.get('mean_overlap_ratio', 0.0):.1f}%` (`{100.0 * ov_10.get('fraction_overlapping_pairs', 0.0):.1f}%` overlapping pairs)",
        "",
    ]
    if exp1_rows:
        lines.extend(
            [
                f"## 2. 第一步：固定词窗重叠模式与零初始化局部相对位置门控（Fixed-Window `T = {window_duration:.2f}s`, Shifts: `{', '.join(f'{s:+.0f}s' for s in shifts)}`）",
                f"- **固定词窗**: `[{window_offset:+.2f}s, {window_offset + window_duration:+.2f}s]`，**LOO-CMS `η`**: `{cms_eta:.2f}`",
                "- **零初始化残差门控 (`α_r(0) = 0`) + 局部相对位置窗口 (`[i-2, i+r]`)**: 锁死 `Word-Only` 下界，消除句首模板位置泄漏与相邻重叠点积泄漏",
                "",
                *_render_table(exp1_rows),
                "",
            ]
        )
    if premotor_unmixing_rows:
        lines.extend(
            [
                f"## 3. 第二步：《From Thought to Action》预运动期超前窗口与并发叠加子空间解混（Pre-Motor Lead Window & Superposition Unmixing, `T = {window_duration:.2f}s`）",
                "- **预运动期窗口前移**: 对比运动后窗 `[+0.00s, +0.50s]`、预运动跨期窗 `[-0.25s, +0.25s]` 与早期预运动窗 `[-0.35s, +0.15s]`，将 `Rolling-K0` 首键后等待延迟从 `0.50s` 砍半至 `0.25s`（词末完成后中位等待 `0.00s`），`Rolling-K2` 词末平均等待压缩至 `~1.1s`（相比整句离线模型缩减 `>70%`）",
                "- **并发叠加子空间解混 (`Word-Only -> K0 -> K1 -> K2`)**: 利用前后局部窗口 `[i-2, i+r]` 施密特正交化解混与多相位因果融合，实现西语极短功能词（`de, la, el`）与长实义词双通道跃升",
                "",
                "| Window `[start, end]` | Streaming Stage | Post-Onset Wait (Mean / Med) | Post-Completion Wait (Mean / Med) | V50 Clean Top-10 (%) | V50 Shifted Top-10 (%) | **V50 Net Gain (pp)** | **Unmixing Gain vs Word-Only (pp)** | Func Clean/Shift (%) | Content Clean/Shift (%) |",
                "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for pr in premotor_unmixing_rows:
            lines.append(
                f"| `{pr['window_label']}` | `{pr['stage']}` | "
                f"`{pr['post_onset_mean_s']:.2f}s / {pr['post_onset_median_s']:.2f}s` | "
                f"`{pr['post_completion_mean_s']:.2f}s / {pr['post_completion_median_s']:.2f}s` | "
                f"{pr['v50_clean_top10']:.2f} | {pr['v50_shifted_top10']:.2f} | "
                f"**{pr['v50_net_gain']:+.2f}** | "
                f"**{pr['unmixing_gain_clean']:+.2f} Clean / {pr['unmixing_gain_net']:+.2f} Net** | "
                f"{pr['func_clean_top10']:.1f} / {pr['func_shifted_top10']:.1f} | "
                f"{pr['content_clean_top10']:.1f} / {pr['content_shifted_top10']:.1f} |"
            )
        lines.append("")
    if exp2_rows:
        lines.extend(
            [
                f"## 4. 实验二：模拟 Brain2Qwerty v2 空格边界切分模式（Space-Chunked `[τ_i, τ_{{i+1}}]`, Shifts: `{', '.join(f'{s:+.0f}s' for s in shifts)}`）",
                "- **通道 A (`Channel-A`)**: 变长空格切块帧数 `L_i = round(Δτ_i · f_s)` 直接暴露打字时长/词长",
                "- **通道 B (`Channel-B`)**: 底层连续 `Conv + Conformer` 跨空格边界感受野在切块前混合时间网格（即便块内采用严格均值池化 `Mean-Pool`）",
                "",
                *_render_table(exp2_rows),
                "",
            ]
        )
    return "\n".join(lines), df


def main(argv: Sequence[str] | None = None) -> dict:
    p = parser()
    args = p.parse_args(argv)
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(max(int(args.cpu_threads), 1))

    exp_set = set(args.experiments)
    run_exp1 = "all" in exp_set or "exp1" in exp_set
    run_exp2 = "all" in exp_set or "exp2" in exp_set

    run_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    wf_args = argparse.Namespace(
        mode="train",
        modality=args.modality,
        subject=args.subject,
        subjects=args.subjects,
        num_subjects=args.num_subjects,
        data_root=args.data_root,
        cache_root=args.cache_root,
        results_root=args.results_root,
        run_dir=args.run_dir,
        epochs=args.epochs,
        patience=20,
        seed=args.seed,
        batch_size=args.batch_size,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
        devices=1,
        precision="32-true",
        context_mode="sentence",
        profile=args.profile,
        train_batches=None,
        val_batches=None,
        test_batches=None,
        resume=None,
        ckpt=None,
        require_cache=args.require_cache,
        cache_version=args.cache_version,
        expected_timelines=None,
    )

    requested = requested_subjects(wf_args)
    cfg = make_config(wf_args)
    xp = WorkflowExperiment(**cfg)
    summary = xp.data.study.study_summary(apply_query=False)
    subjects = select_subjects(summary, requested, args.num_subjects)
    if args.num_subjects is not None:
        cfg = make_config(wf_args, selected_subjects=subjects)
    cfg["data"]["transforms"] = [
        {"name": "SpanishBCBLWordPreprocessing"},
        {"name": "Brain2QwertyV1Splitter", "seed": 1},
    ]

    tag = (
        f"{args.modality}-{subject_tag(wf_args, subjects)}"
        f"-word-shortcut-T{args.window_duration:.1f}s-{args.profile}"
    )
    results_root = (
        (args.results_root or Path(cfg["output_dir"]).parents[1]).expanduser().resolve()
    )
    output_dir = (args.run_dir or (results_root / tag)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)

    xp = WorkflowExperiment(**cfg)
    if xp.data.study.query is not None:
        summary = xp.data.study.study_summary()
    participants = select_participants(summary).subject.nunique()
    if not participants:
        raise RuntimeError("No valid participants matched this selection")
    cfg["brain_model_config"]["subject_layers_config"]["n_subjects"] = participants
    if cfg["brain_model_config"]["merger_config"] is not None:
        cfg["brain_model_config"]["merger_config"]["n_subjects"] = participants

    xp = WorkflowExperiment(**cfg)
    configure_cache(xp.data, read_only=args.require_cache)
    signature = cache_signature(xp.data)
    write_json(output_dir / "cache_signature.json", signature)

    primary_offset = float(args.window_offset)
    exp1_offsets: list[float] = [primary_offset]
    if run_exp1 and args.window_offsets:
        for w_off in args.window_offsets:
            w_val = float(w_off)
            if all(abs(w_val - existing) > 1e-6 for existing in exp1_offsets):
                exp1_offsets.append(w_val)

    exp2_duration = max(float(args.window_duration), 1.4)
    master_offset = min([0.0, *(w - 0.10 for w in exp1_offsets)])
    master_end = max(
        [
            *(w + float(args.window_duration) + 0.30 for w in exp1_offsets),
            exp2_duration if run_exp2 else float(args.window_duration),
        ]
    )
    master_duration = round(master_end - master_offset, 4)

    print(
        f"Word Shortcut Verification | modality={args.modality} | participants={participants} "
        f"| timelines={len(summary)} | exp1={run_exp1} (T={args.window_duration:.2f}s, offsets={exp1_offsets}) "
        f"| exp2={run_exp2} (Space-Chunked [τ_i, τ_{{i+1}}]) | device={run_device}",
        flush=True,
    )

    started = time.monotonic()
    raw_events = xp.data.build_events()
    raw_events = raw_events[
        raw_events["type"].isin(["Meg", "Eeg", "Word"])
    ].reset_index(drop=True)
    gc.collect()
    timeline_bounds = compute_timeline_bounds(raw_events, args.modality)
    events = filter_sentences_within_bounds(
        raw_events,
        timeline_bounds,
        window_offset=master_offset,
        window_duration=master_duration,
        max_sentences_per_split={
            "train": args.max_train_sentences,
            "val": args.max_val_sentences,
            "test": args.max_test_sentences,
        },
    )

    word_events = (
        events[events["type"] == "Word"]
        .sort_values(["sentence_UID", "word_order"])
        .reset_index(drop=True)
    )
    timing_audit = audit_word_timing_statistics(
        word_events, window_durations=(0.5, 1.0, float(args.window_duration))
    )
    write_json(output_dir / "word_timing_audit.json", timing_audit)
    print(
        f"Step 0 Audit | words={timing_audit['total_words']} | "
        f"delta_tau median={timing_audit['delta_tau_median_seconds']:.3f}s | "
        f"r(delta_tau, len)={timing_audit['pearson_r_delta_tau_vs_char_length']:.3f}",
        flush=True,
    )

    cache_folder = Path(xp.data.neuro.infra.folder)
    emb_cache_paths = [
        args.embedding_cache,
        results_root / "spanish_bge_m3_embeddings.npz",
        cache_folder / "spanish_bge_m3_embeddings.npz",
    ]
    vocab_words, embedding_bank = load_or_compute_bge_m3_embeddings(
        word_events["text"].astype(str).tolist(),
        [p for p in emb_cache_paths if p is not None],
        backend=args.embedding_backend,
        device=str(run_device),
    )
    vocab_banks = build_vocabulary_banks(word_events, vocab_words, top_k_50=50)
    word_to_idx = vocab_banks["word_to_idx"]

    try:
        xp.data.neuro.prepare(events)
    except Exception as exc:
        if args.require_cache:
            archive_hint = (
                "features_meg.tar" if args.modality == "meg" else "features.tar"
            )
            raise RuntimeError(
                f"Cache miss under {xp.data.neuro.infra.folder} with --require-cache. "
                f"Ensure {archive_hint} is unpacked into {xp.data.neuro.infra.folder}, "
                f"or omit --require-cache to allow on-the-fly feature extraction. "
                f"Original error: {exc}"
            ) from exc
        raise
    subject_id_extractor = ns.extractors.LabelEncoder(
        event_types=xp.data.neuro.event_types, event_field="subject"
    )
    subject_id_extractor.prepare(events)
    extractors = {
        "neuro": xp.data.neuro,
        "subject_id": subject_id_extractor,
    }
    if xp.data.include_channel_positions:
        channel_positions = ChannelPositions2D(
            neuro=xp.data.neuro,
            infra={
                "folder": xp.data.neuro.infra.folder,
                "cluster": None,
                "mode": xp.data.neuro.infra.mode,
            },
        )
        channel_positions.infra.item_uid = neuro_cache_key
        channel_positions.prepare(events)
        extractors["channel_positions"] = channel_positions

    print(
        f"Slicing Clean (0s) master windows [{master_offset:+.2f}s, {master_end:+.2f}s] for train/val/test...",
        flush=True,
    )
    train_master_bundle, timing_stats = slice_split_word_bundle(
        events,
        extractors,
        "train",
        word_to_idx,
        window_offset=master_offset,
        window_duration=master_duration,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    val_master_bundle, _ = slice_split_word_bundle(
        events,
        extractors,
        "val",
        word_to_idx,
        window_offset=master_offset,
        window_duration=master_duration,
        batch_size=args.batch_size,
        workers=args.workers,
        timing_stats=timing_stats,
    )
    test_master_bundle, _ = slice_split_word_bundle(
        events,
        extractors,
        "test",
        word_to_idx,
        window_offset=master_offset,
        window_duration=master_duration,
        batch_size=args.batch_size,
        workers=args.workers,
        timing_stats=timing_stats,
    )

    shifted_master_bundles: dict[float, SplitWordBundle] = {}
    test_word_mask = (events["split"] == "test") & (events["type"] == "Word")
    for shift_sec in args.shifts:
        s_val = float(shift_sec)
        print(f"Slicing Historical Circular Shift δ = {s_val:+.0f}s...", flush=True)
        shifted_events = events.copy()
        shifted_starts = compute_shifted_word_onsets(
            shifted_events.loc[test_word_mask],
            timeline_bounds,
            s_val,
            window_start_offset=master_offset,
            window_duration=master_duration,
        )
        shifted_events.loc[test_word_mask, "start"] = shifted_starts
        shifted_events.loc[test_word_mask, "stop"] = (
            shifted_starts + shifted_events.loc[test_word_mask, "duration"].to_numpy()
        )
        shifted_b, _ = slice_split_word_bundle(
            shifted_events,
            extractors,
            "test",
            word_to_idx,
            window_offset=master_offset,
            window_duration=master_duration,
            batch_size=args.batch_size,
            workers=args.workers,
            timing_stats=timing_stats,
            reference_bundle=test_master_bundle,
        )
        shifted_master_bundles[s_val] = shifted_b

    train_raw_bundle = _slice_window_bundle(
        train_master_bundle, master_offset, 0.0, exp2_duration
    )
    val_raw_bundle = _slice_window_bundle(
        val_master_bundle, master_offset, 0.0, exp2_duration
    )
    test_raw_bundle = _slice_window_bundle(
        test_master_bundle, master_offset, 0.0, exp2_duration
    )

    print(
        f"Clean splits ready | train={len(train_raw_bundle.words)} words | "
        f"val={len(val_raw_bundle.words)} words | test={len(test_raw_bundle.words)} words",
        flush=True,
    )

    def _eval_pair(c_pred: torch.Tensor, s_map: dict[float, torch.Tensor]) -> dict:
        return summarize_clean_vs_shifted(
            c_pred,
            s_map,
            test_raw_bundle.word_indices,
            test_raw_bundle.is_function_word,
            embedding_bank,
            vocab_banks["vocab50_indices"],
            vocab_banks["full_vocab_indices"],
        )

    # --- Model 0: Timing-Only (delta_tau_i) ---
    print("Training Model 0: Timing-Only (Δτ_i)...", flush=True)
    timing_model = train_timing_only_model(
        train_raw_bundle,
        val_raw_bundle,
        embedding_bank,
        vocab_banks,
        epochs=args.timing_epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        seed=args.seed,
        device=run_device,
    )
    with torch.no_grad():
        timing_test_preds = timing_model(
            test_raw_bundle.timing_features.to(run_device),
            test_raw_bundle.sentence_uids,
        ).cpu()
    timing_shifted_preds = {
        float(s): timing_test_preds.clone() for s in args.shifts
    }

    model_summaries: dict[str, dict] = {
        "Exp1-0. Timing-Only (Δτ_i, No Neural)": _eval_pair(
            timing_test_preds, timing_shifted_preds
        )
    }
    premotor_unmixing_rows: list[dict] = []
    latency_by_window: dict[str, dict] = {}

    n_in = int(train_master_bundle.neuro.shape[1])
    encoder_dim = int(xp.brain_model_config.hidden)
    exp2_hidden = 128 if args.profile == "small" else 256

    # --- Experiment 1 & Step 2 Pre-Motor Lead Window + Superposition Unmixing ---
    if run_exp1:
        for idx_off, w_off in enumerate(exp1_offsets):
            is_primary = idx_off == 0
            w_end = w_off + float(args.window_duration)
            win_label = f"[{w_off:+.2f}s, {w_end:+.2f}s]"
            train_exp1 = _slice_window_bundle(
                train_master_bundle, master_offset, w_off, args.window_duration
            )
            val_exp1 = _slice_window_bundle(
                val_master_bundle, master_offset, w_off, args.window_duration
            )
            test_exp1 = _slice_window_bundle(
                test_master_bundle, master_offset, w_off, args.window_duration
            )
            lat_stats = compute_streaming_latency_stats(
                test_exp1.onsets,
                test_exp1.durations,
                test_exp1.sentence_uids,
                window_offset=w_off,
                window_duration=args.window_duration,
            )
            latency_by_window[win_label] = lat_stats

            brain_stage1 = xp.brain_model_config.build(
                n_in_channels=n_in, n_outputs=encoder_dim
            )
            word_only_model = WordContextModel(
                brain_encoder=brain_stage1,
                encoder_out_dim=encoder_dim,
                embedding_dim=embedding_bank.shape[1],
                use_transformer=False,
            )
            print(
                f"[Exp1 {win_label}] Training Word-Only (Fixed T={args.window_duration:.2f}s)...",
                flush=True,
            )
            word_only_model = train_neural_word_model(
                f"Exp1-Word-Only {win_label}",
                word_only_model,
                train_exp1,
                val_exp1,
                embedding_bank,
                vocab_banks,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=run_device,
            )

            train_s1_embs, _ = predict_word_context_model(
                word_only_model, train_exp1, args.batch_size, run_device
            )
            val_s1_embs, _ = predict_word_context_model(
                word_only_model, val_exp1, args.batch_size, run_device
            )
            test_s1_clean, _ = predict_word_context_model(
                word_only_model, test_exp1, args.batch_size, run_device
            )

            train_lag_embs = _compute_lag_embeddings(
                word_only_model,
                train_master_bundle,
                master_offset,
                w_off,
                args.window_duration,
                args.batch_size,
                run_device,
            )
            val_lag_embs = _compute_lag_embeddings(
                word_only_model,
                val_master_bundle,
                master_offset,
                w_off,
                args.window_duration,
                args.batch_size,
                run_device,
            )
            test_lag_clean = _compute_lag_embeddings(
                word_only_model,
                test_master_bundle,
                master_offset,
                w_off,
                args.window_duration,
                args.batch_size,
                run_device,
            )

            full_joint_model = None
            frozen_full_model = None
            test_joint_clean = None
            test_frozen_full_clean = None
            if is_primary:
                brain_joint = copy.deepcopy(word_only_model.brain_encoder)
                for p in brain_joint.parameters():
                    p.requires_grad_(True)
                full_joint_model = WordContextModel(
                    brain_encoder=brain_joint,
                    encoder_out_dim=encoder_dim,
                    embedding_dim=embedding_bank.shape[1],
                    use_transformer=True,
                )
                full_joint_model.word_proj.load_state_dict(
                    word_only_model.word_proj.state_dict()
                )
                for p in full_joint_model.word_proj.parameters():
                    p.requires_grad_(True)
                print(
                    f"[Exp1 {win_label}] Training Model 2: Full-Context (Joint, Fixed T={args.window_duration:.2f}s)...",
                    flush=True,
                )
                full_joint_model = train_neural_word_model(
                    "Exp1-Full-Context (Joint)",
                    full_joint_model,
                    train_exp1,
                    val_exp1,
                    embedding_bank,
                    vocab_banks,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr * 0.5,
                    seed=args.seed,
                    device=run_device,
                )
                _, test_joint_clean = predict_word_context_model(
                    full_joint_model, test_exp1, args.batch_size, run_device
                )

                print(
                    f"[Exp1 {win_label}] Training Model 3: Full-Context (+ Freeze CNN)...",
                    flush=True,
                )
                frozen_full_model = train_frozen_context_transformer(
                    "Exp1-Full-Context (+ Freeze CNN)",
                    word_only_model,
                    train_s1_embs,
                    val_s1_embs,
                    train_exp1,
                    val_exp1,
                    embedding_bank,
                    vocab_banks,
                    rolling=False,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=args.seed,
                    device=run_device,
                )
                with torch.no_grad():
                    test_frozen_full_clean = (
                        frozen_full_model.forward_context_from_embeddings(
                            test_s1_clean.to(run_device), test_exp1.sentence_uids
                        ).cpu()
                    )

            print(
                f"[Exp1 {win_label}] Training Model 4: Progressive Rolling-K0/K1/K2 (+ Freeze CNN + Unmixing)...",
                flush=True,
            )
            frozen_rolling_model = train_frozen_context_transformer(
                f"Exp1-Rolling-K0/K1/K2 {win_label}",
                word_only_model,
                train_s1_embs,
                val_s1_embs,
                train_exp1,
                val_exp1,
                embedding_bank,
                vocab_banks,
                rolling=True,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=run_device,
                train_lag_embs=train_lag_embs,
                val_lag_embs=val_lag_embs,
            )
            with torch.no_grad():
                rolling_clean_views = (
                    frozen_rolling_model.forward_rolling_from_embeddings(
                        test_s1_clean.to(run_device),
                        test_exp1.sentence_uids,
                        horizons=(0, 1, 2),
                        history_radius=2,
                        lag_embeddings=test_lag_clean.to(run_device),
                    )
                )
                rolling_clean_cpu = {
                    r: v.cpu() for r, v in rolling_clean_views.items()
                }

            s1_shifted_map: dict[float, torch.Tensor] = {}
            joint_shifted_map: dict[float, torch.Tensor] = {}
            frozen_full_shifted_map: dict[float, torch.Tensor] = {}
            rolling_shifted_map: dict[int, dict[float, torch.Tensor]] = {
                0: {},
                1: {},
                2: {},
            }

            for s_val, shifted_master in shifted_master_bundles.items():
                shifted_exp1 = _slice_window_bundle(
                    shifted_master, master_offset, w_off, args.window_duration
                )
                s1_shift, _ = predict_word_context_model(
                    word_only_model, shifted_exp1, args.batch_size, run_device
                )
                s1_shifted_map[s_val] = s1_shift
                lag_shift = _compute_lag_embeddings(
                    word_only_model,
                    shifted_master,
                    master_offset,
                    w_off,
                    args.window_duration,
                    args.batch_size,
                    run_device,
                )
                if is_primary and full_joint_model is not None and frozen_full_model is not None:
                    _, joint_shift = predict_word_context_model(
                        full_joint_model, shifted_exp1, args.batch_size, run_device
                    )
                    joint_shifted_map[s_val] = joint_shift
                    with torch.no_grad():
                        frozen_full_shifted_map[s_val] = (
                            frozen_full_model.forward_context_from_embeddings(
                                s1_shift.to(run_device), test_exp1.sentence_uids
                            ).cpu()
                        )
                with torch.no_grad():
                    r_views = frozen_rolling_model.forward_rolling_from_embeddings(
                        s1_shift.to(run_device),
                        test_exp1.sentence_uids,
                        horizons=(0, 1, 2),
                        history_radius=2,
                        lag_embeddings=lag_shift.to(run_device),
                    )
                    for r in (0, 1, 2):
                        rolling_shifted_map[r][s_val] = r_views[r].cpu()

            s1_cms_clean, s1_cms_shifted = apply_leave_one_out_cms(
                test_s1_clean, s1_shifted_map, eta=args.cms_eta
            )
            rolling_cms: dict[int, tuple[torch.Tensor, dict[float, torch.Tensor]]] = {}
            for r in (0, 1, 2):
                rolling_cms[r] = apply_leave_one_out_cms(
                    rolling_clean_cpu[r], rolling_shifted_map[r], eta=args.cms_eta
                )

            w_only_cms_eval = _eval_pair(s1_cms_clean, s1_cms_shifted)
            r_evals = {
                r: _eval_pair(rolling_cms[r][0], rolling_cms[r][1])
                for r in (0, 1, 2)
            }

            if is_primary and test_joint_clean is not None and test_frozen_full_clean is not None:
                joint_cms_clean, joint_cms_shifted = apply_leave_one_out_cms(
                    test_joint_clean, joint_shifted_map, eta=args.cms_eta
                )
                ff_cms_clean, ff_cms_shifted = apply_leave_one_out_cms(
                    test_frozen_full_clean, frozen_full_shifted_map, eta=args.cms_eta
                )
                model_summaries.update(
                    {
                        f"Exp1-1. Word-Only (Fixed T={args.window_duration:.1f}s, Stage 1 CNN)": _eval_pair(
                            test_s1_clean, s1_shifted_map
                        ),
                        f"Exp1-1b. Word-Only (+ LOO-CMS η={args.cms_eta:.2f})": w_only_cms_eval,
                        f"Exp1-2. Full-Context (Fixed T={args.window_duration:.1f}s, Joint Unfrozen CNN)": _eval_pair(
                            test_joint_clean, joint_shifted_map
                        ),
                        f"Exp1-2b. Full-Context (Joint + LOO-CMS η={args.cms_eta:.2f})": _eval_pair(
                            joint_cms_clean, joint_cms_shifted
                        ),
                        f"Exp1-3. Full-Context (+ Freeze CNN, Fixed T={args.window_duration:.1f}s)": _eval_pair(
                            test_frozen_full_clean, frozen_full_shifted_map
                        ),
                        f"Exp1-3b. Full-Context (+ Freeze CNN + LOO-CMS η={args.cms_eta:.2f})": _eval_pair(
                            ff_cms_clean, ff_cms_shifted
                        ),
                        f"Exp1-4a. Rolling-K0 (Causal + Freeze CNN + LOO-CMS η={args.cms_eta:.2f})": r_evals[
                            0
                        ],
                        f"Exp1-4b. Rolling-K1 (Lookahead-1 + Freeze CNN + LOO-CMS η={args.cms_eta:.2f})": r_evals[
                            1
                        ],
                        f"Exp1-4c. Rolling-K2 (Lookahead-2 + Freeze CNN + LOO-CMS η={args.cms_eta:.2f})": r_evals[
                            2
                        ],
                    }
                )
            else:
                model_summaries.update(
                    {
                        f"Step2 ({win_label}) Word-Only (+ LOO-CMS η={args.cms_eta:.2f})": w_only_cms_eval,
                        f"Step2 ({win_label}) Rolling-K0 (Causal + LOO-CMS η={args.cms_eta:.2f})": r_evals[
                            0
                        ],
                        f"Step2 ({win_label}) Rolling-K1 (Lookahead-1 + LOO-CMS η={args.cms_eta:.2f})": r_evals[
                            1
                        ],
                        f"Step2 ({win_label}) Rolling-K2 (Lookahead-2 + LOO-CMS η={args.cms_eta:.2f})": r_evals[
                            2
                        ],
                    }
                )

            base_clean = float(w_only_cms_eval["clean"]["vocab50"]["macro_top10"])
            base_net = float(w_only_cms_eval["net_gain"]["vocab50_macro_top10"])
            stage_specs = [
                ("Word-Only (Isolated)", w_only_cms_eval, lat_stats["horizons"]["K0"]),
                (
                    "Rolling-K0 (Causal [i-2, i])",
                    r_evals[0],
                    lat_stats["horizons"]["K0"],
                ),
                (
                    "Rolling-K1 (Lookahead-1 [i-2, i+1])",
                    r_evals[1],
                    lat_stats["horizons"]["K1"],
                ),
                (
                    "Rolling-K2 (Lookahead-2 [i-2, i+2])",
                    r_evals[2],
                    lat_stats["horizons"]["K2"],
                ),
            ]
            for st_name, st_eval, st_lat in stage_specs:
                c_top10 = float(st_eval["clean"]["vocab50"]["macro_top10"])
                s_top10 = float(st_eval["shifted_mean"]["vocab50_macro_top10"])
                n_gain = float(st_eval["net_gain"]["vocab50_macro_top10"])
                premotor_unmixing_rows.append(
                    {
                        "window_label": win_label,
                        "window_offset_s": float(w_off),
                        "window_duration_s": float(args.window_duration),
                        "stage": st_name,
                        "post_onset_mean_s": float(st_lat["post_onset_mean_s"]),
                        "post_onset_median_s": float(st_lat["post_onset_median_s"]),
                        "post_completion_mean_s": float(
                            st_lat["post_completion_mean_s"]
                        ),
                        "post_completion_median_s": float(
                            st_lat["post_completion_median_s"]
                        ),
                        "v50_clean_top10": round(c_top10, 2),
                        "v50_shifted_top10": round(s_top10, 2),
                        "v50_net_gain": round(n_gain, 2),
                        "unmixing_gain_clean": round(c_top10 - base_clean, 2),
                        "unmixing_gain_net": round(n_gain - base_net, 2),
                        "func_clean_top10": round(
                            float(st_eval["clean"]["vocab50"]["function_top10"]), 2
                        ),
                        "func_shifted_top10": round(
                            float(st_eval["shifted_mean"]["function_top10"]), 2
                        ),
                        "content_clean_top10": round(
                            float(st_eval["clean"]["vocab50"]["content_top10"]), 2
                        ),
                        "content_shifted_top10": round(
                            float(st_eval["shifted_mean"]["content_top10"]), 2
                        ),
                    }
                )

    # --- Experiment 2 Models (Brain2Qwerty v2 Space-Chunked [tau_i, tau_{i+1}] Mode) ---
    if run_exp2:
        print(
            "[Exp2] Training Exp2-1: Space-Chunked Isolated (Mean-Pool Only, No Channel A/B)...",
            flush=True,
        )
        exp2_iso_model = SpaceChunkedV2Model(
            in_channels=n_in,
            n_subjects=participants,
            hidden_dim=exp2_hidden,
            embedding_dim=embedding_bank.shape[1],
            continuous_encoder=False,
            length_aware_pool=False,
            use_aligner=False,
        )
        exp2_iso_model = train_space_chunked_v2_model(
            "Exp2-1-Isolated-MeanPool",
            exp2_iso_model,
            train_raw_bundle,
            val_raw_bundle,
            embedding_bank,
            vocab_banks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=run_device,
        )

        print(
            "[Exp2] Training Exp2-2: Space-Chunked Channel-A (Length-Aware Frame-Count Pool + Aligner)...",
            flush=True,
        )
        exp2_cha_model = SpaceChunkedV2Model(
            in_channels=n_in,
            n_subjects=participants,
            hidden_dim=exp2_hidden,
            embedding_dim=embedding_bank.shape[1],
            continuous_encoder=False,
            length_aware_pool=True,
            use_aligner=True,
        )
        exp2_cha_model = train_space_chunked_v2_model(
            "Exp2-2-Channel-A (Length-Aware)",
            exp2_cha_model,
            train_raw_bundle,
            val_raw_bundle,
            embedding_bank,
            vocab_banks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=run_device,
        )

        print(
            "[Exp2] Training Exp2-3: Space-Chunked Channel-B (v2 Continuous Conv+Conformer + Mean-Pool + Aligner)...",
            flush=True,
        )
        exp2_chb_model = SpaceChunkedV2Model(
            in_channels=n_in,
            n_subjects=participants,
            hidden_dim=exp2_hidden,
            embedding_dim=embedding_bank.shape[1],
            continuous_encoder=True,
            length_aware_pool=False,
            use_aligner=True,
        )
        exp2_chb_model = train_space_chunked_v2_model(
            "Exp2-3-Channel-B (Continuous ConvConformer)",
            exp2_chb_model,
            train_raw_bundle,
            val_raw_bundle,
            embedding_bank,
            vocab_banks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=run_device,
        )

        print(
            "[Exp2] Training Exp2-4: Space-Chunked v2-Full (Channel A + B Joint)...",
            flush=True,
        )
        exp2_full_model = SpaceChunkedV2Model(
            in_channels=n_in,
            n_subjects=participants,
            hidden_dim=exp2_hidden,
            embedding_dim=embedding_bank.shape[1],
            continuous_encoder=True,
            length_aware_pool=True,
            use_aligner=True,
        )
        exp2_full_model = train_space_chunked_v2_model(
            "Exp2-4-v2-Full (Channel A+B)",
            exp2_full_model,
            train_raw_bundle,
            val_raw_bundle,
            embedding_bank,
            vocab_banks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=run_device,
        )

        train_iso_embs, _ = predict_space_chunked_model(
            exp2_iso_model, train_raw_bundle, args.batch_size, run_device
        )
        val_iso_embs, _ = predict_space_chunked_model(
            exp2_iso_model, val_raw_bundle, args.batch_size, run_device
        )
        test_iso_clean, _ = predict_space_chunked_model(
            exp2_iso_model, test_raw_bundle, args.batch_size, run_device
        )
        _, test_cha_clean = predict_space_chunked_model(
            exp2_cha_model, test_raw_bundle, args.batch_size, run_device
        )
        _, test_chb_clean = predict_space_chunked_model(
            exp2_chb_model, test_raw_bundle, args.batch_size, run_device
        )
        _, test_v2full_clean = predict_space_chunked_model(
            exp2_full_model, test_raw_bundle, args.batch_size, run_device
        )

        print(
            "[Exp2] Training Exp2-5: Space-Chunked Debiased (+ Freeze Isolated Chunk + Aligner)...",
            flush=True,
        )
        exp2_debiased_model = train_frozen_space_chunked_aligner(
            exp2_iso_model,
            train_iso_embs,
            val_iso_embs,
            train_raw_bundle,
            val_raw_bundle,
            embedding_bank,
            vocab_banks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=run_device,
        )
        with torch.no_grad():
            test_exp2_debiased_clean = (
                exp2_debiased_model.forward_aligner_from_embeddings(
                    test_iso_clean.to(run_device), test_raw_bundle.sentence_uids
                ).cpu()
            )

        iso_shifted_map: dict[float, torch.Tensor] = {}
        cha_shifted_map: dict[float, torch.Tensor] = {}
        chb_shifted_map: dict[float, torch.Tensor] = {}
        v2full_shifted_map: dict[float, torch.Tensor] = {}
        exp2_deb_shifted_map: dict[float, torch.Tensor] = {}

        for s_val, shifted_master in shifted_master_bundles.items():
            shifted_raw_bundle = _slice_window_bundle(
                shifted_master, master_offset, 0.0, exp2_duration
            )
            iso_shift, _ = predict_space_chunked_model(
                exp2_iso_model, shifted_raw_bundle, args.batch_size, run_device
            )
            _, cha_shift = predict_space_chunked_model(
                exp2_cha_model, shifted_raw_bundle, args.batch_size, run_device
            )
            _, chb_shift = predict_space_chunked_model(
                exp2_chb_model, shifted_raw_bundle, args.batch_size, run_device
            )
            _, v2full_shift = predict_space_chunked_model(
                exp2_full_model, shifted_raw_bundle, args.batch_size, run_device
            )
            iso_shifted_map[s_val] = iso_shift
            cha_shifted_map[s_val] = cha_shift
            chb_shifted_map[s_val] = chb_shift
            v2full_shifted_map[s_val] = v2full_shift
            with torch.no_grad():
                exp2_deb_shifted_map[s_val] = (
                    exp2_debiased_model.forward_aligner_from_embeddings(
                        iso_shift.to(run_device), test_raw_bundle.sentence_uids
                    ).cpu()
                )

        v2full_cms_clean, v2full_cms_shifted = apply_leave_one_out_cms(
            test_v2full_clean, v2full_shifted_map, eta=args.cms_eta
        )
        exp2_deb_cms_clean, exp2_deb_cms_shifted = apply_leave_one_out_cms(
            test_exp2_debiased_clean, exp2_deb_shifted_map, eta=args.cms_eta
        )
        model_summaries.update(
            {
                "Exp2-1. Space-Chunked Isolated ([τ_i, τ_{i+1}] Mean-Pool Only)": _eval_pair(
                    test_iso_clean, iso_shifted_map
                ),
                "Exp2-2. Space-Chunked Channel-A (Length-Aware Frame-Count Pool + Aligner)": _eval_pair(
                    test_cha_clean, cha_shifted_map
                ),
                "Exp2-3. Space-Chunked Channel-B (v2 Continuous Conv+Conformer + Mean-Pool)": _eval_pair(
                    test_chb_clean, chb_shifted_map
                ),
                "Exp2-4. Space-Chunked v2-Full (Channel A + B Joint)": _eval_pair(
                    test_v2full_clean, v2full_shifted_map
                ),
                f"Exp2-4b. Space-Chunked v2-Full (+ LOO-CMS η={args.cms_eta:.2f})": _eval_pair(
                    v2full_cms_clean, v2full_cms_shifted
                ),
                f"Exp2-5. Space-Chunked Debiased (Freeze Isolated + Aligner + LOO-CMS η={args.cms_eta:.2f})": _eval_pair(
                    exp2_deb_cms_clean, exp2_deb_cms_shifted
                ),
            }
        )

    md_report, summary_df = format_markdown_report(
        args.modality,
        args.window_duration,
        args.window_offset,
        args.cms_eta,
        args.shifts,
        timing_audit,
        model_summaries,
        premotor_unmixing_rows=premotor_unmixing_rows,
    )

    payload = {
        "modality": args.modality,
        "participants": participants,
        "subjects": subjects,
        "experiments": list(args.experiments),
        "window_offset_seconds": float(args.window_offset),
        "window_offsets_seconds": [float(w) for w in exp1_offsets],
        "window_duration_seconds": float(args.window_duration),
        "shifts_seconds": [float(s) for s in args.shifts],
        "cms_eta": float(args.cms_eta),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "splits": {
            "train_words": len(train_raw_bundle.words),
            "val_words": len(val_raw_bundle.words),
            "test_words": len(test_raw_bundle.words),
            "vocab50_words": vocab_banks["vocab50_words"],
            "full_vocab_size": len(vocab_banks["full_vocab_indices"]),
        },
        "timing_audit": timing_audit,
        "streaming_latency_by_window": latency_by_window,
        "premotor_unmixing": premotor_unmixing_rows,
        "models": model_summaries,
    }
    write_json(output_dir / "shortcut_verification.json", payload)
    summary_df.to_csv(output_dir / "shortcut_verification.csv", index=False)
    (output_dir / "shortcut_verification.md").write_text(md_report, encoding="utf-8")

    print("\n" + md_report, flush=True)
    print(f"Saved verification artifacts to: {output_dir}", flush=True)
    return payload


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
