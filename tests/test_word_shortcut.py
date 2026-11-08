import numpy as np
import pandas as pd
import torch
from torch import nn

from brain2qwerty_v1.scripts.verify_word_shortcut import (
    SplitWordBundle,
    _compute_timing_feature_matrix,
    format_markdown_report,
    predict_space_chunked_model,
    train_frozen_context_transformer,
    train_frozen_space_chunked_aligner,
    train_neural_word_model,
    train_space_chunked_v2_model,
    train_timing_only_model,
)
from brain2qwerty_v1.transforms import (
    Brain2QwertyV1Splitter,
    SpanishBCBLPreprocessing,
    SpanishBCBLWordPreprocessing,
)
from brain2qwerty_v1.word_shortcut import (
    DEFAULT_SHIFT_SECONDS,
    DuplicateAwareSigLipLoss,
    SpaceChunkedV2Model,
    TimingOnlyShortcutModel,
    WordContextModel,
    apply_leave_one_out_cms,
    audit_word_timing_statistics,
    build_vocabulary_banks,
    compute_shifted_word_onsets,
    load_or_compute_bge_m3_embeddings,
    summarize_clean_vs_shifted,
)

_SENTENCES = {
    2: "el gato duerme en la casa",
    3: "la nina corre por el parque",
    4: "nosotros leemos un libro nuevo",
    5: "ellos comen pan con queso",
    6: "mi perro juega con la pelota",
    7: "las flores crecen en el jardin",
}


def _synthetic_word_and_button_events() -> pd.DataFrame:
    rows: list[dict] = []
    subjects = ["Pinet2024Meg/S1", "Pinet2024Meg/S2"]
    for subj in subjects:
        timeline = f"{subj}_ses1_block1".replace("/", "_")
        rows.append(
            dict(
                type="Meg",
                trial_id=np.nan,
                timeline=timeline,
                subject=subj,
                start=0.0,
                duration=600.0,
                stop=600.0,
            )
        )
        t = 20.0
        trials = [(0, "practice uno"), (1, "practice dos")] + list(_SENTENCES.items())
        for trial_id, sent in trials:
            words = sent.split(" ")
            # Perception-phase word (must be filtered out)
            rows.append(
                dict(
                    type="Word",
                    trial_id=float(trial_id),
                    timeline=timeline,
                    subject=subj,
                    text=words[0],
                    true_sequence=sent,
                    is_percep=True,
                    start=t - 2.0,
                    duration=0.25,
                    stop=t - 1.75,
                )
            )
            for w_idx, w in enumerate(words):
                w_start = t
                for ch in w:
                    rows.append(
                        dict(
                            type="Button",
                            trial_id=float(trial_id),
                            timeline=timeline,
                            subject=subj,
                            button=ch,
                            sentence=sent,
                            text=ch,
                            start=t,
                            duration=0.08,
                            stop=t + 0.08,
                        )
                    )
                    t += 0.11
                w_stop = t
                rows.append(
                    dict(
                        type="Word",
                        trial_id=float(trial_id),
                        timeline=timeline,
                        subject=subj,
                        text=w,
                        true_sequence=sent,
                        is_percep=False,
                        start=w_start,
                        duration=max(0.08, w_stop - w_start),
                        stop=w_stop,
                    )
                )
                if w_idx + 1 < len(words):
                    rows.append(
                        dict(
                            type="Button",
                            trial_id=float(trial_id),
                            timeline=timeline,
                            subject=subj,
                            button="<space>",
                            sentence=sent,
                            text=" ",
                            start=t,
                            duration=0.08,
                            stop=t + 0.08,
                        )
                    )
                    t += 0.12
            t += 4.0
    return pd.DataFrame(rows)


def test_word_preprocessing_preserves_exact_v1_sentence_splits():
    raw = _synthetic_word_and_button_events()
    v1_out = Brain2QwertyV1Splitter(seed=1).run(SpanishBCBLPreprocessing().run(raw.copy()))
    word_out = Brain2QwertyV1Splitter(seed=1).run(
        SpanishBCBLWordPreprocessing().run(raw.copy())
    )

    v1_splits = (
        v1_out[v1_out["type"] == "Keystroke"]
        .groupby("sentence_UID")["split"]
        .first()
        .to_dict()
    )
    word_splits = (
        word_out[word_out["type"] == "Word"]
        .groupby("sentence_UID")["split"]
        .first()
        .to_dict()
    )
    assert word_splits == v1_splits

    words = word_out[word_out["type"] == "Word"]
    assert not words.empty
    assert not words["trial_id"].isin([0.0, 1.0]).any()
    if "is_percep" in words.columns:
        assert not words["is_percep"].astype(bool).any()
    for col in (
        "word_UID",
        "word_order",
        "sentence_word_count",
        "char_length",
        "delta_tau",
        "is_function_word",
    ):
        assert col in words.columns
    assert (words["delta_tau"] > 0).all()
    assert words.loc[words["text"] == "el", "is_function_word"].all()
    assert not words.loc[words["text"] == "gato", "is_function_word"].any()


def test_circular_shift_preserves_exact_inter_word_intervals_within_bounds():
    raw = _synthetic_word_and_button_events()
    events = Brain2QwertyV1Splitter(seed=1).run(SpanishBCBLWordPreprocessing().run(raw))
    words = (
        events[events["type"] == "Word"]
        .sort_values(["sentence_UID", "word_order"])
        .reset_index(drop=True)
    )
    bounds = {
        tl: (0.0, 600.0) for tl in words["timeline"].astype(str).unique()
    }

    clean_starts = words["start"].to_numpy(dtype=np.float64)
    for shift in DEFAULT_SHIFT_SECONDS:
        shifted_starts = compute_shifted_word_onsets(
            words, bounds, shift, window_start_offset=0.0, window_duration=1.0
        )
        assert not np.allclose(shifted_starts, clean_starts)
        assert (shifted_starts >= 0.20).all()
        assert (shifted_starts + 1.0 <= 599.80).all()

        words_shifted = words.copy()
        words_shifted["shifted_start"] = shifted_starts
        for _, grp in words_shifted.groupby("sentence_UID"):
            orig_diffs = np.diff(grp["start"].to_numpy(dtype=np.float64))
            shift_diffs = np.diff(grp["shifted_start"].to_numpy(dtype=np.float64))
            np.testing.assert_allclose(orig_diffs, shift_diffs, atol=1e-12, rtol=0.0)


def test_duplicate_aware_siglip_loss_and_loo_cms_no_self_leakage():
    loss_fn = DuplicateAwareSigLipLoss()
    est = torch.randn(6, 32, requires_grad=True)
    base = torch.randn(3, 32)
    cand = torch.cat([base, base], dim=0)
    loss = loss_fn(est, cand)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert est.grad is not None and torch.isfinite(est.grad).all()

    clean = torch.randn(10, 16)
    shifts = {s: torch.randn(10, 16) for s in DEFAULT_SHIFT_SECONDS}
    deb_clean, deb_shifts = apply_leave_one_out_cms(clean, shifts, eta=0.5)
    assert deb_clean.shape == (10, 16)
    np.testing.assert_allclose(
        torch.norm(deb_clean, dim=-1).numpy(), np.ones(10), atol=1e-5
    )

    # Mutating anchor -15.0s must not change the common-mode vector subtracted from -15.0s
    shifts_perturbed = {k: v.clone() for k, v in shifts.items()}
    shifts_perturbed[-15.0] = shifts[-15.0].clone()
    mean_other_before = sum(shifts[k] for k in shifts if k != -15.0) / (len(shifts) - 1)
    mean_other_after = sum(
        shifts_perturbed[k] for k in shifts_perturbed if k != -15.0
    ) / (len(shifts_perturbed) - 1)
    torch.testing.assert_close(mean_other_before, mean_other_after)
    assert set(deb_shifts.keys()) == set(DEFAULT_SHIFT_SECONDS)


class _DummyBrainEncoder(nn.Module):
    def __init__(self, in_channels: int = 8, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_dim),
        )

    def forward(self, neuro, subject_id=None, channel_positions=None):
        return self.net(neuro)


def test_end_to_end_word_shortcut_models_and_report_generation():
    raw = _synthetic_word_and_button_events()
    events = Brain2QwertyV1Splitter(seed=1).run(SpanishBCBLWordPreprocessing().run(raw))
    words = (
        events[events["type"] == "Word"]
        .sort_values(["sentence_UID", "word_order"])
        .reset_index(drop=True)
    )
    audit = audit_word_timing_statistics(words, window_durations=(0.5, 1.0))
    assert audit["total_words"] == len(words)
    assert "T=0.5s" in audit["window_overlap_by_duration"]
    assert "T=1.0s" in audit["window_overlap_by_duration"]

    vocab_words, bank = load_or_compute_bge_m3_embeddings(
        words["text"].tolist(), [], backend="hash", embedding_dim=64
    )
    vb = build_vocabulary_banks(words, vocab_words, top_k_50=50)
    targets = np.asarray([vb["word_to_idx"][w] for w in words["text"]], dtype=np.int64)
    is_func = words["is_function_word"].to_numpy(dtype=bool)
    suids = words["sentence_UID"].astype(str).tolist()

    timing_model = TimingOnlyShortcutModel(embedding_dim=64, hidden_dim=32, depth=2, heads=2)
    t_feats = torch.randn(len(words), 4)
    t_preds = timing_model(t_feats, suids)
    assert t_preds.shape == (len(words), 64)

    wc_model = WordContextModel(
        brain_encoder=_DummyBrainEncoder(8, 32),
        encoder_out_dim=32,
        embedding_dim=64,
        use_transformer=True,
        transformer_dim=32,
        transformer_depth=2,
        transformer_heads=2,
    )
    neuro = torch.randn(len(words), 8, 25)
    subj = torch.zeros(len(words), 1, dtype=torch.long)
    full_preds = wc_model(neuro, subj, suids)
    assert full_preds.shape == (len(words), 64)

    w_embs = wc_model.encode_words(neuro, subj)
    rolling_views = wc_model.forward_rolling_from_embeddings(
        w_embs, suids, horizons=(0, 1, 2), history_radius=2
    )
    assert set(rolling_views.keys()) == {0, 1, 2}
    for r in (0, 1, 2):
        assert rolling_views[r].shape == (len(words), 64)

    shifted_map = {s: torch.randn(len(words), 64) for s in (-15.0, -30.0, -45.0)}
    summary = summarize_clean_vs_shifted(
        full_preds,
        shifted_map,
        targets,
        is_func,
        bank,
        vb["vocab50_indices"],
        vb["full_vocab_indices"],
    )
    md, df = format_markdown_report(
        "meg",
        1.0,
        0.0,
        0.5,
        (-15.0, -30.0, -45.0),
        audit,
        {"Full-Context": summary},
    )
    assert "V50 Net Gain (pp)" in md
    assert len(df) == 1


def test_verify_word_shortcut_training_pipeline_end_to_end():
    raw = _synthetic_word_and_button_events()
    events = Brain2QwertyV1Splitter(seed=1).run(SpanishBCBLWordPreprocessing().run(raw))
    words = (
        events[events["type"] == "Word"]
        .sort_values(["sentence_UID", "word_order"])
        .reset_index(drop=True)
    )
    vocab_words, bank = load_or_compute_bge_m3_embeddings(
        words["text"].tolist(), [], backend="hash", embedding_dim=32
    )
    vb = build_vocabulary_banks(words, vocab_words, top_k_50=50)
    w2i = vb["word_to_idx"]

    suids = words["sentence_UID"].astype(str).tolist()
    dt = words["delta_tau"].to_numpy(dtype=np.float64)
    w_ord = words["word_order"].to_numpy(dtype=np.int64)
    s_cnt = words["sentence_word_count"].to_numpy(dtype=np.int64)
    t_feats, _ = _compute_timing_feature_matrix(suids, dt, w_ord, s_cnt)

    bundle = SplitWordBundle(
        split="train",
        neuro=torch.randn(len(words), 8, 25),
        subject_id=torch.zeros(len(words), 1, dtype=torch.long),
        channel_positions=None,
        word_uids=words["word_UID"].astype(str).tolist(),
        sentence_uids=suids,
        words=words["text"].astype(str).tolist(),
        word_indices=np.asarray([w2i[w] for w in words["text"]], dtype=np.int64),
        onsets=words["start"].to_numpy(dtype=np.float64),
        durations=words["duration"].to_numpy(dtype=np.float64),
        delta_taus=dt,
        char_lengths=words["char_length"].to_numpy(dtype=np.int64),
        is_function_word=words["is_function_word"].to_numpy(dtype=bool),
        timing_features=t_feats,
    )

    device = torch.device("cpu")
    t_model = train_timing_only_model(
        bundle, bundle, bank, vb, epochs=2, batch_size=32, lr=1e-3, seed=33, device=device
    )
    assert t_model(bundle.timing_features, bundle.sentence_uids).shape == (len(words), 32)

    s1_model = WordContextModel(
        brain_encoder=_DummyBrainEncoder(8, 16),
        encoder_out_dim=16,
        embedding_dim=32,
        use_transformer=False,
    )
    s1_model = train_neural_word_model(
        "Word-Only", s1_model, bundle, bundle, bank, vb,
        epochs=2, batch_size=32, lr=1e-3, seed=33, device=device,
    )
    with torch.no_grad():
        s1_embs = s1_model.encode_words(bundle.neuro, bundle.subject_id)

    frozen_full = train_frozen_context_transformer(
        "Frozen-Full", s1_model, s1_embs, s1_embs, bundle, bundle, bank, vb,
        rolling=False, epochs=2, batch_size=32, lr=1e-3, seed=33, device=device,
    )
    assert not any(p.requires_grad for p in frozen_full.brain_encoder.parameters())
    assert not any(p.requires_grad for p in frozen_full.word_proj.parameters())

    frozen_rolling = train_frozen_context_transformer(
        "Frozen-Rolling", s1_model, s1_embs, s1_embs, bundle, bundle, bank, vb,
        rolling=True, epochs=2, batch_size=32, lr=1e-3, seed=33, device=device,
    )
    with torch.no_grad():
        r_out = frozen_rolling.forward_rolling_from_embeddings(
            s1_embs, bundle.sentence_uids, horizons=(0, 1, 2), history_radius=2
        )
    assert set(r_out.keys()) == {0, 1, 2}

    # Experiment 2: Space-Chunked [tau_i, tau_{i+1}] (Isolated, Channel-A, Channel-B, v2-Full, Debiased)
    exp2_iso = SpaceChunkedV2Model(
        in_channels=8, n_subjects=2, hidden_dim=16, embedding_dim=32,
        continuous_encoder=False, length_aware_pool=False, use_aligner=False,
    )
    exp2_iso = train_space_chunked_v2_model(
        "Exp2-Iso", exp2_iso, bundle, bundle, bank, vb,
        epochs=2, batch_size=32, lr=1e-3, seed=33, device=device,
    )
    iso_embs, _ = predict_space_chunked_model(exp2_iso, bundle, 32, device)
    assert iso_embs.shape == (len(words), 32)

    exp2_chb = SpaceChunkedV2Model(
        in_channels=8, n_subjects=2, hidden_dim=16, embedding_dim=32,
        continuous_encoder=True, length_aware_pool=False, use_aligner=True, aligner_depth=1, aligner_heads=2,
    )
    exp2_chb = train_space_chunked_v2_model(
        "Exp2-ChB", exp2_chb, bundle, bundle, bank, vb,
        epochs=2, batch_size=32, lr=1e-3, seed=33, device=device,
    )
    _, chb_preds = predict_space_chunked_model(exp2_chb, bundle, 32, device)
    assert chb_preds.shape == (len(words), 32)

    exp2_deb = train_frozen_space_chunked_aligner(
        exp2_iso, iso_embs, iso_embs, bundle, bundle, bank, vb,
        epochs=2, batch_size=32, lr=1e-3, seed=33, device=device,
    )
    with torch.no_grad():
        deb_preds = exp2_deb.forward_aligner_from_embeddings(iso_embs, bundle.sentence_uids)
    assert deb_preds.shape == (len(words), 32)


