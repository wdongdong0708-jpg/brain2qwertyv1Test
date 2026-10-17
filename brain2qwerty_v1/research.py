import hashlib
from collections import defaultdict

import Levenshtein
import torch
from torch.nn import functional as F

from .context import context_window_indices
from .pl_module import BrainModule
from .utils import CHAR_INDEX


def sentence_layout(uids):
    groups = defaultdict(list)
    for index, uid in enumerate(uids):
        groups[uid].append(index)
    positions = [0] * len(uids)
    lengths = [0] * len(uids)
    for group in groups.values():
        for position, index in enumerate(group):
            positions[index], lengths[index] = position, len(group)
    return groups, positions, lengths


def delayed_availability(uids, delays, device=None):
    _, positions, lengths = sentence_layout(uids)
    return {
        delay: torch.tensor(
            [position + delay < length for position, length in zip(positions, lengths)],
            dtype=torch.bool,
            device=device,
        )
        for delay in delays
    }


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().data)
    return digest.hexdigest()


def summarize_predictions(rows, future=2, history=2):
    summaries = {}
    for cohort, selected in {
        "all_target": rows,
        "common_future": [r for r in rows if r["position"] + future < r["length"]],
        "strict_interior": [
            r
            for r in rows
            if r["position"] >= history and r["position"] + future < r["length"]
        ],
    }.items():
        if not selected:
            summaries[cohort] = {"events": 0}
            continue
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["sentence_UID"]].append(row)
        distances, references, sentence_cer, prompt_cer, sentences = [], [], [], [], []
        for uid, group in grouped.items():
            group = sorted(group, key=lambda row: row["position"])
            positions = [row["position"] for row in group]
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise ValueError("CER requires a contiguous sequence of target events")
            true = "".join(CHAR_INDEX[row["label"]] for row in group)
            predicted = "".join(CHAR_INDEX[row["prediction"]] for row in group)
            distance = Levenshtein.distance(true, predicted)
            distances.append(distance)
            references.append(len(true))
            sentence_cer.append(distance / len(true))
            prompt = group[0].get("prompt", "")
            if cohort == "all_target" and prompt:
                prompt_cer.append(Levenshtein.distance(prompt, predicted) / len(prompt))
            sentences.append(
                {
                    "sentence_UID": uid,
                    "typed": true,
                    "prediction": predicted,
                    "CER": sentence_cer[-1],
                    "prompt": prompt,
                }
            )
        summaries[cohort] = {
            "events": len(selected),
            "sentence_instances": len(grouped),
            "accuracy": sum(r["prediction"] == r["label"] for r in selected)
            / len(selected),
            "CE": sum(r["ce"] for r in selected) / len(selected),
            "CER_corpus": sum(distances) / sum(references),
            "CER_sentence_mean": sum(sentence_cer) / len(sentence_cer),
            "reference_characters": sum(references),
            "edit_distance": sum(distances),
            "CER_prompt_sentence_mean": sum(prompt_cer) / len(prompt_cer)
            if prompt_cer
            else None,
            "sentences": sentences,
        }
    return summaries


class ResearchModule(BrainModule):
    """Shared V1 decoder for CNN pretraining, isolated windows and dense revision."""

    def __init__(
        self,
        model,
        transformer,
        loss,
        optimizer,
        research_mode="local",
        left=2,
        right=2,
        delays=(0, 1, 2),
        freeze_cnn_epochs=0,
        query_batch_size=64,
    ):
        super().__init__(model, transformer, loss, {}, optimizer, context_mode="sentence")
        if research_mode not in {"cnn_only", "sentence", "local", "revision"}:
            raise ValueError(f"Unknown research mode: {research_mode}")
        if left < 0 or right < 0 or query_batch_size < 1:
            raise ValueError("Context radii must be nonnegative and query batch positive")
        if not delays or min(delays) < 0 or tuple(sorted(set(delays))) != tuple(delays):
            raise ValueError("Delays must be unique, nonnegative and increasing")
        self.research_mode = research_mode
        self.left, self.right, self.delays = left, right, tuple(delays)
        self.freeze_cnn_epochs = freeze_cnn_epochs
        self.query_batch_size = query_batch_size
        self.validation_rows = {}
        self.validation_summary = {}
        self.save_hyperparameters(
            {
                "research_mode": research_mode,
                "left": left,
                "right": right,
                "delays": tuple(delays),
                "freeze_cnn_epochs": freeze_cnn_epochs,
                "query_batch_size": query_batch_size,
            }
        )

    @property
    def primary_delay(self):
        return max(self.delays) if self.research_mode == "revision" else self.right

    @property
    def checkpoint_monitor(self):
        return (
            f"val_ce_common_k{max(self.delays)}"
            if self.research_mode == "revision"
            else "val_ce_all"
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "freeze_cnn_epochs", 0) > self.current_epoch:
            self.model.eval()
        return self

    def on_train_epoch_start(self):
        frozen = self.current_epoch < self.freeze_cnn_epochs
        for parameter in self.model.parameters():
            parameter.requires_grad_(not frozen)
        self.model.train(not frozen)

    def local_logits(self, embeddings, uids, left, right):
        indices, valid = context_window_indices(uids, left, right, embeddings.device)
        outputs = []
        for start in range(0, len(embeddings), self.query_batch_size):
            stop = start + self.query_batch_size
            mask = valid[start:stop]
            windows = (
                embeddings[indices[start:stop]].float().masked_fill(~mask[..., None], 0)
            )
            outputs.append(self.linear(self.transformer(windows, mask=mask)[:, left]))
        return torch.cat(outputs)

    def prediction_views(self, batch, embeddings):
        uids = [segment.trigger.extra["sentence_UID"] for segment in batch.segments]
        if self.research_mode == "cnn_only":
            return {self.primary_delay: self.linear(embeddings)}
        if self.research_mode == "sentence":
            return {self.primary_delay: super()._transformer_forward(batch, embeddings)}
        horizons = self.delays if self.research_mode == "revision" else (self.right,)
        return {
            delay: self.local_logits(embeddings, uids, self.left, delay)
            for delay in horizons
        }

    def loss_views(self, views, labels, uids):
        available = delayed_availability(uids, views, labels.device)
        losses, counts = [], {}
        for delay, logits in views.items():
            valid = (
                available[delay]
                if self.research_mode == "revision"
                else torch.ones_like(labels, dtype=torch.bool)
            )
            counts[delay] = int(valid.sum())
            if counts[delay]:
                losses.append(F.cross_entropy(logits[valid], labels[valid]))
        return torch.stack(losses).mean(), counts

    def training_step(self, batch, batch_idx):
        labels = batch.data[self.y_name].squeeze(1).long()
        views = self.prediction_views(batch, self.forward(batch))
        uids = [segment.trigger.extra["sentence_UID"] for segment in batch.segments]
        loss, counts = self.loss_views(views, labels, uids)
        self.last_target_counts = counts
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(labels))
        for delay, count in counts.items():
            self.log(
                f"train_targets_k{delay}",
                float(count),
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
                batch_size=1,
            )
        return loss

    def on_validation_epoch_start(self):
        self.validation_rows = {}

    def validation_step(self, batch, batch_idx):
        labels = batch.data[self.y_name].squeeze(1).long()
        uids = [segment.trigger.extra["sentence_UID"] for segment in batch.segments]
        groups, positions, lengths = sentence_layout(uids)
        views = self.prediction_views(batch, self.forward(batch))
        for delay, logits in views.items():
            ce = F.cross_entropy(logits, labels, reduction="none")
            if delay == self.primary_delay:
                valid = (
                    delayed_availability(uids, [max(self.delays)], labels.device)[
                        max(self.delays)
                    ]
                    if self.research_mode == "revision"
                    else torch.ones_like(labels, dtype=torch.bool)
                )
                if valid.any():
                    self.log(
                        self.checkpoint_monitor,
                        ce[valid].mean(),
                        on_step=False,
                        on_epoch=True,
                        batch_size=int(valid.sum()),
                    )
            predictions = logits.argmax(-1).detach().cpu().tolist()
            values = ce.detach().float().cpu().tolist()
            targets = labels.detach().cpu().tolist()
            store = self.validation_rows.setdefault(delay, [])
            for index, segment in enumerate(batch.segments):
                group = groups[uids[index]]
                position = positions[index]
                effective = (
                    lengths[index] - 1 - position
                    if self.research_mode == "sentence"
                    else min(delay, lengths[index] - 1 - position)
                )
                last = batch.segments[group[position + effective]].trigger
                onset = float(getattr(segment.trigger, "start", 0.0))
                store.append(
                    {
                        "sentence_UID": uids[index],
                        "position": position,
                        "length": lengths[index],
                        "label": targets[index],
                        "prediction": predictions[index],
                        "ce": values[index],
                        "prompt": segment.trigger.extra.get("sentence", ""),
                        "requested_delay": "full"
                        if self.research_mode == "sentence"
                        else delay,
                        "effective_delay": effective,
                        "onset": onset,
                        "signal_wait_seconds": float(getattr(last, "start", onset))
                        + 0.3
                        - onset,
                    }
                )

    def on_validation_epoch_end(self):
        self.validation_summary = {
            str(delay): summarize_predictions(rows)
            for delay, rows in self.validation_rows.items()
        }
        primary = self.validation_summary[str(self.primary_delay)]["all_target"]
        self.log("val_CER_corpus", primary["CER_corpus"], batch_size=primary["events"])
        self.log("val_accuracy", primary["accuracy"], batch_size=primary["events"])
