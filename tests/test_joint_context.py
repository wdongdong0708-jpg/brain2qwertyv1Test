from types import SimpleNamespace

import pytest
import torch

from brain2qwerty_v1.fast_context import build_fast_module, decoder_state
from brain2qwerty_v1.research import state_digest
from brain2qwerty_v1.scripts.research_queue import plot_results
from neuralset.dataloader import Batch


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 2048)

    def forward(self, neuro, subject_id=None, channel_positions=None):
        return self.linear(neuro.flatten(1))


def example():
    return Batch(
        segments=[
            SimpleNamespace(trigger=SimpleNamespace(extra={"sentence_UID": "a"}))
            for _ in range(4)
        ],
        data={
            "neuro": torch.randn(4, 3, 4),
            "feature": torch.tensor([[0], [1], [2], [3]]),
            "subject_id": torch.zeros(4, 1, dtype=torch.long),
            "channel_positions": torch.zeros(4, 3, 2),
        },
    )


@pytest.mark.parametrize("mode,left,right", [("sentence", 0, 0), ("local", 8, 8)])
def test_joint_matches_cached_decoder_at_initialization_and_updates_encoder(
    mode, left, right
):
    encoder = ToyEncoder()
    torch.manual_seed(33)
    frozen = build_fast_module(mode, left, right).eval()
    torch.manual_seed(33)
    joint = build_fast_module(mode, left, right, encoder=encoder).eval()
    assert state_digest(decoder_state(joint)) == state_digest(frozen.state_dict())
    batch = example()
    with torch.no_grad():
        cached = frozen.model(encoder(batch.data["neuro"]))
    expected = frozen.prediction_views(batch, cached)[joint.primary_delay]
    actual = joint.prediction_views(batch, joint(batch))[joint.primary_delay]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    before = encoder.linear.weight.detach().clone()
    optimizer = torch.optim.AdamW(joint.optimizer_groups(), weight_decay=1e-3)
    torch.nn.functional.cross_entropy(actual, batch.data["feature"].flatten()).backward()
    assert encoder.linear.weight.grad is not None
    assert torch.isfinite(encoder.linear.weight.grad).all()
    assert encoder.linear.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(encoder.linear.weight, before)


def test_joint_optimizer_covers_each_parameter_once_with_separate_encoder_lr():
    module = build_fast_module("sentence", 0, 0, encoder=ToyEncoder())
    groups = module.optimizer_groups()
    identifiers = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == {id(parameter) for parameter in module.parameters()}
    assert [group["lr"] for group in groups] == [5e-5, 3e-4]


def test_two_job_joint_queue_can_skip_full_matrix_plot(tmp_path):
    plot_results(tmp_path, [{"job_id": "full_s33"}, {"job_id": "r8_s33"}])
