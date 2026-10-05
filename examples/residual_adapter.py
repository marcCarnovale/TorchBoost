"""Minimal function-preserving neural→tree residual-adapter example."""

import torch
from torch import nn

from torchboost.adaptive.architecture_corners import CompositionalTreeNetwork
from torchboost.adaptive.residual_adapter import (
    adapter_signature,
    grow_frozen_backbone_adapter,
    partition_adapter_parameters,
)


class MLP(nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


torch.manual_seed(7)
x = torch.randn(256, 8)
y = (x[:, 0] * x[:, 1] + 0.5 * x[:, 2] > 0).float()

# 1. Train any ordinary ReLU MLP.
anchor_mlp = MLP()
anchor_opt = torch.optim.AdamW(anchor_mlp.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
for _ in range(20):
    anchor_opt.zero_grad(set_to_none=True)
    loss = loss_fn(anchor_mlp(x), y)
    loss.backward()
    anchor_opt.step()

# 2. Embed it exactly in TorchBoost's tree-expandable architecture.
anchor = CompositionalTreeNetwork.from_mlp(
    anchor_mlp,
    max_tree_depth=2,
    seed=7,
)
anchor.eval()

# 3. Add one zero-at-birth residual-tree refinement per hidden layer.
adapter = grow_frozen_backbone_adapter(anchor, learn_scales=True)
adapter.eval()

with torch.no_grad():
    birth_error = (adapter(x) - anchor(x)).abs().max().item()

print(f"maximum birth-function error: {birth_error:.3e}")
print(adapter_signature(adapter))
assert birth_error < 1e-5

# 4. TRAIN updates newborn residual routing/packets.
#    A held-out split may separately update only the architecture scales.
residual_params, scale_params = partition_adapter_parameters(adapter)
residual_opt = torch.optim.AdamW(residual_params, lr=1e-3)
scale_opt = torch.optim.Adam(scale_params, lr=1e-2)

adapter.train()
residual_opt.zero_grad(set_to_none=True)
loss_fn(adapter(x[:192]), y[:192]).backward()
residual_opt.step()

# Illustrative held-out architecture update.
scale_opt.zero_grad(set_to_none=True)
loss_fn(adapter(x[192:]), y[192:]).backward()
scale_opt.step()
