import torch

from torchboost import TorchBoostModel, train_torchboost

torch.manual_seed(7)
x = torch.randn(128, 4)
y = (x[:, 0] - 0.5 * x[:, 1]).unsqueeze(1)

model = TorchBoostModel(
    num_trees=4,
    input_dim=4,
    tree_depth=2,
    task_type="regression",
    dropout_rate=0,
)
train_torchboost(model, x[:96], y[:96], x[96:], y[96:], epochs=5, early_stopping=False)
model.eval()
with torch.no_grad():
    print(model(x[96:100]))
