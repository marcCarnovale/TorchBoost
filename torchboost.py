import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Simple linear regression example
input_dim = 1  # Single feature
num_samples = 500

# Generate synthetic data
X_train = torch.randn(num_samples, input_dim)
y_train = 2 * X_train[:, 0] + 3  # y = 2x + 3

X_val = torch.randn(100, input_dim)
y_val = 2 * X_val[:, 0] + 3

# Model configuration
model = TorchBoostModel(
    num_trees=200,
    input_dim=input_dim,
    tree_depth=5,
    task_type='regression',
    num_classes=1,
    init_temp=4.0,
    hardening_rate=0.01,
    dropout_rate=0.2,             # Disable tree-level dropout
    feature_dropout_rate=0.0,     # Disable feature dropout
    sample_dropout_rate=0.0,      # Disable sample-level dropout
    temperature_penalty=0.0,      # Disable temperature penalty
    shrinkage_rate=0.1,           # Shrinkage parameter
    lambda_reg=1.0,               # Disable L2 regularization
    use_hessian=False             # Deactivate experimental Hessian-based updates
)

# Train the model
train_torchboost(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    lr=0.1,
    reg_lambda=0.0,
    optimizer_type='adamW',
    weight_decay=0.0,
    scheduler_type=None,
    patience=10,
    early_stopping=False,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_hessian=False
)

# Evaluate the model
model.eval()
with torch.no_grad():
    train_preds = model(X_train).cpu().numpy().flatten()
    val_preds = model(X_val).cpu().numpy().flatten()

# Plot predictions vs. actual values
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(), label='Actual')
plt.scatter(X_train.cpu().numpy(), train_preds, label='Predicted', alpha=0.7)
plt.title('Training Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_val.cpu().numpy(), y_val.cpu().numpy(), label='Actual')
plt.scatter(X_val.cpu().numpy(), val_preds, label='Predicted', alpha=0.7)
plt.title('Validation Data')
plt.legend()

plt.show()

# Print Mean Squared Error
train_mse = ((y_train.cpu().numpy().flatten() - train_preds) ** 2).mean()
val_mse = ((y_val.cpu().numpy().flatten() - val_preds) ** 2).mean()
print(f"Training MSE: {train_mse:.4f}")
print(f"Validation MSE: {val_mse:.4f}")
