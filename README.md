# TorchBoost: A Differentiable Gradient Boosting Framework in PyTorch

TorchBoost is a flexible and powerful implementation of stochastic-gradient-descent ‘boosted’ soft-gated decision trees in PyTorch. It combines the strengths of traditional gradient boosting frameworks like XGBoost with modern deep learning techniques. TorchBoost supports regression, binary classification, multiclass classification, and multitarget tasks.

## Key Features

- **Differentiable Soft Decision Trees**: Utilizes soft splits with temperature scaling for differentiability, enabling end-to-end training using backpropagation.

- **Attention Mechanism for Tree Weighting**: Incorporates an attention network with dropout and batch normalization to dynamically adjust tree contributions based on the input data.

- **Residual-Based Learning with Shrinkage**: Mimics gradient boosting by having each tree focus on correcting the residuals of the previous ensemble, scaled by a shrinkage parameter.

- **Learnable Pruning Coefficients**: Allows adaptive control of pruning strength at each depth level through learnable parameters.

- **Regularization and Dropout Techniques**:
  - **Pruning Regularization**: Encourages simpler trees by penalizing deeper splits.
  - **Temperature Penalization**: Promotes the hardening of splits over time.
  - **L1 Regularization**: Encourages sparsity in residual scaling factors.
  - **Feature Dropout**: Implements hard feature dropout to prevent over-reliance on specific features.
  - **Sample Dropout (Bagging)**: Reduces overfitting by training each tree on a random subset of the data.
  - **Feature Dropout Regularization**: Applies L2 regularization to the tree weights.
  - **L2 Regularization on Leaf Values (Ridge Regression)**: Adds L2 regularization specifically on leaf values to control overfitting.

- **Supports Multiple Task Types**: Capable of handling regression, binary classification, multiclass classification, and multitarget tasks by adjusting the output layer and loss functions.

- **Handling Missing Data**: Routes missing values in a default direction during splits.

- **Feature Importance Calculation**: Provides feature importance metrics for model interpretability.

- **Advanced Optimizers and Regularization**: Utilizes optimizers like AdamW with weight decay for better regularization.

- **Learning Rate Scheduling**: Adjusts learning rate during training using schedulers like `ReduceLROnPlateau`.

- **Early Stopping**: Stops training when validation loss stops improving to prevent overfitting.

- **Gradient Clipping**: Prevents exploding gradients during training.

- **Custom Loss Functions**: Supports custom loss functions provided by the user.

- **Parameter Initialization**: Uses Xavier initialization for weights and zeros for biases in neural networks.

## Installation

Clone the repository and ensure you have PyTorch installed:

```bash
git clone https://github.com/yourusername/torchboost.git
cd torchboost
Install required packages:

pip install torch torchvision
Usage

Initialization
from torchboost import TorchBoostModel

model = TorchBoostModel(
    num_trees=10,
    input_dim=10,
    tree_depth=3,
    task_type='multiclass_classification',
    num_classes=5,
    init_temp=2.0,
    hardening_rate=0.01,
    dropout_rate=0.2,
    feature_dropout_rate=0.1,
    sample_dropout_rate=0.1,
    temperature_penalty=0.1,
    shrinkage_rate=0.3,  # Shrinkage parameter
    lambda_reg=1.0       # L2 regularization term on leaf values
)
```
# Usage

## Initialization
```
from torchboost import TorchBoostModel

model = TorchBoostModel(
    num_trees=10,
    input_dim=10,
    tree_depth=3,
    task_type='multiclass_classification',
    num_classes=5,
    init_temp=2.0,
    hardening_rate=0.01,
    dropout_rate=0.2,
    feature_dropout_rate=0.1,
    sample_dropout_rate=0.1,
    temperature_penalty=0.1,
    shrinkage_rate=0.3,  # Shrinkage parameter
    lambda_reg=1.0       # L2 regularization term on leaf values
)
```
## Training
```
from torchboost import train_torchboost

train_torchboost(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    lr=0.01,
    reg_lambda=0.1,
    optimizer_type='adamw',
    weight_decay=1e-5,
    scheduler_type='ReduceLROnPlateau',
    patience=10,
    early_stopping=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

## Evaluation
```
model.eval()
with torch.no_grad():
    predictions = model(X_test)
```

## Feature Importance
```
feature_importances = model.feature_importance()
print("Feature Importances:", feature_importances)
```

# Examples

See the examples directory for detailed examples on how to use TorchBoost for different tasks.

# Contributing

Contributions are welcome! Please open an issue or submit a pull request.

# License

This project is licensed under the MIT License.
