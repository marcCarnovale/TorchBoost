# TorchBoost: A Differentiable Gradient Boosting Framework in PyTorch

TorchBoost is a flexible and powerful implementation of stochastic-gradient-descent ‘boosted’ soft-gated decision trees in PyTorch. It combines the strengths of traditional gradient boosting frameworks like XGBoost with modern deep learning techniques. TorchBoost supports regression, binary classification, multiclass classification, and multitarget tasks.

## Key Features

- **Differentiable Soft Decision Trees**: Utilizes soft splits with temperature scaling for differentiability, enabling end-to-end training using backpropagation.

- **Attention Mechanism for Tree Weighting**: Incorporates an attention network with dropout and batch normalization to dynamically adjust tree contributions based on the input data.

- **Learnable Pruning Coefficients**: Allows adaptive control of pruning strength at each depth level through learnable parameters.

- **Regularization and Dropout Techniques**:
  - **Pruning Regularization**: Encourages simpler trees by penalizing deeper splits.
  - **Temperature Penalization**: Promotes the hardening of splits over time.
  - **L1 Regularization**: Encourages sparsity in scaling factors.
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
Please see the Examples folder.

# Grokking and Deep Double Descent: Training Dynamics Observed

## **Deep Double Descent Observed in Training**

The training logs below highlight characteristics of **deep double descent** and **grokking**, phenomena where model performance exhibits significant fluctuations before a phase transition to improved generalization.

### **Training and Validation Loss Over Epochs**

- **Phase 1:** Gradual improvement in training and validation loss.
- **Phase 2:** Fluctuations and spikes in validation loss, indicative of the model struggling to generalize.
- **Phase 3 (Epoch ~36):** Sudden improvement in validation performance, possibly due to the model discovering a generalizable structure in the data.

---

```plaintext
Epoch 1/50, Loss: 31.3433, Val Loss: 33.7022, Reg Loss: 1131.9139
Epoch 2/50, Loss: 31.4156, Val Loss: 33.5344, Reg Loss: 517.2280
Epoch 3/50, Loss: 31.1985, Val Loss: 33.6515, Reg Loss: 225.7029
Epoch 4/50, Loss: 31.1338, Val Loss: 32.9448, Reg Loss: 260.9095
Epoch 5/50, Loss: 29.7845, Val Loss: 31.8369, Reg Loss: 239.7323
Epoch 6/50, Loss: 27.9809, Val Loss: 29.9465, Reg Loss: 146.5982
Epoch 7/50, Loss: 24.5061, Val Loss: 24.2006, Reg Loss: 115.4503
Epoch 8/50, Loss: 21.0814, Val Loss: 19.4988, Reg Loss: 146.3978
Epoch 9/50, Loss: 16.2921, Val Loss: 14.8435, Reg Loss: 134.6486
Epoch 10/50, Loss: 12.0479, Val Loss: 11.4106, Reg Loss: 95.1880
Epoch 11/50, Loss: 10.0514, Val Loss: 10.1216, Reg Loss: 74.6541
Epoch 12/50, Loss: 11.2637, Val Loss: 10.7189, Reg Loss: 80.0230
Epoch 13/50, Loss: 11.8447, Val Loss: 11.9450, Reg Loss: 109.1737
Epoch 14/50, Loss: 11.4881, Val Loss: 9.9922, Reg Loss: 121.3766
Epoch 15/50, Loss: 9.8583, Val Loss: 8.9252, Reg Loss: 101.8225
Epoch 16/50, Loss: 7.6707, Val Loss: 8.3787, Reg Loss: 64.6770
Epoch 17/50, Loss: 7.2454, Val Loss: 8.3326, Reg Loss: 40.2868
Epoch 18/50, Loss: 6.8811, Val Loss: 8.1310, Reg Loss: 68.6169
Epoch 19/50, Loss: 7.1206, Val Loss: 7.0910, Reg Loss: 67.0331
Epoch 20/50, Loss: 6.1827, Val Loss: 4.9658, Reg Loss: 47.4662
Epoch 21/50, Loss: 5.8680, Val Loss: 4.8095, Reg Loss: 38.3000
Epoch 22/50, Loss: 5.5374, Val Loss: 4.4963, Reg Loss: 39.0331
Epoch 23/50, Loss: 4.8373, Val Loss: 3.6009, Reg Loss: 41.0023
Epoch 24/50, Loss: 4.6580, Val Loss: 3.9487, Reg Loss: 39.7916
Epoch 25/50, Loss: 5.0214, Val Loss: 5.3262, Reg Loss: 29.2221
Epoch 26/50, Loss: 4.5154, Val Loss: 3.1111, Reg Loss: 35.9907
Epoch 27/50, Loss: 3.8136, Val Loss: 2.8106, Reg Loss: 32.9817
**Epoch 28/50, Loss: 3.1440, Val Loss: 2.7574, Reg Loss: 27.8697**
**Epoch 29/50, Loss: 4.0443, Val Loss: 5.0673, Reg Loss: 26.7645**
Epoch 30/50, Loss: 3.3682, Val Loss: 8.3288, Reg Loss: 31.8566
Epoch 31/50, Loss: 2.8219, Val Loss: 9.0571, Reg Loss: 28.4102
Epoch 32/50, Loss: 2.9918, Val Loss: 8.5857, Reg Loss: 23.5178
Epoch 33/50, Loss: 2.9534, Val Loss: 8.4809, Reg Loss: 24.4203
Epoch 34/50, Loss: 2.6855, Val Loss: 8.2976, Reg Loss: 23.1677
**Epoch 35/50, Loss: 2.6912, Val Loss: 7.3977, Reg Loss: 22.7469**
**Epoch 36/50, Loss: 3.5889, Val Loss: 1.8081, Reg Loss: 20.3004**
Epoch 37/50, Loss: 2.1422, Val Loss: 1.5238, Reg Loss: 30.1116
Epoch 38/50, Loss: 1.9478, Val Loss: 1.3942, Reg Loss: 23.8594
Epoch 39/50, Loss: 1.8450, Val Loss: 1.3041, Reg Loss: 19.5260
Epoch 40/50, Loss: 1.7382, Val Loss: 1.0609, Reg Loss: 22.5895
Epoch 41/50, Loss: 1.5139, Val Loss: 1.0543, Reg Loss: 21.6949
Epoch 42/50, Loss: 1.5057, Val Loss: 0.7873, Reg Loss: 18.5872
Epoch 43/50, Loss: 1.2268, Val Loss: 0.7189, Reg Loss: 22.4377
Epoch 44/50, Loss: 1.1276, Val Loss: 0.6419, Reg Loss: 18.5636
Epoch 45/50, Loss: 1.0309, Val Loss: 0.5097, Reg Loss: 19.0123
Epoch 46/50, Loss: 0.8793, Val Loss: 0.3790, Reg Loss: 20.2160
Epoch 47/50, Loss: 0.7336, Val Loss: 0.3732, Reg Loss: 18.9200
Epoch 48/50, Loss: 0.6680, Val Loss: 0.2480, Reg Loss: 18.1010
Epoch 49/50, Loss: 0.5452, Val Loss: 0.5336, Reg Loss: 18.8293
Epoch 50/50, Loss: 0.7399, Val Loss: 0.2130, Reg Loss: 17.5326
![image](https://github.com/user-attachments/assets/0616621b-a52f-48eb-b28f-0a7cb0ea26de)




# Contributing

Contributions are welcome! Please open an issue or submit a pull request.

# License

This project is licensed under the MIT License.
