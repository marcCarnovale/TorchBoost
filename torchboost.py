# torchboost.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom weight initialization function
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class SoftTree(nn.Module):
    def __init__(
        self,
        input_dim,
        depth,
        output_dim=1,
        init_temp=2.0,
        feature_dropout_rate=0.0,
        lambda_reg=1.0
    ):
        super(SoftTree, self).__init__()
        self.depth = depth
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.feature_dropout_rate = feature_dropout_rate
        self.output_dim = output_dim
        self.lambda_reg = lambda_reg
        num_nodes = 2 ** depth - 1

        # Parameters for internal nodes
        self.weights = nn.Parameter(torch.randn(num_nodes, input_dim))
        self.biases = nn.Parameter(torch.zeros(num_nodes))

        # Parameters for leaves
        self.leaf_values = nn.Parameter(torch.randn(2 ** depth, output_dim))

        # Learnable pruning coefficients for each depth
        self.alpha = nn.Parameter(torch.ones(depth) * 0.5)

    def forward(self, x):
        batch_size, input_dim = x.size()
        device = x.device

        # Apply feature (column) dropout
        if self.training and self.feature_dropout_rate > 0:
            feature_mask = (torch.rand(input_dim, device=device) > self.feature_dropout_rate).float()
        else:
            feature_mask = torch.ones(input_dim, device=device)

        x = x * feature_mask  # Element-wise multiplication

        # Handle missing data by replacing NaNs with zeros
        missing_mask = torch.isnan(x)
        x = torch.where(missing_mask, torch.zeros_like(x), x)

        # Start with the root node
        routing_prob = torch.ones(batch_size, 1, device=device)

        # Routing through the tree
        for depth_level in range(self.depth):
            node_indices = torch.arange(2 ** depth_level - 1, 2 ** (depth_level + 1) - 1)
            weights = self.weights[node_indices] * feature_mask  # Apply feature mask to weights
            biases = self.biases[node_indices]

            # Compute decisions at current depth
            logits = x @ weights.t() + biases
            decisions = torch.sigmoid(logits / self.temperature)
            decisions = decisions.unsqueeze(-1)  # Shape: [batch_size, num_nodes_at_depth, 1]

            # Adjust decisions for missing values
            decisions = torch.where(
                missing_mask.any(dim=1, keepdim=True),
                torch.full_like(decisions, 0.5),  # Equal probability
                decisions
            )

            # Update routing probabilities
            routing_prob = routing_prob.repeat(1, 2)
            routing_prob = routing_prob * torch.cat([decisions, 1 - decisions], dim=1)

        # Leaf node probabilities
        leaf_probs = routing_prob[:, -2 ** self.depth:]  # Shape: [batch_size, num_leaves]

        # Compute output as weighted sum of leaf values
        output = leaf_probs @ self.leaf_values  # Shape: [batch_size, output_dim]
        return output

    def pruning_regularization(self):
        reg_term = 0.0
        # Apply learnable pruning coefficients
        for depth_level in range(self.depth):
            penalty = torch.exp(-self.alpha[depth_level] * depth_level)
            reg_term += penalty ** 2
        return reg_term

    def leaf_regularization(self):
        # L2 regularization on leaf values
        return self.lambda_reg * torch.sum(self.leaf_values ** 2)

class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, num_trees):
        super(AttentionNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_trees)
        )

    def forward(self, x):
        # Compute attention scores and normalize
        scores = self.fc(x)
        weights = F.softmax(scores, dim=1)  # Shape: [batch_size, num_trees]
        return weights

class TorchBoostModel(nn.Module):
    def __init__(
        self,
        num_trees,
        input_dim,
        tree_depth,
        task_type='regression',
        num_classes=1,
        init_temp=2.0,
        hardening_rate=0.01,
        dropout_rate=0.2,
        feature_dropout_rate=0.0,
        sample_dropout_rate=0.0,
        temperature_penalty=0.1,
        shrinkage_rate=0.3,
        lambda_reg=1.0,
        use_hessian=False  # Experimental flag
    ):
        super(TorchBoostModel, self).__init__()
        self.num_trees = num_trees
        self.task_type = task_type
        self.num_classes = num_classes if task_type == 'multiclass_classification' else 1
        self.hardening_rate = hardening_rate  # Controls how quickly splits harden
        self.dropout_rate = dropout_rate  # Tree-level dropout rate
        self.sample_dropout_rate = sample_dropout_rate  # Sample-level dropout rate
        self.temperature_penalty = temperature_penalty  # Penalty coefficient for high temperatures
        self.shrinkage_rate = shrinkage_rate  # Shrinkage parameter
        self.use_hessian = use_hessian  # Experimental feature flag

        self.trees = nn.ModuleList([
            SoftTree(
                input_dim,
                tree_depth,
                output_dim=self.num_classes,
                init_temp=init_temp,
                feature_dropout_rate=feature_dropout_rate,
                lambda_reg=lambda_reg
            ) for _ in range(num_trees)
        ])

        # Attention mechanism for dynamic tree weighting
        self.attention_network = AttentionNetwork(input_dim, num_trees)

        # Learnable residual scaling factors between 0 and 1
        self.residual_weights = nn.Parameter(torch.sigmoid(torch.randn(num_trees)))

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        residual = self.current_residual  # Start with the current residual

        # Apply sample (row) dropout
        if self.training and self.sample_dropout_rate > 0:
            sample_mask = (torch.rand(batch_size, device=x.device) > self.sample_dropout_rate)
            residual = residual[sample_mask]
            x = x[sample_mask]
            y_train = self.current_y_train[sample_mask]
        else:
            y_train = self.current_y_train

        # Compute attention weights
        attention_weights = self.attention_network(x)  # Shape: [batch_size, num_trees]

        for i, tree in enumerate(self.trees):
            # Tree-level dropout
            if self.training and torch.rand(1).item() < self.dropout_rate:
                continue  # Skip this tree during training

            tree_output = tree(residual)  # Shape: [batch_size, output_dim]

            # Scale the residual correction with shrinkage_rate
            residual_correction = self.shrinkage_rate * self.residual_weights[i] * tree_output

            # Update residual for next tree
            residual = residual - residual_correction

            # Accumulate the weighted output using attention weights
            tree_weight = attention_weights[:, i].unsqueeze(-1)  # Shape: [batch_size, 1]
            outputs.append(tree_weight * tree_output)

        if outputs:
            total_output = torch.sum(torch.stack(outputs, dim=0), dim=0)  # Shape: [batch_size, output_dim]
        else:
            # If all trees are dropped, output zeros
            total_output = torch.zeros(x.size(0), self.num_classes, device=x.device)

        # For classification tasks, apply appropriate activation
        if self.task_type == 'binary_classification':
            total_output = torch.sigmoid(total_output)
        elif self.task_type == 'multiclass_classification':
            total_output = F.softmax(total_output, dim=1)

        return total_output

    def pruning_regularization(self):
        reg_term = 0.0
        for tree in self.trees:
            reg_term += tree.pruning_regularization()
        return reg_term

    def leaf_regularization(self):
        reg_term = 0.0
        for tree in self.trees:
            reg_term += tree.leaf_regularization()
        return reg_term

    def temperature_regularization(self):
        temp_reg = 0.0
        for tree in self.trees:
            # Penalize high temperatures to encourage hardening
            temp_reg += tree.temperature ** 2
        return self.temperature_penalty * temp_reg

    def lasso_regularization(self):
        # L1 regularization on residual weights to encourage sparsity
        return torch.sum(torch.abs(self.residual_weights))

    def feature_dropout_regularization(self):
        # Soft feature dropout (L2 regularization on weights)
        reg = 0.0
        for tree in self.trees:
            reg += torch.sum(tree.weights ** 2)
        return reg

    def regularization(self):
        return (
            self.pruning_regularization()
            + self.temperature_regularization()
            + self.lasso_regularization()
            + self.feature_dropout_regularization()
            + self.leaf_regularization()
        )

    def harden_splits(self):
        # Gradually decrease the temperature to harden splits
        for tree in self.trees:
            with torch.no_grad():
                tree.temperature -= self.hardening_rate
                # Ensure temperature doesn't go below a minimum value
                tree.temperature.clamp_(min=0.1)

    def feature_importance(self):
        importance = torch.zeros(self.trees[0].weights.size(1), device=self.trees[0].weights.device)
        with torch.no_grad():
            for tree in self.trees:
                # Sum the absolute weights of each feature
                importance += torch.sum(torch.abs(tree.weights), dim=0)
        # Normalize
        importance = importance / torch.sum(importance)
        return importance.cpu().numpy()

def train_torchboost(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    lr=0.01,
    reg_lambda=0.1,
    loss_function=None,
    optimizer_type='adamw',
    weight_decay=1e-5,
    scheduler_type='ReduceLROnPlateau',
    patience=10,
    early_stopping=True,
    device='cpu',
    use_hessian=False  # Experimental flag
):
    # Choose optimizer
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Choose scheduler
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience//2, factor=0.5)
    else:
        scheduler = None

    # Use custom loss function if provided
    if loss_function is not None:
        criterion = loss_function
    else:
        # Choose appropriate loss function
        if model.task_type == 'regression':
            criterion = nn.MSELoss()
        elif model.task_type == 'binary_classification':
            criterion = nn.BCELoss()
        elif model.task_type == 'multiclass_classification':
            criterion = nn.CrossEntropyLoss()
        elif model.task_type == 'multitarget':
            criterion = nn.MSELoss()  # Adjust as needed
        else:
            raise ValueError("Unsupported task type.")

    # Move data to the specified device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    model.to(device)

    # Initialize model weights
    initialize_weights(model)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Keep track of y_train and residual for sample dropout
        model.current_y_train = y_train
        if epoch == 0:
            # Initial residual is the target itself
            if model.task_type == 'multiclass_classification':
                y_one_hot = F.one_hot(y_train.long(), num_classes=model.num_classes).float()
                model.current_residual = y_one_hot.clone()
            else:
                model.current_residual = y_train.clone().unsqueeze(-1)  # Shape: [batch_size, output_dim]
        else:
            # Update residual based on previous epoch's predictions
            with torch.no_grad():
                preds = model(X_train)
                if model.task_type == 'multiclass_classification':
                    y_one_hot = F.one_hot(y_train.long(), num_classes=model.num_classes).float()
                    gradients = y_one_hot - preds
                    hessians = torch.ones_like(preds)  # Assuming constant Hessian for CrossEntropy
                    if use_hessian:
                        # Experimental: Hessian-based residual update
                        model.current_residual = gradients / (hessians + 1e-8)
                    else:
                        # Traditional residual update
                        model.current_residual = gradients
                elif model.task_type == 'regression':
                    gradients = y_train.clone().unsqueeze(-1) - preds
                    hessians = torch.ones_like(preds)  # For MSE, Hessian is constant
                    if use_hessian:
                        # Experimental: Hessian-based residual update
                        model.current_residual = gradients / (hessians + 1e-8)
                    else:
                        # Traditional residual update
                        model.current_residual = gradients
                else:
                    gradients = y_train.clone().unsqueeze(-1) - preds
                    hessians = torch.ones_like(preds)
                    if use_hessian:
                        # Experimental: Hessian-based residual update
                        model.current_residual = gradients / (hessians + 1e-8)
                    else:
                        # Traditional residual update
                        model.current_residual = gradients

        outputs = model(X_train)

        # Adjust targets and outputs based on task type
        if model.task_type == 'multiclass_classification':
            loss = criterion(outputs, y_train.long())
        else:
            loss = criterion(outputs.squeeze(), y_train)

        # Regularization
        reg_loss = reg_lambda * model.regularization()
        total_loss = loss + reg_loss

        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Harden the splits after parameter update
        model.harden_splits()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)

            if model.task_type == 'multiclass_classification':
                val_loss = criterion(val_outputs, y_val.long())
            else:
                val_loss = criterion(val_outputs.squeeze(), y_val)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}, "
            f"Reg Loss: {reg_loss.item():.4f}"
        )

        # Early stopping
        if early_stopping:
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save the best model state
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

    # Load the best model state if early stopping was triggered
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

# Example Usage
if __name__ == "__main__":
    # Synthetic dataset for multiclass classification
    num_classes = 5
    input_dim = 10
    X_train = torch.randn(500, input_dim)
    y_train = torch.randint(0, num_classes, (500,))
    X_val = torch.randn(100, input_dim)
    y_val = torch.randint(0, num_classes, (100,))

    # Model configuration
    model = TorchBoostModel(
        num_trees=10,
        input_dim=input_dim,
        tree_depth=3,
        task_type='multiclass_classification',
        num_classes=num_classes,
        init_temp=2.0,
        hardening_rate=0.01,
        dropout_rate=0.2,
        feature_dropout_rate=0.1,
        sample_dropout_rate=0.1,
        temperature_penalty=0.1,
        shrinkage_rate=0.3,  # Shrinkage parameter
        lambda_reg=1.0,       # L2 regularization term on leaf values
        use_hessian=True      # Activate experimental Hessian-based updates
    )

    # Train the model
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
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_hessian=True  # Activate experimental Hessian-based updates during training
    )

    # Compute feature importance
    feature_importances = model.feature_importance()
    print("Feature Importances:", feature_importances)
