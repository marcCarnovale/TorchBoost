# torchboost.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        dropout_rate=0.0,  # Soft feature dropout rate
        lambda_reg=1.0
    ):
        super(SoftTree, self).__init__()
        self.depth = depth
        self.temperature = nn.Parameter(torch.tensor(init_temp))
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

        # Soft feature dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size, input_dim = x.size()
        device = x.device

        # Apply soft feature dropout
        x = self.dropout(x)

        # Handle missing data by replacing NaNs with zeros
        missing_mask = torch.isnan(x)
        x = torch.where(missing_mask, torch.zeros_like(x), x)

        # Start with the root node
        routing_prob = torch.ones(batch_size, 1, device=device)

        # Routing through the tree
        for depth_level in range(self.depth):
            node_indices = torch.arange(2 ** depth_level - 1, 2 ** (depth_level + 1) - 1, device=device)
            weights = self.weights[node_indices]
            biases = self.biases[node_indices]

            # Compute decisions at current depth
            logits = x @ weights.t() + biases
            decisions = torch.sigmoid(logits / self.temperature)

            # Adjust decisions for missing values
            missing_any = missing_mask.any(dim=1, keepdim=True).expand(-1, decisions.size(1))
            decisions = torch.where(
                missing_any,
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
        temperature_penalty=0.1,
        shrinkage_rate=0.3,
        lambda_reg=1.0,
        lasso_reg_weight=0.1,
        pruning_reg_weight=0.1,
        temp_reg_weight=0.1,
        leaf_reg_weight=0.1,  # NEW: weight for leaf regularization
        diversity_reg_weight=0.1,
        hardening_adjustment_factor=0.1,   # NEW: backoff adjustment factor
        hardening_backoff_decay_rate=0.05, # NEW: backoff decay rate
        use_hessian=False
    ):
        super(TorchBoostModel, self).__init__()
        self.num_trees = num_trees
        self.task_type = task_type
        self.num_classes = num_classes if task_type == 'multiclass_classification' else 1
        self.hardening_rate = hardening_rate
        self.dropout_rate = dropout_rate
        self.temperature_penalty = temperature_penalty
        self.shrinkage_rate = shrinkage_rate
        self.lambda_reg = lambda_reg
        self.lasso_reg_weight = lasso_reg_weight
        self.pruning_reg_weight = pruning_reg_weight
        self.temp_reg_weight = temp_reg_weight
        self.leaf_reg_weight = leaf_reg_weight
        self.diversity_reg_weight = diversity_reg_weight
        self.hardening_adjustment_factor = hardening_adjustment_factor
        self.hardening_backoff_decay_rate = hardening_backoff_decay_rate
        self.use_hessian = use_hessian
        self.init_temp = init_temp  # Store initial temperature

        self.trees = nn.ModuleList([
            SoftTree(
                input_dim,
                tree_depth,
                output_dim=self.num_classes,
                init_temp=init_temp,
                dropout_rate=dropout_rate,
                lambda_reg=lambda_reg
            ) for _ in range(num_trees)
        ])

        # Attention mechanism for dynamic tree weighting
        self.attention_network = AttentionNetwork(input_dim, num_trees)

        # Learnable residual scaling factors between 0 and 1
        self.residual_weights = nn.Parameter(torch.sigmoid(torch.randn(num_trees)))

        # Initialize backoff charge
        self.backoff_charge = 0.0

    def forward(self, x):
        batch_size = x.size(0)

        # Compute attention weights
        attention_weights = self.attention_network(x)  # Shape: [batch_size, num_trees]

        # Initialize prediction
        prediction = torch.zeros(batch_size, self.num_classes, device=x.device)

        for i, tree in enumerate(self.trees):
            # Tree-level dropout
            if self.training and torch.rand(1).item() < self.dropout_rate:
                continue  # Skip this tree during training

            tree_output = tree(x)  # Shape: [batch_size, output_dim]

            # Scale the tree output with shrinkage_rate and residual weight
            scaled_output = self.shrinkage_rate * self.residual_weights[i] * tree_output

            # Accumulate the weighted output using attention weights
            tree_weight = attention_weights[:, i].unsqueeze(-1)  # Shape: [batch_size, 1]
            weighted_output = tree_weight * scaled_output

            prediction += weighted_output

        # For classification tasks, apply appropriate activation
        if self.task_type == 'binary_classification':
            prediction = torch.sigmoid(prediction)
        elif self.task_type == 'multiclass_classification':
            prediction = F.softmax(prediction, dim=1)

        return prediction

    def pruning_regularization(self):
        reg_term = 0.0
        for tree in self.trees:
            reg_term += tree.pruning_regularization()
        return self.pruning_reg_weight * reg_term  # Apply pruning regularization weight

    def leaf_regularization(self):
        reg_term = 0.0
        for tree in self.trees:
            reg_term += tree.leaf_regularization()
        return self.leaf_reg_weight * reg_term  # Apply leaf regularization weight

    def temperature_regularization(self):
        temp_reg = 0.0
        for tree in self.trees:
            # Penalize high temperatures to encourage hardening
            temp_reg += tree.temperature ** 2
        return self.temperature_penalty * temp_reg  # Apply temperature regularization weight

    def lasso_regularization(self):
        # L1 regularization on residual weights to encourage sparsity
        return self.lasso_reg_weight * torch.sum(torch.abs(self.residual_weights))

    def diversity_regularization(self, tree_outputs):
        """
        Compute the diversity regularization term.
        Penalizes high correlation between tree outputs.
        """
        # Stack tree outputs: [num_trees, batch_size, output_dim]
        tree_outputs = torch.stack(tree_outputs, dim=0)
        num_trees = tree_outputs.size(0)
        correlations = 0.0

        # Flatten the outputs to [num_trees, batch_size * output_dim]
        tree_outputs_flat = tree_outputs.view(num_trees, -1)

        # Normalize the outputs
        tree_outputs_norm = (tree_outputs_flat - tree_outputs_flat.mean(dim=1, keepdim=True)) / (
            tree_outputs_flat.std(dim=1, keepdim=True) + 1e-8
        )

        # Compute pairwise correlations
        for i in range(num_trees):
            for j in range(i + 1, num_trees):
                corr = torch.dot(tree_outputs_norm[i], tree_outputs_norm[j]) / tree_outputs_norm[i].numel()
                correlations += corr ** 2  # Squared correlation penalty

        return self.diversity_reg_weight * correlations

    def regularization(self, tree_outputs=None):
        reg_loss = (
            self.pruning_regularization() +
            self.temperature_regularization() +
            self.lasso_regularization() +
            self.leaf_regularization()
        )
        if tree_outputs is not None and self.diversity_reg_weight > 0:
            reg_loss += self.diversity_regularization(tree_outputs)
        return reg_loss

    def harden_splits(self, epoch, max_epochs, val_loss=None, prev_val_loss=None, method="cosine"):
        """
        Adjusts tree temperature based on the selected method and backoff mechanism.
        Implements a capacitor-like backoff to modulate hardening based on validation loss trends.
        """
        # Adjust backoff charge based on validation loss trend
        if val_loss is not None and prev_val_loss is not None:
            if val_loss > prev_val_loss:
                # Validation loss worsened, accumulate backoff charge
                self.backoff_charge += self.hardening_adjustment_factor
            else:
                # Validation loss improved or stayed the same, dissipate backoff charge
                self.backoff_charge = max(0.0, self.backoff_charge - self.hardening_backoff_decay_rate)

        # Compute adjustment factor
        adjustment_factor = 1 / (1 + self.backoff_charge)

        # Apply non-linear hardening schedule with adjustment factor
        for tree in self.trees:
            with torch.no_grad():
                if method == "logarithmic":
                    adjustment = math.log(epoch + 1)
                    new_temp = max(self.init_temp - adjustment * adjustment_factor, 0.1)
                elif method == "exponential":
                    rate = 0.05  # Control the decay rate
                    new_temp = max(self.init_temp * math.exp(-rate * epoch) * adjustment_factor, 0.1)
                elif method == "polynomial":
                    p = 2  # Control the polynomial degree
                    new_temp = max(
                        (self.init_temp - 0.1) * (1 - epoch / max_epochs) ** p * adjustment_factor + 0.1, 
                        0.1
                    )
                elif method == "cosine":
                    # Ensure torch.pi is available; use math.pi otherwise
                    pi = torch.pi if hasattr(torch, 'pi') else math.pi
                    cosine_component = math.cos(pi * epoch / max_epochs)
                    new_temp = 0.1 + 0.5 * (self.init_temp - 0.1) * (1 + cosine_component)
                    new_temp *= adjustment_factor
                else:
                    raise ValueError("Unsupported hardening method")

                # Update temperature
                tree.temperature.fill_(new_temp)

    def harden_splits_cosine(self, epoch, max_epochs, val_loss=None, prev_val_loss=None):
        """
        Convenience method for cosine hardening with backoff.
        """
        self.harden_splits(epoch, max_epochs, val_loss, prev_val_loss, method="cosine")

    def harden_splits_logarithmic(self, epoch, max_epochs, val_loss=None, prev_val_loss=None):
        """
        Convenience method for logarithmic hardening with backoff.
        """
        self.harden_splits(epoch, max_epochs, val_loss, prev_val_loss, method="logarithmic")

    # Add other hardening methods as needed

    def harden_splits_exponential(self, epoch, max_epochs, val_loss=None, prev_val_loss=None):
        """
        Convenience method for exponential hardening with backoff.
        """
        self.harden_splits(epoch, max_epochs, val_loss, prev_val_loss, method="exponential")

    def harden_splits_polynomial(self, epoch, max_epochs, val_loss=None, prev_val_loss=None):
        """
        Convenience method for polynomial hardening with backoff.
        """
        self.harden_splits(epoch, max_epochs, val_loss, prev_val_loss, method="polynomial")

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
    use_hessian=False,
    hardening_method="cosine"  # NEW: Select hardening schedule
):
    # Choose optimizer
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
    prev_val_loss = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        tree_outputs = []
        outputs = model(X_train)
        for tree in model.trees:
            tree_outputs.append(tree(X_train))

        # Compute loss based on task type
        if model.task_type == 'multiclass_classification':
            loss = criterion(outputs, y_train.long())
        else:
            loss = criterion(outputs.squeeze(), y_train)

        # Compute regularization with diversity term
        reg_loss = model.regularization(tree_outputs)
        total_loss = loss + reg_lambda * reg_loss

        # Backward pass and optimization
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            if model.task_type == 'multiclass_classification':
                val_loss = criterion(val_outputs, y_val.long())
            else:
                val_loss = criterion(val_outputs.squeeze(), y_val)
            val_loss_value = val_loss.item()

        # Adjust splits using non-linear hardening and backoff mechanism
        if hardening_method == "cosine":
            model.harden_splits_cosine(
                epoch=epoch, 
                max_epochs=epochs, 
                val_loss=val_loss_value, 
                prev_val_loss=prev_val_loss
            )
        elif hardening_method == "logarithmic":
            model.harden_splits_logarithmic(
                epoch=epoch, 
                max_epochs=epochs, 
                val_loss=val_loss_value, 
                prev_val_loss=prev_val_loss
            )
        elif hardening_method == "exponential":
            model.harden_splits_exponential(
                epoch=epoch, 
                max_epochs=epochs, 
                val_loss=val_loss_value, 
                prev_val_loss=prev_val_loss
            )
        elif hardening_method == "polynomial":
            model.harden_splits_polynomial(
                epoch=epoch, 
                max_epochs=epochs, 
                val_loss=val_loss_value, 
                prev_val_loss=prev_val_loss
            )
        else:
            raise ValueError("Unsupported hardening method")

        # Update previous validation loss
        prev_val_loss = val_loss_value

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        # Logging
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss_value:.4f}, "
            f"Reg Loss: {reg_loss.item():.4f}, "
            f"Backoff Charge: {model.backoff_charge:.4f}"
        )

        # Early stopping
        if early_stopping:
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
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
