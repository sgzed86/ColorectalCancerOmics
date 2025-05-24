import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the GNN model
class GCN(nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.batch_norm1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.batch_norm2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.batch_norm3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.2, training=self.training)
        
        x = self.fc(x3)
        return x

def load_and_preprocess_data(train_file, test_file):
    # Load both datasets
    print("Loading datasets...")
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Get common columns between both datasets
    common_cols = list(set(train_df.columns) & set(test_df.columns))
    print(f"Common features between datasets: {len(common_cols)}")
    
    # Ensure both datasets have the same features
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]
    
    # Separate features and target for both datasets
    y_train = train_df['recurrence_target'].values
    y_test = test_df['recurrence_target'].values
    
    X_train = train_df.drop(['recurrence_target', 'Patient_ID'], axis=1, errors='ignore')
    X_test = test_df.drop(['recurrence_target', 'Patient_ID'], axis=1, errors='ignore')
    
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True).round(3))
    print("\nClass distribution in test set:")
    print(pd.Series(y_test).value_counts(normalize=True).round(3))
    
    # Handle non-numeric values
    print("\nCleaning data...")
    # Replace non-numeric values with NaN
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    
    # Count missing values in training set
    missing_cols_train = X_train.isnull().sum()
    print(f"\nColumns with missing values in training set: {sum(missing_cols_train > 0)}")
    
    # Remove columns with too many missing values (>50%) from both sets
    too_many_missing = missing_cols_train[missing_cols_train > len(X_train) * 0.5].index
    X_train = X_train.drop(columns=too_many_missing)
    X_test = X_test.drop(columns=too_many_missing)
    print(f"Removed {len(too_many_missing)} columns with >50% missing values")
    
    # Replace infinite values with NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Compute statistics on training data
    train_medians = X_train.median()
    train_99th = X_train.quantile(0.99)
    train_1st = X_train.quantile(0.01)
    
    # Fill missing values with training medians
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    
    # Handle extreme values using training set statistics
    for column in X_train.columns:
        upper_limit = train_99th[column]
        lower_limit = train_1st[column]
        X_train[column] = X_train[column].clip(lower=lower_limit, upper=upper_limit)
        X_test[column] = X_test[column].clip(lower=lower_limit, upper=upper_limit)
    
    # Scale the features using training data statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nFinal dataset shapes:")
    print(f"Training: {X_train_scaled.shape}")
    print(f"Testing: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()

def create_graph_structure(X, threshold=0.85):  # Increased threshold for sparser graph
    # Calculate feature correlations
    print("Calculating feature correlations...")
    corr_matrix = np.corrcoef(X.T)
    
    # Create edges based on correlation threshold
    print(f"Creating edges with correlation threshold {threshold}...")
    edges = []
    n_nodes = X.shape[0]  # Number of samples
    
    # Create edges between samples based on feature similarity
    for i in range(n_nodes):
        # Calculate similarity between samples using batched operations
        if i % 50 == 0:  # Progress indicator
            print(f"Processing node {i}/{n_nodes}")
        similarities = np.corrcoef(X[i:i+1], X)[0, 1:]
        # Connect to top k most similar samples
        k = min(5, n_nodes - 1)  # Connect to at most 5 neighbors
        top_k_indices = np.argsort(similarities)[-k:]
        for j in top_k_indices:
            if i != j:
                edges.append([i, j])
                edges.append([j, i])
    
    if not edges:
        print("Warning: No edges created. Creating minimal spanning tree...")
        # Create minimal connections
        for i in range(n_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    print(f"Created graph with {len(edges)//2} edges")
    return edge_index

def train_gnn(model, data, optimizer, epochs=150):
    model.train()
    best_loss = float('inf')
    patience = 20
    counter = 0
    
    # Calculate class weights - increase weight for positive class
    pos_weight = torch.tensor([10.0])  # Much higher weight for positive class
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Use BCEWithLogitsLoss for better numerical stability
        loss = F.binary_cross_entropy_with_logits(
            out.squeeze(), 
            data.y.float(),
            pos_weight=pos_weight
        )
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')

def evaluate_model(model, data, y_true, threshold=0.2):  # Lower threshold further
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        # Apply sigmoid here since we're using BCEWithLogitsLoss
        pred_proba = torch.sigmoid(pred.squeeze()).numpy()
        pred_class = (pred_proba > threshold).astype(int)
        
        accuracy = accuracy_score(y_true, pred_class)
        auc = roc_auc_score(y_true, pred_proba)
        conf_matrix = confusion_matrix(y_true, pred_class)
        
        # Calculate metrics at different thresholds
        thresholds = np.arange(0.05, 0.5, 0.05)  # Lower threshold range
        best_f1 = 0
        best_threshold = threshold
        threshold_results = []
        
        for t in thresholds:
            pred_t = (pred_proba > t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred_t).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
            
            threshold_results.append({
                'threshold': t,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'f1': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        print("\nThreshold Analysis:")
        for result in threshold_results:
            print(f"Threshold {result['threshold']:.2f}: "
                  f"Sensitivity={result['sensitivity']:.3f}, "
                  f"Specificity={result['specificity']:.3f}, "
                  f"PPV={result['ppv']:.3f}, "
                  f"F1={result['f1']:.3f}")
        
        print(f"\nBest threshold (F1): {best_threshold:.2f}")
        
        # Use best threshold for final predictions
        pred_class = (pred_proba > best_threshold).astype(int)
        conf_matrix = confusion_matrix(y_true, pred_class)
        
        return accuracy, auc, conf_matrix, pred_proba

def plot_results(conf_matrix, y_true, pred_proba):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('gnn_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('gnn_roc_curve.png')
    plt.close()

def cross_validate_gnn(X, y, n_splits=5):
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    cv_results = {
        'accuracy': [],
        'auc': [],
        'sensitivity': [],
        'specificity': [],
        'ppv': [],
        'f1': []
    }
    
    fold = 1
    for train_idx, val_idx in kf.split(X):
        print(f"\nFold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create graph structures
        edge_index_train = create_graph_structure(X_train)
        edge_index_val = create_graph_structure(X_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        
        # Create PyTorch Geometric Data objects
        train_data = Data(x=X_train_tensor, edge_index=edge_index_train, y=y_train_tensor)
        val_data = Data(x=X_val_tensor, edge_index=edge_index_val)
        
        # Initialize and train model
        model = GCN(num_features=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.005)
        train_gnn(model, train_data, optimizer)
        
        # Evaluate
        accuracy, auc, conf_matrix, pred_proba = evaluate_model(model, val_data, y_val)
        
        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        # Store results
        cv_results['accuracy'].append(accuracy)
        cv_results['auc'].append(auc)
        cv_results['sensitivity'].append(sensitivity)
        cv_results['specificity'].append(specificity)
        cv_results['ppv'].append(ppv)
        cv_results['f1'].append(f1)
        
        print(f"Fold {fold} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"PPV: {ppv:.4f}")
        print(f"F1: {f1:.4f}")
        
        fold += 1
    
    return cv_results

def plot_cv_results(cv_results):
    # Calculate mean and std for each metric
    metrics_summary = {}
    for metric, values in cv_results.items():
        metrics_summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Plot results
    plt.figure(figsize=(12, 6))
    metrics = list(cv_results.keys())
    means = [metrics_summary[m]['mean'] for m in metrics]
    stds = [metrics_summary[m]['std'] for m in metrics]
    
    # Create bar plot
    bars = plt.bar(metrics, means, yerr=stds, capsize=5)
    plt.title('Cross-validation Results')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cross_validation_results.png')
    plt.close()
    
    # Print summary statistics
    print("\nCross-validation Summary:")
    for metric, stats in metrics_summary.items():
        print(f"{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")

def analyze_feature_importance(model, data, feature_names):
    model.eval()
    importances = []
    
    # Register hook for gradient computation
    feature_gradients = []
    def hook_fn(grad):
        feature_gradients.append(grad.detach().cpu().numpy())
    
    data.x.requires_grad = True
    data.x.register_hook(hook_fn)
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute gradients
    out.sum().backward()
    
    # Get feature gradients
    gradients = feature_gradients[0]
    
    # Compute importance scores (gradient * input)
    importance_scores = np.abs(gradients * data.x.detach().cpu().numpy())
    
    # Average importance across samples
    mean_importance = np.mean(importance_scores, axis=0)
    
    # Create feature importance dictionary
    feature_importance = dict(zip(feature_names, mean_importance))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_features

def plot_feature_importance(feature_importance, top_n=20):
    # Plot top N most important features
    plt.figure(figsize=(12, 8))
    
    features = [x[0] for x in feature_importance[:top_n]]
    scores = [x[1] for x in feature_importance[:top_n]]
    
    # Create bar plot
    bars = plt.bar(range(len(features)), scores)
    
    # Customize plot
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Save feature importance to file
    with open('feature_importance.tsv', 'w') as f:
        f.write("Feature\tImportance Score\n")
        for feature, score in feature_importance:
            f.write(f"{feature}\t{score}\n")

def analyze_network_structure(edge_index, feature_names, threshold=0.85):
    # Create adjacency matrix from edge index
    n_features = len(feature_names)
    adj_matrix = np.zeros((n_features, n_features))
    
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        adj_matrix[src, dst] = 1
    
    # Calculate node degree
    node_degrees = np.sum(adj_matrix, axis=0)
    
    # Create network metrics dictionary
    network_metrics = dict(zip(feature_names, node_degrees))
    
    # Sort features by degree centrality
    sorted_metrics = sorted(network_metrics.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_metrics

def plot_network_metrics(network_metrics, top_n=20):
    plt.figure(figsize=(12, 8))
    
    features = [x[0] for x in network_metrics[:top_n]]
    scores = [x[1] for x in network_metrics[:top_n]]
    
    # Create bar plot
    bars = plt.bar(range(len(features)), scores)
    
    # Customize plot
    plt.title(f'Top {top_n} Most Connected Features')
    plt.xlabel('Features')
    plt.ylabel('Number of Connections')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('network_structure.png')
    plt.close()
    
    # Save network metrics to file
    with open('network_metrics.tsv', 'w') as f:
        f.write("Feature\tNumber of Connections\n")
        for feature, score in network_metrics:
            f.write(f"{feature}\t{int(score)}\n")

def main():
    print("Loading data...")
    torch.manual_seed(42)
    
    # Load and preprocess both datasets
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
        'balanced_recurrence_dataframe.tsv',
        'joined_recurrence_dataframe.tsv'
    )
    
    # Perform cross-validation
    cv_results = cross_validate_gnn(X_train, y_train, n_splits=5)
    plot_cv_results(cv_results)
    
    print("\nTraining final model on full training set...")
    print("\nCreating graph structures...")
    edge_index_train = create_graph_structure(X_train)
    edge_index_test = create_graph_structure(X_test)
    
    # Analyze network structure
    print("\nAnalyzing network structure...")
    network_metrics = analyze_network_structure(edge_index_train, feature_names)
    plot_network_metrics(network_metrics)
    print("Network analysis saved to network_structure.png and network_metrics.tsv")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create PyTorch Geometric Data objects
    train_data = Data(x=X_train_tensor, edge_index=edge_index_train, y=y_train_tensor)
    test_data = Data(x=X_test_tensor, edge_index=edge_index_test)
    
    # Initialize model and optimizer
    print("\nInitializing model...")
    model = GCN(num_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.005)
    
    # Train the model
    print("\nTraining model...")
    train_gnn(model, train_data, optimizer)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(model, train_data, feature_names)
    plot_feature_importance(feature_importance)
    print("Feature importance analysis saved to feature_importance.png and feature_importance.tsv")
    
    # Print top 10 most important features
    print("\nTop 10 Most Important Features:")
    for feature, importance in feature_importance[:10]:
        print(f"{feature}: {importance:.4f}")
    
    # Print top 10 most connected features
    print("\nTop 10 Most Connected Features:")
    for feature, connections in network_metrics[:10]:
        print(f"{feature}: {int(connections)} connections")
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    accuracy, auc, conf_matrix, pred_proba = evaluate_model(model, test_data, y_test)
    
    print(f'\nTest Set Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC-ROC: {auc:.4f}')
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    print("\nDetailed Metrics:")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Positive Predictive Value: {ppv:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(conf_matrix, y_test, pred_proba)
    print("Analysis complete! Check the following files for results:")
    print("- gnn_confusion_matrix.png")
    print("- gnn_roc_curve.png")
    print("- cross_validation_results.png")
    print("- feature_importance.png")
    print("- feature_importance.tsv")
    print("- network_structure.png")
    print("- network_metrics.tsv")

if __name__ == '__main__':
    main() 