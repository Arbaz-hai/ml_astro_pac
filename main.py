import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from train import train_model, predict
from dataset import AstroTimeSeriesDataset
from model import AstroClassifier


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)



# Main execution function
def main():
    # Load data
    meta_data = pd.read_csv('training_set_metadata.csv')
    training_set = pd.read_csv("training_set.csv")

    
    targets = np.hstack([np.unique(meta_data['target']), [99]])
    target_map = {j:i for i, j in enumerate(targets)}
    target_ids = [target_map[i] for i in meta_data['target']]
    meta_data['target_id'] = target_ids
    
    # Check if we're using target_id instead of target for classification
    target_column = 'target_id'
    num_classes = meta_data[target_column].nunique()
    print(f"Using {target_column} for classification with {num_classes} classes")

    
    # Parameters
    max_seq_length = 300
    batch_size = 64
    num_epochs = 100
    initial_lr = 0.001
    hidden_dim = 128
    num_layers = 2
    dropout = 0.4
    mixup_alpha = 0.2  # Mixup augmentation parameter
    
    # Get unique object IDs
    train_object_ids = meta_data.object_id.values
    
    # Split data
    train_ids, val_ids = train_test_split(
        train_object_ids, 
        test_size=0.2, 
        random_state=42, 
        stratify=meta_data[target_column]
    )
    # stratify -> Ensure that the distribution of classes is the same in both training and validation sets
    # Calculate metadata dimension
    metadata_dim = len(['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'mwebv'])
    if 'hostgal_photoz' in meta_data.columns:
        metadata_dim += 2  # hostgal_photoz and hostgal_photoz_err
    if 'distmod' in meta_data.columns:
        metadata_dim += 1
    
    # Create datasets
    train_dataset = AstroTimeSeriesDataset(train_ids, training_set, meta_data, max_seq_length)
    val_dataset = AstroTimeSeriesDataset(val_ids, training_set, meta_data, max_seq_length)
    test_dataset = AstroTimeSeriesDataset(val_ids, training_set, meta_data, 
                                         max_seq_length, is_test=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    input_dim = 4  # [time, flux, flux_err, detected]
    
    model = AstroClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        metadata_dim=metadata_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    )
    
    # Define loss function and optimizer
    # Use label smoothing to improve stability
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # label_smoothing -> 
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train the model
    model, train_losses, val_losses, val_accuracies = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=num_epochs,
        patience=10,
        mixup_alpha=mixup_alpha
    )
    
    # Make predictions on test set
    predictions, test_obj_ids = predict(model, test_loader, device)
    
    # Create submission dataframe with class probabilities
    submission = pd.DataFrame({
        'object_id': test_obj_ids
    })
    
    # Add probability for each class
    for i in range(num_classes):
        submission[f'class_{i}_prob'] = predictions[:, i]
    
    # Add predicted class (argmax)
    submission['predicted_class'] = np.argmax(predictions, axis=1)
    
    # Save submission
    submission.to_csv('submission_with_probabilities.csv', index=False)
    
    print("Prediction complete. Results saved to 'submission_with_probabilities.csv'")

if __name__ == "__main__":
    main()