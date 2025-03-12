import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(67)

# Custom dataset class for time series data with metadata
class AstroTimeSeriesDataset(Dataset):
    def __init__(self, object_ids, flux_data, metadata, max_seq_length=150, is_test=False):
        self.object_ids = object_ids
        self.flux_data = flux_data
        self.metadata = metadata
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        
        # Initialize scaler for metadata
        self.metadata_features = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'mwebv']
        
        # Add hostgal features if they exist in the metadata
        if 'hostgal_photoz' in metadata.columns:
            self.metadata_features.extend(['hostgal_photoz', 'hostgal_photoz_err'])
        
        if 'distmod' in metadata.columns:
            # Fill NaN values in distmod with median
            median_distmod = metadata['distmod'].median()
            self.metadata['distmod'] = metadata['distmod'].fillna(median_distmod)
            self.metadata_features.append('distmod')
            
        # Scale metadata features
        self.metadata_scaler = StandardScaler()
        self.metadata[self.metadata_features] = self.metadata_scaler.fit_transform(
            self.metadata[self.metadata_features].fillna(0)
        )
        
    def __len__(self):
        return len(self.object_ids)
    
    def __getitem__(self, idx):
        obj_id = self.object_ids[idx]
        
        # Get data for this object
        obj_data = self.flux_data[self.flux_data.object_id == obj_id]
        obj_metadata = self.metadata[self.metadata.object_id == obj_id].iloc[0]
        
        # Extract metadata features
        metadata_tensor = torch.tensor(
            obj_metadata[self.metadata_features].values, 
            dtype=torch.float32
        )
        
        # Initialize sequence tensors for each passband
        sequences = []
        masks = []  # Masks to indicate which time steps have data
        
        # Process each passband
        for passband in range(6):
            band_data = obj_data[obj_data.passband == passband]
            
            if len(band_data) > 0:
                # Sort by time
                band_data = band_data.sort_values('mjd')
                
                # Normalize time within this passband
                min_mjd = band_data.mjd.min()
                max_mjd = band_data.mjd.max()
                if max_mjd > min_mjd:
                    band_data['mjd_norm'] = (band_data.mjd - min_mjd) / (max_mjd - min_mjd)
                else:
                    band_data['mjd_norm'] = 0.0
                
                # Create sequence [time, flux, flux_err, detected]
                sequence = band_data[['mjd_norm', 'flux', 'flux_err', 'detected']].values
                
                # Create robust normalization of flux and flux_err
                # Use median absolute deviation for robustness
                flux_median = np.median(sequence[:, 1])
                flux_mad = np.median(np.abs(sequence[:, 1] - flux_median)) * 1.4826  # Scale factor for normal distribution
                if flux_mad > 0:
                    sequence[:, 1] = (sequence[:, 1] - flux_median) / (flux_mad + 1e-8)
                
                flux_err_median = np.median(sequence[:, 2])
                flux_err_mad = np.median(np.abs(sequence[:, 2] - flux_err_median)) * 1.4826
                if flux_err_mad > 0:
                    sequence[:, 2] = (sequence[:, 2] - flux_err_median) / (flux_err_mad + 1e-8)
                
                # Clip extreme values
                sequence[:, 1] = np.clip(sequence[:, 1], -10, 10)
                sequence[:, 2] = np.clip(sequence[:, 2], -10, 10)
                
                # Create mask (1 for real data, 0 for padding)
                mask = np.ones(len(sequence))
            else:
                # Empty sequence if no data
                sequence = np.zeros((0, 4))
                mask = np.zeros(0)
            
            # Pad or truncate sequence to max_seq_length
            if len(sequence) > self.max_seq_length:
                # Subsample if too long
                indices = np.linspace(0, len(sequence)-1, self.max_seq_length).astype(int)
                sequence = sequence[indices]
                mask = mask[indices]
            elif len(sequence) < self.max_seq_length:
                # Pad with zeros if too short
                padding = np.zeros((self.max_seq_length - len(sequence), 4))
                sequence = np.vstack([sequence, padding]) if len(sequence) > 0 else padding
                
                # Pad mask with zeros
                mask_padding = np.zeros(self.max_seq_length - len(mask))
                mask = np.concatenate([mask, mask_padding]) if len(mask) > 0 else mask_padding
            
            sequences.append(sequence)
            masks.append(mask)
        
        # Stack all passbands together
        sequence_tensor = torch.tensor(np.stack(sequences), dtype=torch.float32)
        mask_tensor = torch.tensor(np.stack(masks), dtype=torch.float32)
        
        # If test set, return object_id instead of label
        if self.is_test:
            return sequence_tensor, mask_tensor, metadata_tensor, obj_id
        else:
            # Get label (target_id instead of target)
            label = obj_metadata['target_id']
            return sequence_tensor, mask_tensor, metadata_tensor, torch.tensor(label, dtype=torch.long)

# Define the model architecture
import torch
import torch.nn as nn

class AstroClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, metadata_dim=6, dropout=0.3, num_classes=14, num_heads=4, num_transformer_layers=2):
        super(AstroClassifier, self).__init__()
        
        self.num_passbands = 6
        
        # Input projection to increase dimensionality
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding can be learned or fixed
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Assuming max sequence length of 1000
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # Metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        
        # Fully connected layers
        combined_dim = hidden_dim * self.num_passbands + 64  # Transformer output + metadata
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout/2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout/2)
        )
        
        # Output layer
        self.output = nn.Linear(64, num_classes)
        
    def process_sequence(self, x, mask):
        # x shape: (batch_size, seq_len, input_dim)
        # mask shape: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Project input to higher dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Create padding mask for transformer (True values are masked positions)
        padding_mask = ~mask.bool()  # (batch_size, seq_len)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Get sequence representation (mean of non-padded elements)
        expanded_mask = mask.unsqueeze(-1).expand_as(transformer_output)
        masked_sum = (transformer_output * expanded_mask).sum(dim=1)
        seq_repr = masked_sum / mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        
        return seq_repr
    
    def forward(self, x, mask, metadata):
        # x shape: (batch_size, num_passbands, seq_len, input_dim)
        # mask shape: (batch_size, num_passbands, seq_len)
        # metadata shape: (batch_size, metadata_dim)
        
        # Process metadata
        metadata_features = self.metadata_fc(metadata)  # (batch_size, 64)
        
        # Process each passband separately
        passband_representations = []
        for i in range(self.num_passbands):
            # Get data for this passband
            passband_data = x[:, i, :, :]  # (batch_size, seq_len, input_dim)
            passband_mask = mask[:, i, :]  # (batch_size, seq_len)
            
            # Process sequence with transformer
            seq_repr = self.process_sequence(passband_data, passband_mask)  # (batch_size, hidden_dim)
            passband_representations.append(seq_repr)
        
        # Concatenate representations from all passbands
        combined = torch.cat(passband_representations, dim=1)  # (batch_size, num_passbands*hidden_dim)
        
        # Concatenate with metadata features
        combined = torch.cat([combined, metadata_features], dim=1)
        
        # Fully connected layers
        x = self.fc_layers(combined)
        
        # Output layer
        logits = self.output(x)
         
        return logits
# Learning rate scheduler with warmup
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Mixup data augmentation
def mixup_data(x, mask, metadata, y, alpha=0.2):
    '''Returns mixed inputs, masks, metadata, targets'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_mask = mask  # Keep original mask
    mixed_metadata = lam * metadata + (1 - lam) * metadata[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, mixed_mask, mixed_metadata, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=50, patience=15, mixup_alpha=0.2):
    model.to(device)
    
    # Initialize learning rate scheduler
    lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=num_epochs)
    
    # For early stopping
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Set learning rate
        current_lr = lr_scheduler.step(epoch)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, masks, metadata, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, masks, metadata, labels = (
                inputs.to(device), 
                masks.to(device), 
                metadata.to(device), 
                labels.to(device)
            )
            
            # Apply mixup with probability 0.5
            if mixup_alpha > 0 and np.random.random() < 0.5:
                inputs, masks, metadata, targets_a, targets_b, lam = mixup_data(
                    inputs, masks, metadata, labels, mixup_alpha
                )
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs, masks, metadata)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs, masks, metadata)
                loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, masks, metadata, labels in val_loader:
                inputs, masks, metadata, labels = (
                    inputs.to(device), 
                    masks.to(device), 
                    metadata.to(device), 
                    labels.to(device)
                )
                
                outputs = model(inputs, masks, metadata)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'LR: {current_lr:.6f}, '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_accuracy': val_accuracy
            }, 'best_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, train_losses, val_losses, val_accuracies

# Function to make predictions
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    object_ids = []
    
    with torch.no_grad():
        for inputs, masks, metadata, obj_id in tqdm(test_loader, desc='Predicting'):
            inputs = inputs.to(device)
            masks = masks.to(device)
            metadata = metadata.to(device)
            
            outputs = model(inputs, masks, metadata)
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Save predictions and object IDs
            predictions.append(probs.cpu().numpy())
            if isinstance(obj_id, torch.Tensor):
                object_ids.append(obj_id.numpy())
            else:
                object_ids.append(np.array([obj_id]))
    
    # Concatenate all predictions and object IDs
    predictions = np.vstack(predictions)
    object_ids = np.concatenate(object_ids)
    
    return predictions, object_ids

# Main execution function
def main():
    # Load data
    meta_data = pd.read_csv('/kaggle/input/ml-astro/training_set_metadata.csv')
    training_set = pd.read_csv("/kaggle/input/ml-astro/training_set.csv")
    targets = np.hstack([np.unique(meta_data['target']), [99]])
    target_map = {j:i for i, j in enumerate(targets)}
    target_ids = [target_map[i] for i in meta_data['target']]
    meta_data['target_id'] = target_ids
    
    # Check if we're using target_id instead of target for classification
    if 'target_id' in meta_data.columns:
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
    mixup_alpha = 0.0  # Mixup augmentation parameter
    
    # Get unique object IDs
    train_object_ids = meta_data.object_id.values
    
    # Split data
    train_ids, val_ids = train_test_split(
        train_object_ids, 
        test_size=0.2, 
        random_state=42, 
        stratify=meta_data[target_column]
    )
    
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
        patience=15,
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