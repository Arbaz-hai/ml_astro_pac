import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils import WarmupCosineScheduler, mixup_data, mixup_criterion



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
