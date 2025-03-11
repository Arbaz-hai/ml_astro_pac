
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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
                # flux_mad is the median absolute deviation
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
                sequence = np.zeros((0, 4)) # time, flux, flux_err, detected -> we know nothing so we put 0
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



if __name__ == '__main__':
    import pandas as pd
    meta_data = pd.read_csv('training_set_metadata.csv')
    training_set = pd.read_csv("training_set.csv")

    
    targets = np.hstack([np.unique(meta_data['target']), [99]])
    target_map = {j:i for i, j in enumerate(targets)}
    target_ids = [target_map[i] for i in meta_data['target']]
    meta_data['target_id'] = target_ids

    data = AstroTimeSeriesDataset(meta_data.object_id.values, training_set, meta_data, max_seq_length=300)
    print(len(data))
    # describe data[0]

    print(len(data[0]))
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[0][2].shape)
    print(data[0][3].shape)
    print("label", data[0][3])  # label
    print("metadata", data[0][2])  # metadata
    print("first 5 time steps of first passband", data[0][0][0, :5])  # first 5 time steps of first passband
    print("first 5 masks of first passband", data[0][1][0, :5])  # first 5 masks of first passband
    print("-"*100)
    print("last 5 time steps of last passband", data[0][0][-1, -5:])  # last 5 time steps of last passband
    print("last 5 masks of last passband", data[0][1][-1, -5:])  # last 5 masks of last passband
    print("-"*100)
    print("Seq denotes ")
    # time, flux, flux_err, detected
    print(data[0][0][0,0]) # -> time, flux, flux_err, detected
    