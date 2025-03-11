import torch
import torch.nn as nn

# Define the model architecture
class AstroClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, metadata_dim=6, num_layers=2, dropout=0.3, num_classes=14):
        super(AstroClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_passbands = 6
        
        # LSTM for each passband -> parameters shared across passbands
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Metadata processing
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        
        # Fully connected layers
        combined_dim = hidden_dim * 2 * self.num_passbands + 64  # LSTM + metadata
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout/2),  # Less dropout in later layers
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout/2)
        )
        
        # Output layer
        self.output = nn.Linear(64, num_classes)
        
    def apply_attention(self, lstm_output, mask):
        # lstm_output shape: (batch_size, seq_len, hidden_dim*2)
        # mask shape: (batch_size, seq_len)
        
        # Calculate attention scores
        attn_scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        
        # Apply mask: set attention scores to -inf for padded positions
        mask = mask.unsqueeze(2)  # (batch_size, seq_len, 1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights to LSTM outputs
        context = torch.bmm(lstm_output.transpose(1, 2), attn_weights)  # (batch_size, hidden_dim*2, 1)
        context = context.squeeze(2)  # (batch_size, hidden_dim*2)
        
        return context
    
    def forward(self, x, mask, metadata):
        # x shape: (batch_size, num_passbands, seq_len, input_dim)
        # mask shape: (batch_size, num_passbands, seq_len)
        # metadata shape: (batch_size, metadata_dim)
        batch_size = x.size(0)
        
        # Process metadata
        metadata_features = self.metadata_fc(metadata)  # (batch_size, 64)
        
        # Process each passband separately
        passband_contexts = []
        for i in range(self.num_passbands):
            # Get data for this passband
            passband_data = x[:, i, :, :]  # (batch_size, seq_len, input_dim)
            passband_mask = mask[:, i, :]  # (batch_size, seq_len)
            
            # Pass through LSTM
            lstm_out, _ = self.lstm(passband_data)  # (batch_size, seq_len, hidden_dim*2)
            
            # Apply attention with mask
            context = self.apply_attention(lstm_out, passband_mask)  # (batch_size, hidden_dim*2)
            passband_contexts.append(context)
        
        # Concatenate contexts from all passbands
        combined = torch.cat(passband_contexts, dim=1)  # (batch_size, num_passbands*hidden_dim*2)
        
        # Concatenate with metadata features
        combined = torch.cat([combined, metadata_features], dim=1)
        
        # Fully connected layers
        x = self.fc_layers(combined)
        
        # Output layer - returns logits
        logits = self.output(x)
         
        return logits
