import torch
import torch.nn as nn

class AstroClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, metadata_dim=6, num_layers=2, dropout=0.3, num_classes=14, num_heads=4):
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
        
        # Replace custom attention with MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Metadata processing (unchanged)
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        
        # Rest of the model remains unchanged
        combined_dim = hidden_dim * 2 * self.num_passbands + 64
        
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
        
        self.output = nn.Linear(64, num_classes)
        
    def apply_attention(self, lstm_output, mask):
        # Convert mask to attention mask format (False where padded, True elsewhere)
        attn_mask = mask == 1
        
        # Apply multihead attention
        # In self-attention, query, key, and value are all the same
        context, _ = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=~attn_mask  # PyTorch expects True for padding positions
        )
        
        # Pool over sequence dimension to get a fixed-size representation
        # You could use mean pooling over non-padded elements:
        expanded_mask = mask.unsqueeze(-1).expand_as(context)
        masked_sum = (context * expanded_mask).sum(dim=1)
        context = masked_sum / mask.sum(dim=1, keepdim=True)
        
        return context
    
    def forward(self, x, mask, metadata):
        # Code unchanged from your original implementation
        batch_size = x.size(0)
        
        metadata_features = self.metadata_fc(metadata)
        
        passband_contexts = []
        for i in range(self.num_passbands):
            passband_data = x[:, i, :, :]
            passband_mask = mask[:, i, :]
            
            lstm_out, _ = self.lstm(passband_data)
            
            context = self.apply_attention(lstm_out, passband_mask)
            passband_contexts.append(context)
        
        combined = torch.cat(passband_contexts, dim=1)
        combined = torch.cat([combined, metadata_features], dim=1)
        
        x = self.fc_layers(combined)
        logits = self.output(x)
         
        return logits