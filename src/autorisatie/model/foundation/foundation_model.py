import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class LabTestEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(1, d_model)
        self.test_embedding = nn.Embedding(input_dim, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, test_indices):
        # x: [batch_size, seq_len, 1]
        # test_indices: [batch_size, seq_len]
        value_emb = self.value_embedding(x)  # [batch_size, seq_len, d_model]
        test_emb = self.test_embedding(test_indices)  # [batch_size, seq_len, d_model]
        combined = value_emb + test_emb
        combined = combined.transpose(0, 1)  # [seq_len, batch_size, d_model]
        combined = self.position_encoding(combined)
        return self.dropout(combined)

class FoundationModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Embedding layers
        self.embedding = LabTestEmbedding(input_dim, d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layers
        self.output_layer = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x):
        """Create padding mask for transformer."""
        return (x == -1).transpose(0, 1)  # [seq_len, batch_size]
    
    def forward(self, x, test_indices):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, 1]
            test_indices: Test indices tensor of shape [batch_size, seq_len]
        """
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Embed inputs
        embedded = self.embedding(x, test_indices)  # [seq_len, batch_size, d_model]
        
        # Transformer encoding
        memory = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Transformer decoding (using memory as both source and target for autoencoder)
        output = self.transformer_decoder(
            memory,
            memory,
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=padding_mask
        )
        
        # Project to output
        output = self.output_layer(output)  # [seq_len, batch_size, 1]
        output = output.transpose(0, 1)  # [batch_size, seq_len, 1]
        
        return output
    
    def encode(self, x, test_indices):
        """Get the encoded representation of the input."""
        padding_mask = self.create_padding_mask(x)
        embedded = self.embedding(x, test_indices)
        return self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
    
    def decode(self, z, test_indices):
        """Reconstruct the input from the encoded representation."""
        padding_mask = self.create_padding_mask(z)
        output = self.transformer_decoder(
            z,
            z,
            memory_key_padding_mask=padding_mask,
            tgt_key_padding_mask=padding_mask
        )
        return self.output_layer(output).transpose(0, 1)
    
    def predict_next_values(self, x, test_indices, num_predictions=1):
        """
        Predict future values based on current sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, 1]
            test_indices: Test indices tensor of shape [batch_size, seq_len]
            num_predictions: Number of future values to predict
            
        Returns:
            Predicted values of shape [batch_size, num_predictions, 1]
        """
        # Get encoded representation
        encoded = self.encode(x, test_indices)
        
        # Initialize predictions
        predictions = []
        current_input = x
        current_indices = test_indices
        
        for _ in range(num_predictions):
            # Decode current sequence
            decoded = self.decode(encoded, current_indices)
            
            # Get last prediction
            last_pred = decoded[:, -1:, :]
            predictions.append(last_pred)
            
            # Update input for next prediction
            current_input = torch.cat([current_input, last_pred], dim=1)
            current_indices = torch.cat([
                current_indices,
                torch.ones((current_indices.shape[0], 1), dtype=torch.long, device=current_indices.device)
            ], dim=1)
            
            # Re-encode with new input
            encoded = self.encode(current_input, current_indices)
        
        return torch.cat(predictions, dim=1) 