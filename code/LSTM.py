import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, sensor_embed_dim=60, hidden_size=64, num_layers=2, num_classes=4):
        """
        sensor_embed_dim: Output dimension after projecting each sensor's original 5D features
                         (must be divisible by 3 for multi-head attention)
        hidden_size:     Hidden dimension of the LSTM
        num_layers:      Number of LSTM layers
        num_classes:     Number of output classes
        """
        super(BiLSTMModel, self).__init__()
        self.sensor_embed_dim = sensor_embed_dim

        # Project the 5D sensor input into sensor_embed_dim
        self.sensor_embed = nn.Linear(5, sensor_embed_dim)

        # Self-attention across 4 sensors using 3 attention heads
        self.sensor_attn = nn.MultiheadAttention(
            embed_dim=sensor_embed_dim,
            num_heads=3,
            batch_first=True
        )

        # After attention, each timestep becomes [4, sensor_embed_dim]
        # Flatten to shape [batch, seq_len, 4 * sensor_embed_dim]
        lstm_input_dim = 4 * sensor_embed_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Classification layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Input x shape: [batch_size, seq_len, num_sensors, feature_dim] → e.g., [32, 20, 4, 5]
        """
        bs, seq_len, n, d_prime = x.shape  # n = 4 sensors, d' = 5 features

        # Step 1: Flatten and project sensor features
        x = x.reshape(bs * seq_len * n, d_prime)  # [bs*seq_len*n, 5]
        x = self.sensor_embed(x)  # [bs*seq_len*n, sensor_embed_dim]
        x = x.reshape(bs, seq_len, n, self.sensor_embed_dim)  # [bs, seq_len, n, embed_dim]

        # Step 2: Apply self-attention across sensors at each timestep
        x = x.reshape(bs * seq_len, n, self.sensor_embed_dim)  # [bs*seq_len, n, embed_dim]
        attn_output, _ = self.sensor_attn(x, x, x)  # [bs*seq_len, n, embed_dim]
        attn_output = attn_output.reshape(bs, seq_len, n, self.sensor_embed_dim)  # restore shape

        # Step 3: Flatten sensor dimension
        attn_output = attn_output.reshape(bs, seq_len, n * self.sensor_embed_dim)  # [bs, seq_len, input_dim]

        # Step 4: Feed into LSTM
        lstm_out, _ = self.lstm(attn_output)  # [bs, seq_len, hidden_size]

        # Step 5: Temporal pooling (mean over time steps)
        pooled = torch.mean(lstm_out, dim=1)  # [bs, hidden_size]

        # Step 6: Final classification
        logits = self.fc(pooled)  # [bs, num_classes]
        return logits


# Example usage
if __name__ == '__main__':
    model = BiLSTMModel()
    # Simulated input: batch_size=32, sequence_length=20, 4 sensors, 5 features per sensor
    x = torch.randn(32, 20, 4, 5)
    logits = model(x)
    print("Logits shape:", logits.shape)  # Expected: [32, 4]

