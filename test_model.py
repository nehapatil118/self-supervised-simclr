import torch
from simclr_model import SimCLR

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SimCLR model
model = SimCLR(feature_dim=128).to(device)

# Create dummy input batch (8 images, 3x32x32)
dummy_input = torch.randn(8, 3, 32, 32).to(device)

# Forward pass
output = model(dummy_input)

# Print output details
print("Output shape:", output.shape)          # Expected: (8, 128)
print("Norm (should be ~1):", output.norm(dim=1))  # Check L2 normalization
