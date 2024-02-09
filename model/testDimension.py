import torch
from myvit import *

# Initialize the model
input_shape = (3, 224, 224)  # Example input shape (C, H, W)
n_patches = 7
hidden_d = 64
n_heads = 2
out_d = 10
model = MyViT(input_shape=input_shape, n_patches=n_patches, hidden_d=hidden_d, n_heads=n_heads, out_d=out_d)

# Create a dummy input tensor
batch_size = 1  # Example batch size
dummy_input = torch.randn(batch_size, *input_shape)  # (batch_size, C, H, W)

# Forward pass to get the output
output = model(dummy_input)

# Print the output shape
print(f"Output shape: {output.shape}")
