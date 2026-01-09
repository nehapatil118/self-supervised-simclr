import torch
import torchvision
from torch.utils.data import DataLoader
from simclr_model import SimCLR
from augmentations import get_simclr_transform
from utils import nt_xent_loss


# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
epochs = 50
lr = 3e-4

# Dataset & data loader
transform = get_simclr_transform()
dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=None
)

# Custom collate function (no transform here)
def collate_fn(batch):
    images, labels = zip(*batch)
    return images, labels


loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=collate_fn
)


# Initialize SimCLR model and optimizer
model = SimCLR(feature_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    total_loss = 0

    for images, _ in loader:

        # Create two augmented views of each image
        images_1 = torch.stack([transform(img) for img in images])
        images_2 = torch.stack([transform(img) for img in images])

        images_1 = images_1.to(device)
        images_2 = images_2.to(device)

        # Forward pass
        z1 = model(images_1)
        z2 = model(images_2)

        # Compute contrastive loss
        loss = nt_xent_loss(z1, z2)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save trained encoder
torch.save(model.state_dict(), "results/checkpoints/simclr_encoder.pth")
print("Model saved.")
