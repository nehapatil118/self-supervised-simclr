import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from simclr_model import SimCLR


# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained SimCLR encoder
model = SimCLR(feature_dim=128).to(device)
model.load_state_dict(
    torch.load("results/checkpoints/simclr_encoder.pth", map_location=device)
)

# Freeze encoder weights
for param in model.encoder.parameters():
    param.requires_grad = False

model.eval()  # Set encoder to evaluation mode

# Dataset with basic transform (no augmentation)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)

# CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=transform
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

# Linear classifier on top of frozen encoder
classifier = nn.Linear(128, 10).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 50

for epoch in range(epochs):
    classifier.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Extract features without updating encoder
        with torch.no_grad():
            features = model(images)

        # Forward pass through classifier
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Evaluation on test set
classifier.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = model(images)
        outputs = classifier(features)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Linear Evaluation Accuracy: {accuracy:.2f}%")
