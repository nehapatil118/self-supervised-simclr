import torchvision
from augmentations import get_simclr_transform
import matplotlib.pyplot as plt

# Get SimCLR augmentation pipeline
transform = get_simclr_transform()

# CIFAR-10 dataset with SimCLR augmentations
dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)

# Generate two different augmented views of the same image
img1, _ = dataset[0]
img2, _ = dataset[0]

# Plot the two views
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].imshow(img1.permute(1, 2, 0))  # Convert tensor to image format
axs[0].set_title("View 1")
axs[0].axis("off")

axs[1].imshow(img2.permute(1, 2, 0))  # Convert tensor to image format
axs[1].set_title("View 2")
axs[1].axis("off")

plt.show()
