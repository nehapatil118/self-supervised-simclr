import torchvision
import torchvision.transforms as transforms

# Basic transform to convert image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 training dataset
dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Print total number of images
print("Total images:", len(dataset))

# Get one sample image and label
image, label = dataset[0]
print("Image shape:", image.shape)  # Tensor shape (C, H, W)
print("Label:", label)              # Class label
