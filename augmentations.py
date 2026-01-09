import torchvision.transforms as transforms  # Image transformations
from PIL import ImageFilter                  # For Gaussian blur
import random                                # Random number generation


# Custom Gaussian blur transform
class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma  # Range for blur strength

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])  # Pick random sigma
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))  # Apply blur


# SimCLR data augmentation pipeline
def get_simclr_transform(image_size=32):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),   # Random crop
        transforms.RandomHorizontalFlip(p=0.5),     # Horizontal flip
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8
        ),                                          # Color jitter
        transforms.RandomGrayscale(p=0.2),          # Random grayscale
        GaussianBlur(),                              # Gaussian blur
        transforms.ToTensor()                        # Convert to tensor
    ])
    return transform
