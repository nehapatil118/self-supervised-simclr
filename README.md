## Self-Supervised Representation Learning with SimCLR (Contrastive Learning)

# Description 
This project implements SimCLR (Simple Framework for Contrastive Learning of Visual Representations) from scratch in PyTorch to learn visual representations of images without labels and evaluate their quality using linear evaluation.
Instead of using explicit class annotations, the model learns to discriminate between augmented views of the same image and other images in the batch, showing that good visual representations can be learned in a self-supervised manner.
Key Highlights :-
- The SimCLR pipeline was also trained end-to-end.
- Used contrastive NT-Xent loss.
- Trained encoder without labels.
- It uses linear probing to evaluate representations.
- Achieved 60.19% accuracy on CIFAR-10.

# Dependencies
The list all libraries, packages and other dependencies that need to be installed to run your project.
- Install python of version 3.10.x.
- Create the virtual environment.
- Install the torch, torchvision, torchaudio, matplotlib and numpy.

# Usage
- Activate the virtual environment.
- Install all the dependencies.
- Create the SimCLR model.
- Train the SimCLR model.
- Perform linear evalution.
- Check sanity of model.

# Roadmap
- Increase batch size for stronger contrastive learning.
- Replace encoder with ResNet18.
- Extend experiments to STL-10 dataset.
- Tune temperature and augmentation strength.

# Contributing
- Fork the repository.
- Create a feature branch.
- Make changes with clear documentation.
- Submit a pull request.

# License
This project is intended for academic and educational use.
Dataset usage follows CIFAR-10 licensing terms.

# Author
Neha Patil

Thank you
Neha Patil

Thank you
