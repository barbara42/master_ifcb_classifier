# transforms.py: Image transformations and augmentations using torchvision.
from torchvision import transforms

# ResNet Transformations
# For ResNet, images are typically resized to 256x256 pixels, then center-cropped to 224x224 pixels to match the input
# size the network expects. Additionally, we normalize the images with mean and standard deviation values that were used
# during the pretraining on ImageNet. These steps are standard preprocessing for models trained on ImageNet.
resnet_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ViT Transformations
# Vision Transformers (ViT) require input images to be a fixed size (e.g., 224x224). Unlike CNNs, ViT doesn't require
# the images to be normalized with specific mean and standard deviation, but normalization is still applied for
# consistency. The key difference is the addition of RandomResizedCrop and RandomHorizontalFlip for data augmentation,
# which helps ViT models to learn more robust features from varied perspectives and scales.
vit_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# SimCLR Transformations
# SimCLR uses a series of aggressive data augmentations to learn robust representations. These include random cropping
# and resizing, random color distortions, and random Gaussian blur. These augmentations force the model to learn
# invariant and robust features from the images. The color jitter and Gaussian blur are particularly important for
# SimCLR's contrastive learning approach, helping it to distinguish between different features under various conditions.
simclr_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # Randomly change the brightness, contrast, saturation, and hue
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=224//20*2+1, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# These sets of transformations are tailored to the characteristics of each model architecture and the expected input
# data format. They play a crucial role in data preprocessing and augmentation, significantly impacting the model's
# learning efficiency and performance.

# Note: The actual application of these transformations would be in the data loading process, specifically when
# initializing the dataset instances for training, validation, and testing.

