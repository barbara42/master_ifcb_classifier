# ViT helper
#imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import RandAugment


# MixUp functions 
class Mixup:
    """
    Implements Mixup for data augmentation.
    Args:
        alpha (float): Mixup interpolation parameter.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Applies Mixup augmentation.
        Args:
            x (Tensor): Batch of input images.
            y (Tensor): Batch of labels.
        Returns:
            Tuple[Tensor, Tensor]: Mixed inputs and labels.
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        return mixed_x, mixed_y

def mixup_collate_fn(batch, mixup_fn, num_classes):
    """
    Custom collate function for applying Mixup and one-hot encoding.
    Args:
        batch (list): List of tuples (image, label).
        mixup_fn (Mixup): Mixup augmentation instance.
        num_classes (int): Number of classes in the dataset.
    Returns:
        Tuple[Tensor, Tensor]: Augmented inputs and one-hot encoded labels.
    """
    inputs, labels = zip(*batch)

    # Stack inputs and convert labels to one-hot
    inputs = torch.stack(inputs, dim=0)
    labels = torch.tensor(labels)
    labels = F.one_hot(labels, num_classes=num_classes).float()

    # Apply Mixup
    return mixup_fn(inputs, labels)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
width, height = 224, 224 # for PadToSize
data_transforms = {
    'train_none': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'train_basic': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'train_augmix': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'train_randaug':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    # 'train_padToSize': transforms.Compose([
    #     PadToMaxSize(width, height),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]),
    # 'val_padToSize': transforms.Compose([
    #     PadToMaxSize(width, height),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}