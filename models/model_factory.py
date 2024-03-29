# model_factory.py: Factory method to initialize and return the requested model architecture 
# (ResNet, ViT, SimCLR) with pre-trained weights.

# Implementing the model_factory.py script to include a factory method for initializing and returning
# the requested model architecture with pre-trained weights. This script will support ResNet, ViT, and
# a placeholder for SimCLR, acknowledging that SimCLR is not directly available in torchvision and might require
# custom implementation or loading.

from torchvision import models

def model_factory(model_name, pretrained=True):
    """
    Factory method to initialize and return the requested model architecture with pre-trained weights.

    Args:
        model_name (str): The name of the model architecture to initialize. Supported values: 'resnet', 'vit', 'simclr'.
        pretrained (bool): If True, initializes the model with pre-trained weights. Defaults to True.

    Returns:
        torch.nn.Module: The initialized model architecture.
    """
    if model_name == 'resnet':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=pretrained)
    elif model_name == 'simclr':
        # SimCLR is not directly available in torchvision. This would typically require custom implementation
        # or loading from a different source. For the sake of this example, we'll return None and recommend
        # implementing or loading SimCLR model as needed.
        model = None  # Replace this with actual SimCLR model initialization or loading.
        raise NotImplementedError("SimCLR model loading is not implemented. Please provide a custom implementation.")
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Supported architectures: 'resnet', 'vit', 'simclr'.")

    return model

# This factory function simplifies the process of model initialization, allowing for easy switching between
# different model architectures by just specifying the model name. For the 'simclr' model, since it's not
# available directly through torchvision, you would need to add the custom logic to initialize or load the model
# with pre-trained weights, depending on your specific requirements or the availability of pre-trained SimCLR
# models in your environment.
