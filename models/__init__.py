# models/__init__.py

# Import the model_factory function to make it directly accessible
# from the models package namespace.
from .model_factory import model_factory

# Optionally, you can define a list of supported models here, which can be
# used for validation or documentation purposes elsewhere in your package.
SUPPORTED_MODELS = ['resnet', 'vit', 'simclr']

# You might also include any initialization code needed for the models
# subpackage, such as configuring logging, setting up environment variables,
# or other setup tasks specific to model management.

# Example of making the SUPPORTED_MODELS list accessible:
def get_supported_models():
    """
    Returns a list of model architectures supported by the package.

    Returns:
        list: A list of strings, each representing a supported model architecture.
    """
    return SUPPORTED_MODELS
