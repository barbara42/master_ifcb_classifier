# model_utils.py: Utilities to save and load models, and get model architecture
# based on the --model argument.

# The model_utils.py script will include functions to save and load PyTorch models and to retrieve the model architecture
# based on the --model argument. This will support a dynamic selection of models for training, fine-tuning, or prediction.

import torch
from torchvision import models
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


# def save_model(model, path, model_name, class_names):
#     """
#     Saves a PyTorch model to the specified path.

#     Args:
#         model (torch.nn.Module): The model to save.
#         path (str): Directory where the model will be saved.
#         model_name (str): The name of the model for saving.
#         class_names (list): List of class names associated with the model's outputs.
#     """
#     if not os.path.isdir(path):
#         os.makedirs(path)
#     model_path = os.path.join(path, f"{model_name}.pth")
#     torch.save(model.state_dict(), model_path)
#     class_names_path = os.path.join(path, f"{model_name}_class_names.txt")
#     with open(class_names_path, 'w') as f:
#         for class_name in class_names:
#             f.write(f"{class_name}\n")
    
def save_model(model, output_dir, model_name="model"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_weights.pth"))
    # Optionally save the whole model (not recommended for portability)
    torch.save(model, os.path.join(output_dir, f"{model_name}_full.pth"))

def load_model(path, model_name, model_architecture):
    """
    Loads a PyTorch model from the specified path.

    Args:
        path (str): Directory where the model is saved.
        model_name (str): The name of the model to load.
        model_architecture (str): The architecture of the model to load.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model = get_model_architecture(model_architecture)
    model_path = os.path.join(path, f"{model_name}.pth")
    model.load_state_dict(torch.load(model_path))
    return model

def get_model_architecture(model_name):
    """
    Retrieves a model architecture based on the model name.

    Args:
        model_name (str): The name of the model architecture to retrieve.

    Returns:
        torch.nn.Module: The model architecture.
    """
    if model_name == 'resnet':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)
    elif model_name == 'simclr':
        # Placeholder for SimCLR or any custom model initialization
        # SimCLR is not directly available in torchvision, so this would
        # typically require a custom implementation or loading from a different source.
        model = None  # Replace with actual model initialization
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    return model

# Note: The actual model architectures and their initialization may vary based on the available models in torchvision
# or any custom models defined elsewhere. The save and load functions assume the models are saved and loaded
# using PyTorch's state_dict format, which is a common practice for PyTorch models.

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, output_dir="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("training model")
    model.train()
    train_loader = dataloaders["train"]
    # TODO: put in loading bar 
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    val_epochs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        train_accs.append(accuracy)
        train_losses.append(running_loss/len(train_loader))
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}')

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_epochs.append(epoch)
            print("Validating...")
            val_acc, val_loss = validate_model(model, dataloaders["val"], criterion)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), train_accs, label="training accuracy", c="green")
    ax.plot(val_epochs, val_accs, label="validation accuracy", c="red")
    plt.savefig(output_dir + "/training_acc.png")

    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), train_losses, label="training loss", c="green")
    ax.plot(val_epochs, val_losses, label="validation loss", c="red")
    plt.savefig(output_dir + "/training_loss.png")
    
def validate_model(model, val_loader, criterion):
    # returns accuracy, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%')
    return accuracy, val_loss/len(val_loader)

def generate_unique_model_name(base_name="model"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{base_name}_{current_time}"
    return unique_name
