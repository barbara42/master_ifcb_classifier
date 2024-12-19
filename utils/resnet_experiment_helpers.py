# Imports
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
from collections import defaultdict


import torch
import torchprofile
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import torch.nn.functional as F
import torchvision.transforms.functional as TF


from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cudnn.benchmark = True
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import os
from PIL import Image
import json
import torch

class WHOIDataset(Dataset):
    """
    A custom dataset class that loads images based on a CSV file containing image ids, labels,
    and data splits (train, val, test).
    """
    def __init__(self, data_dir, csv_file, split, transform=None):
        """
        Initializes the dataset.

        Args:
            data_dir (str): Path to the directory containing images.
            csv_file (str): Path to the CSV file containing image ids, labels, and splits.
            split (str): The split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        csv_file = pd.read_csv(csv_file)

        # Count the occurrences of each class
        label_counts = csv_file['label'].value_counts()

        # Filter out classes with less than min_data_points
        min_data_points = 3
        self.filtered_labels = label_counts[label_counts >= min_data_points].index.tolist()
        self.df = csv_file[csv_file['label'].isin(self.filtered_labels)]
        
        self.labels = self.df['label'].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Store ignored classes information
        self.ignored_classes = label_counts[label_counts < min_data_points].index.tolist()

        self.split = split
        self.transform = transform
        self.data = self.df[self.df['split'] == split]
        self.classes = pd.unique(self.data['label'])

    def num_classes(self):
        return len(self.labels)

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the class label.
        """
        img_name = os.path.join(f"{self.data_dir}/{self.data.iloc[idx]['label']}", self.data.iloc[idx]['image_name'])
        image = Image.open(img_name).convert('RGB')
        #label = self.data.iloc[idx, 1]
        label = self.label_to_idx[self.data.iloc[idx]['label']]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_history': model.history,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.history = checkpoint['model_history']
    return model, optimizer, epoch

def calculate_macs(model, input_size=(3, 224, 224), device="cuda"):
    """
    Calculate the Multiply-Accumulate Operations (MACs) for a given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_size (tuple): The input size for the model (default is (3, 224, 224) for typical image inputs).
        device (str): Device to perform computation on ("cuda" or "cpu").

    Returns:
        float: The number of MACs in millions (for easier readability).
    """
    model.to(device)  # Move model to specified device
    model.eval()  # Set model to evaluation mode

    # Create a sample input tensor with the specified input size
    sample_input = torch.randn(1, *input_size).to(device)

    # Calculate MACs using torchprofile
    with torch.no_grad():  # No gradients needed for MAC calculations
        macs = torchprofile.profile_macs(model, args=(sample_input,))

    # Convert MACs to millions for readability
    macs_in_millions = macs / 1e6

    return macs_in_millions

def calculate_vit_macs(model, device="cuda"):
    """
    Calculate the Multiply-Accumulate Operations (MACs) for a given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_size (tuple): The input size for the model (default is (3, 224, 224) for typical image inputs).
        device (str): Device to perform computation on ("cuda" or "cpu").

    Returns:
        float: The number of MACs in millions (for easier readability).
    """
    model.to(device)  # Move model to specified device
    model.eval()  # Set model to evaluation mode

    sample_input = torch.randn(1, 3, 224, 224)
    macs = torchprofile.profile_macs(model, args=(sample_input,))
    # Convert MACs to millions for readability
    macs_in_millions = macs / 1e6

    return macs_in_millions

def record_metrics(metrics_df, model_name, loss_function, optimizer, scheduler, learning_rate, 
                   batch_size, per_image_accuracy, per_class_accuracy, MACs, wall_time, history):
    """
    Append a new row of metrics to the DataFrame.

    Args:
        metrics_df (pd.DataFrame): DataFrame to store metrics.
        epoch (int): Epoch number.
        loss_function (str): Loss function name (e.g., "Cross-Entropy").
        optimizer (str): Optimizer name (e.g., "SGD").
        scheduler (str): Scheduler name (e.g., "StepLR").
        learning_rate (float): Current learning rate.
        per_image_accuracy (float): Per-image accuracy for the epoch.
        per_class_accuracy (float): Per-class accuracy for the epoch.
        composite_score (float): Composite score combining per-image and per-class accuracy.
        MACs (float): Multiply-Accumulate Operations in millions.
        wall_time (float): Wall time for the epoch in seconds.

    Returns:
        pd.DataFrame: Updated DataFrame with the new row added.
    """
    new_row = {
        'model_name': model_name,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'per_image_accuracy': per_image_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'MACs': MACs,
        'wall_time': wall_time, 
        'history': history
    }
    new_row_df = pd.DataFrame([new_row])
    metrics_df = pd.concat([metrics_df, new_row_df], ignore_index=True)
    return metrics_df

def get_opt_sched(opt, sched, model):
  if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
  if opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
  if opt == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

  if sched == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
  if sched == 'cos':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
  if sched == 'ReduceLROnPlateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
  
  return optimizer, scheduler

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, start_epoch=0, save_checkpoints = False, DEST='', model_name="ResNet", val_fqn=1, device="cpu"):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        #best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        best_model_params_path = f'{DEST}/{model_name}-best.pt'
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        # intialize training history if model doesn't already have it
        if hasattr(model, 'history') == False: 
            model.history = {
                'train_loss': [],
                'train_acc': [],
                'train_class_acc': [],
                'train_epoch_duration': [],
                'val_epochs': [],
                'val_loss': [],
                'val_acc': [],
                'val_class_acc': [],
                'val_epoch_duration': []
            }
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        for epoch in range(start_epoch, num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            epoch_start_time = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'val' and epoch % val_fqn !=0:
                    continue # only validate model every nth time
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.history['val_epochs'].append(epoch)
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_class_acc = 0.0
                batch_counter = 0
                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    batch_counter += 1
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # labels = torch.tensor(labels).to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # print(inputs.shape)
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    # print("labels.data.shape:", labels.data.shape)
                    # print("preds.shape:", preds.shape)
                    label_indices = labels.data
                    # if labels are one-hot-encoded, turn them back to indices
                    if len(label_indices.shape) > 1:
                        label_indices =  torch.argmax(labels, dim=1)
                    running_corrects += torch.sum(preds == label_indices)
                    avg_class_acc, _ = calculate_per_class_accuracy(labels.data.cpu().numpy(), preds.cpu().numpy())
                    running_class_acc += avg_class_acc
                if phase == 'train':
                    if type(scheduler) == lr_scheduler.ReduceLROnPlateau:
                        scheduler.step(running_loss)
                    else:
                      scheduler.step() 

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                epoch_class_acc = running_class_acc / batch_counter
                epoch_duration = time.time() - epoch_start_time

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                model.history[f'{phase}_loss'].append(epoch_loss)
                model.history[f'{phase}_acc'].append(epoch_acc.item())
                model.history[f'{phase}_class_acc'].append(epoch_class_acc)
                model.history[f'{phase}_epoch_duration'].append(epoch_duration)


                if save_checkpoints:
                  # write over the old checkpoint - saves mem space
                  PATH = f"{DEST}/{model_name}-{dt_string}-checkpoint.pt"
                  save_checkpoint(model, optimizer, PATH, epoch)


                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        model.history['time_elapsed'] = time_elapsed
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        # model.load_state_dict(torch.load(best_model_params_path))
    return model

def evaluate_per_class_accuracy(model, dataloader, device):
    """
    Calculate and record per-class accuracy for a given model on a provided dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on.
        device (torch.device): Device to perform computation on (e.g., "cuda" or "cpu").

    Returns:
        dict: Dictionary where keys are class indices and values are per-class accuracy.
    """
    model.eval()  # Set model to evaluation mode
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Count correct predictions and total samples for each class
            for label, pred in zip(labels, preds):
                total_per_class[label.item()] += 1
                if label == pred:
                    correct_per_class[label.item()] += 1

    # Calculate per-class accuracy
    per_class_accuracy = {
        class_idx: (correct_per_class[class_idx] / total_per_class[class_idx] if total_per_class[class_idx] > 0 else 0)
        for class_idx in total_per_class
    }
    per_class_accuracy_avg = sum(per_class_accuracy.values()) / len(per_class_accuracy)

    return per_class_accuracy_avg, (per_class_accuracy, total_per_class)


def calculate_per_class_accuracy(y_true, y_pred):
    # Initialize dictionaries to store correct counts and total counts per class
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    # Loop through each true and predicted label pair
    for true, pred in zip(y_true, y_pred):
        # if true is a tensor, 
        print("true:", true)
        print(type(true))
        print("pred:", pred)
        total_counts[true] += 1  # Increment total count for the true class
        if true == pred:
            correct_counts[true] += 1  # Increment correct count if prediction matches the true label
    
    # Calculate per-class accuracy
    per_class_accuracy = {cls: correct_counts[cls] / total_counts[cls] for cls in total_counts if total_counts[cls] > 0}
    
    # Calculate average per-class accuracy
    average_per_class_accuracy = np.mean(list(per_class_accuracy.values()))
    
    return average_per_class_accuracy, per_class_accuracy

def get_validation_results(model, dataloader):
    was_training = model.training
    model.eval()

    true_vals = []
    pred_vals = []
    #device = "cpu"
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_vals.append(labels)
            pred_vals.append(preds)
    return true_vals, pred_vals

def freeze_layers(model):
  for param in model.parameters():
    param.requires_grad = False
  
  # Unfreeze last layer
  for param in model.fc.parameters():
    param.requires_grad = True
  return model

def append_dropout(model, rate=0.2):
  for name, module in model.named_children():
      if len(list(module.children())) > 0:
          append_dropout(module)
      if isinstance(module, nn.ReLU):
          new = nn.Sequential(module, nn.Dropout2d(p=rate))
          setattr(model, name, new)

# custom transform
class PadToMaxSize:
    def __init__(self, max_width, max_height):
        self.max_width = max_width
        self.max_height = max_height

    def __call__(self, img):
        width, height = img.size
        pad_width = (self.max_width - width) // 2
        pad_height = (self.max_height - height) // 2
        padding = (pad_width, pad_height, self.max_width - width - pad_width, self.max_height - height - pad_height)
        # Compute the average color of the edge pixels
        img_np = np.array(img)
        # Get edge pixels (top, bottom, left, right)
        top_edge = img_np[0, :, :]
        bottom_edge = img_np[-1, :, :]
        left_edge = img_np[:, 0, :]
        right_edge = img_np[:, -1, :]
        # Stack all edge pixels together and calculate the mean
        edge_pixels = np.vstack([top_edge, bottom_edge, left_edge, right_edge])
        avg_color = edge_pixels.mean(axis=0).astype(int)
        avg_color_tuple = tuple(avg_color)  # Convert to tuple for padding
        return TF.pad(img, padding, fill=avg_color_tuple)

# ImageNet mean and STD - NO NOTICABLE IMPROVEMENT 
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
    'train_padToSize': transforms.Compose([
        PadToMaxSize(width, height),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_padToSize': transforms.Compose([
        PadToMaxSize(width, height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
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