import argparse
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import get_dataloaders, save_label_to_idx
from utils.model_utils import train_model, save_model, generate_unique_model_name
from models.model_factory import model_factory


# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune a model on a custom dataset.')
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory of images')
parser.add_argument('--label_path', type=str, required=True, help='Path to the labels CSV file')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output')
parser.add_argument('--min_data_points', type=int, default=3, help='Minimum data points per class')
args = parser.parse_args()

# Accessing arguments
model_name = args.model
data_dir = args.data_dir
label_path = args.label_path
output_dir = args.output_dir
min_data_points = args.min_data_points

print("loading data...")
dataloaders = get_dataloaders(data_dir, label_path, batch_size=32, num_workers=4)

print("loading model...")
model = model_factory(model_name)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust as necessary

print("training model...")
train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, num_epochs=10)

print("saving model...")
unique_model_name = generate_unique_model_name(base_name=model_name)
save_model(model, output_dir, model_name=unique_model_name)

# saves an associated file with the list of human-readable class names and their associated id
save_label_to_idx(dataloaders['train'].label_to_idx, output_dir=output_dir)

