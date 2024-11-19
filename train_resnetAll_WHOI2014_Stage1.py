# imports 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
import random

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


from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import resnet_experiment_helpers as helper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
NUM_WORKERS = 4
EPOCHS = 31
dataset_name = "WHOI2014"
DEST_ROOT = f"/nobackup/users/birdy/ResnetAll-stage1-{dataset_name}-output"

# Initialize an empty DataFrame with columns for metrics
results_df = pd.DataFrame(columns=[
    'model_name', 'loss_function', 'optimizer', 'scheduler', 
    'learning_rate','batch_size', 'per_image_accuracy', 'per_class_accuracy',
    'MACs', 'wall_time', 'history'
])

# LOAD DATA 

data_dir = f"/nobackup/projects/public/WHOI-Plankton/2014"
csv_file = f"WHOI2014_labels.csv"
train_dataset = helper.WHOIDataset(data_dir, csv_file, "train", transform=helper.data_transforms["train_basic"])
val_dataset = helper.WHOIDataset(data_dir, csv_file, "val", transform=helper.data_transforms["val"])
NUM_CLASSES = len(train_dataset.classes)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

###################################################################
########################### Resnet18 ###########################

# Load the pretrained ResNet18 model.
model = models.resnet18(weights='IMAGENET1K_V1')
# Configure the classifier layer to match the number of classes in the dataset.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = nn.DataParallel(model)
model= model.to(device)
model_name = "resnet18"
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
DEST = f"{DEST_ROOT}/{model_name}_{dt_string}"
os.makedirs(DEST)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
opt = "adam"
crit = "cross_entropy"
sched = "step"
macs = None

# TRAIN
model = helper.train_model(model, dataloaders, criterion, optimizer, scheduler,
                  num_epochs=EPOCHS, save_checkpoints = True, DEST=DEST, 
                  model_name=model_name, val_fqn=5, device="cuda")

# RECORD
learning_rate = optimizer.param_groups[0]['lr'] # TODO: might not be right 
batch_size = dataloaders['train'].batch_size
y_true, y_pred = helper.get_validation_results(model, dataloaders['val'])
# per_image_accuracy = accuracy_score(y_true.cpu().data.numpy(), y_pred.cpu().data.numpy())
per_image_accuracy = accuracy_score(torch.cat(y_true).cpu().data.numpy(), torch.cat(y_pred).cpu().data.numpy())
per_class_accuracy, _ = helper.evaluate_per_class_accuracy(model, dataloaders['val'], device)
wall_time = model.history['time_elapsed']
history = model.history
results_df = helper.record_metrics(results_df, model_name, criterion, opt, sched, learning_rate, 
              batch_size, per_image_accuracy, per_class_accuracy, macs, wall_time, history)
print(results_df)
# save dataframe 
results_df.to_csv(f"{DEST_ROOT}/ResnetAll_{dataset_name}_stage1_results.csv", index=False)
print("")

###################################################################
########################### Resnet50 ###########################

# Load the pretrained ResNet18 model.
model = models.resnet50(weights='IMAGENET1K_V1')
# Configure the classifier layer to match the number of classes in the dataset.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = nn.DataParallel(model)
model= model.to(device)
model_name = "resnet50"
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
DEST = f"{DEST_ROOT}/{model_name}_{dt_string}"
os.makedirs(DEST)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
opt = "adam"
crit = "cross_entropy"
sched = "step"
macs = None

# TRAIN
model = helper.train_model(model, dataloaders, criterion, optimizer, scheduler,
                  num_epochs=EPOCHS, save_checkpoints = True, DEST=DEST, 
                  model_name=model_name, val_fqn=5, device="cuda")

# RECORD
learning_rate = optimizer.param_groups[0]['lr'] # TODO: might not be right 
batch_size = dataloaders['train'].batch_size
y_true, y_pred = helper.get_validation_results(model, dataloaders['val'])
# per_image_accuracy = accuracy_score(y_true.cpu().data.numpy(), y_pred.cpu().data.numpy())
per_image_accuracy = accuracy_score(torch.cat(y_true).cpu().data.numpy(), torch.cat(y_pred).cpu().data.numpy())
per_class_accuracy, _ = helper.evaluate_per_class_accuracy(model, dataloaders['val'], device)
wall_time = model.history['time_elapsed']
history = model.history
results_df = helper.record_metrics(results_df, model_name, criterion, opt, sched, learning_rate, 
              batch_size, per_image_accuracy, per_class_accuracy, macs, wall_time, history)
print(results_df)
# save dataframe 
results_df.to_csv(f"{DEST_ROOT}/ResnetAll_{dataset_name}_stage1_results.csv", index=False)
print("")

###################################################################
########################### Resnet152 ###########################

# Load the pretrained ResNet18 model.
model = models.resnet152(weights='IMAGENET1K_V1')
# Configure the classifier layer to match the number of classes in the dataset.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = nn.DataParallel(model)
model= model.to(device)
model_name = "resnet152"
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
DEST = f"{DEST_ROOT}/{model_name}_{dt_string}"
os.makedirs(DEST)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
opt = "adam"
crit = "cross_entropy"
sched = "step"
macs = None

# TRAIN
model = helper.train_model(model, dataloaders, criterion, optimizer, scheduler,
                  num_epochs=EPOCHS, save_checkpoints = True, DEST=DEST, 
                  model_name=model_name, val_fqn=5, device="cuda")

# RECORD
learning_rate = optimizer.param_groups[0]['lr'] # TODO: might not be right 
batch_size = dataloaders['train'].batch_size
y_true, y_pred = helper.get_validation_results(model, dataloaders['val'])
# per_image_accuracy = accuracy_score(y_true.cpu().data.numpy(), y_pred.cpu().data.numpy())
per_image_accuracy = accuracy_score(torch.cat(y_true).cpu().data.numpy(), torch.cat(y_pred).cpu().data.numpy())
per_class_accuracy, _ = helper.evaluate_per_class_accuracy(model, dataloaders['val'], device)
wall_time = model.history['time_elapsed']
history = model.history
results_df = helper.record_metrics(results_df, model_name, criterion, opt, sched, learning_rate, 
              batch_size, per_image_accuracy, per_class_accuracy, macs, wall_time, history)
print(results_df)
# save dataframe 
results_df.to_csv(f"{DEST_ROOT}/ResnetAll_{dataset_name}_stage1_results.csv", index=False)
print("")
