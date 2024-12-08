# Imports
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
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
from utils import resnet_experiment_helpers as helper

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cudnn.benchmark = True
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
NUM_WORKERS = 4
NUM_EPOCHS = 31
#DEST_ROOT = '/home/birdy/meng_thesis/code/master_ifcb_classifier/output/ResNet18-Stage1'
dataset_name = "UH"
DEST_ROOT = f"/nobackup/users/birdy/resnet-{dataset_name}-stage2-freeze18_50"
data_dir = "/home/birdy/meng_thesis/data/split_MGL1704_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load up train and val dataloaders 
train_dataset = torchvision.datasets.ImageFolder(f"{data_dir}/train", helper.data_transforms["train_basic"])
val_dataset = torchvision.datasets.ImageFolder(f"{data_dir}/test", helper.data_transforms["val"])
class_names = train_dataset.classes
NUM_CLASSES = len(train_dataset.classes)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

# Initialize an empty DataFrame with columns for metrics
results_df = pd.DataFrame(columns=[
    'model_name', 'loss_function', 'optimizer', 'scheduler',
    'learning_rate','batch_size', 'per_image_accuracy', 'per_class_accuracy',
    'MACs', 'wall_time', 'history'
])

###########################################################################
############### Freeze Layers #############################################
###########################################################################

# ResNet18
###########################################################################

# Load the pretrained ResNet18 model.
model = models.resnet18(weights='IMAGENET1K_V1')
# Configure the classifier layer to match the number of classes in the dataset.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = helper.freeze_layers(model)
model = nn.DataParallel(model)
model= model.to(device)

# measure macs 
macs = helper.calculate_macs(model, input_size=(3, 224, 224), device="cuda")
print(f"MACs: {macs}")

# adam and step 
# adamw and cos

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

opt = "adam"
sched = "step"
crit = "cross_entropy"

# TRAIN
model_name = f"resnet18-{dataset_name}-freeze-{optim}-{sched}"
DEST = f"{DEST_ROOT}/{model_name}"
os.makedirs(DEST, exist_ok=True)
model = helper.train_model(model, dataloaders, criterion, optimizer, scheduler,
                    num_epochs=NUM_EPOCHS, save_checkpoints = True, DEST=DEST,
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
results_df = helper.record_metrics(results_df, model_name, crit, opt, sched, learning_rate,
              batch_size, per_image_accuracy, per_class_accuracy, macs, wall_time, history)
results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_stage2_freeze_results.csv", index=False)
print(results_df)
print("")

# save df
results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_stage2_freeze_results.csv", index=False)

# ResNet50
###########################################################################

# Load the pretrained ResNet50 model.
model = models.resnet50(weights='IMAGENET1K_V1')
model_name = f"resnet50-{dataset_name}-freeze"
# Configure the classifier layer to match the number of classes in the dataset.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = helper.freeze_layers(model)
model = nn.DataParallel(model)
model= model.to(device)

# measure macs 
macs = helper.calculate_macs(model, input_size=(3, 224, 224), device="cuda")
print(f"MACs: {macs}")

# adam and step 
# adamw and cos

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

opt = "adam"
sched = "step"
crit = "cross_entropy"

# TRAIN
DEST = f"{DEST_ROOT}/{model_name}"
os.makedirs(DEST, exist_ok=True)
model = helper.train_model(model, dataloaders, criterion, optimizer, scheduler,
                    num_epochs=NUM_EPOCHS, save_checkpoints = True, DEST=DEST,
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
results_df = helper.record_metrics(results_df, model_name, crit, opt, sched, learning_rate,
              batch_size, per_image_accuracy, per_class_accuracy, macs, wall_time, history)
results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_stage2_freeze_results.csv", index=False)
print(results_df)
print("")

# save df
results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_stage2_freeze_results.csv", index=False)

# # ResNet152
# ###########################################################################

# # Load the pretrained ResNet18 model.
# model = models.resnet152(weights='IMAGENET1K_V1')
# model_name = f"resnet152-{dataset_name}-freeze"
# # Configure the classifier layer to match the number of classes in the dataset.
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model = helper.freeze_layers(model)
# model = nn.DataParallel(model)
# model= model.to(device)

# # measure macs 
# macs = helper.calculate_macs(model, input_size=(3, 224, 224), device="cuda")
# print(f"MACs: {macs}")

# # adam and step 
# # adamw and cos

# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# criterion = nn.CrossEntropyLoss()

# opt = "adam"
# sched = "step"
# crit = "cross_entropy"

# # TRAIN
# DEST = f"{DEST_ROOT}/{model_name}"
# os.makedirs(DEST, exist_ok=True)
# model = helper.train_model(model, dataloaders, criterion, optimizer, scheduler,
#                     num_epochs=NUM_EPOCHS, save_checkpoints = True, DEST=DEST,
#                     model_name=model_name, val_fqn=5, device="cuda")

# # RECORD
# learning_rate = optimizer.param_groups[0]['lr'] # TODO: might not be right
# batch_size = dataloaders['train'].batch_size
# y_true, y_pred = helper.get_validation_results(model, dataloaders['val'])
# # per_image_accuracy = accuracy_score(y_true.cpu().data.numpy(), y_pred.cpu().data.numpy())
# per_image_accuracy = accuracy_score(torch.cat(y_true).cpu().data.numpy(), torch.cat(y_pred).cpu().data.numpy())
# per_class_accuracy, _ = helper.evaluate_per_class_accuracy(model, dataloaders['val'], device)
# wall_time = model.history['time_elapsed']
# history = model.history
# results_df = helper.record_metrics(results_df, model_name, crit, opt, sched, learning_rate,
#               batch_size, per_image_accuracy, per_class_accuracy, macs, wall_time, history)
# results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_stage2_freeze_results.csv", index=False)
# print(results_df)
# print("")

# # save df
# results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_stage2_freeze_results.csv", index=False)

