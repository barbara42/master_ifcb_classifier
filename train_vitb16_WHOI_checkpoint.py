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
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

cudnn.benchmark = True
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add commandline params to specify which model checkpoint to load 
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model on a dataset from an existing checkpoint")
parser.add_argument("--model_save_dir", type=str, required=True, help="path to the folder containing the checkpoint to load")
parser.add_argument("--model_save_name", type=str, required=True, help="name of checkpoint file")
args = parser.parse_args()

BATCH_SIZE = 128 #512
NUM_WORKERS = 4
NUM_EPOCHS = 31
LEARNING_RATE = 0.001
dataset_name = "WHOI2014"
model_name = "vitb16"
DEST_ROOT = f"/nobackup/users/birdy/{model_name}-{dataset_name}"

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

# Initialize an empty DataFrame with columns for metrics
results_df = pd.DataFrame(columns=[
    'model_name', 'loss_function', 'optimizer', 'scheduler',
    'learning_rate','batch_size', 'per_image_accuracy', 'per_class_accuracy',
    'MACs', 'wall_time', 'history'
])

model = models.vit_b_16(weights='IMAGENET1K_V1')
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES) 
model = nn.DataParallel(model)
model= model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.03)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
opt = "adamw"
sched = "CosineAnnealingLR"
crit = "CrossEntropyLoss"

# load up checkpoint from old training loop 

model_save_dir = args.model_save_dir
model_save_name = args.model_save_name
load_path = f"{model_save_dir}/{model_save_name}.pt"
model, optimizer, last_epoch = helper.load_checkpoint(model, optimizer, load_path)

# measure macs 
macs = helper.calculate_vit_macs(model, device="cuda")
print(f"Number of MACs: {macs}")

DEST = f"{DEST_ROOT}/{model_name}"
os.makedirs(DEST, exist_ok=True)

model = helper.train_model_from_checkpoint(model, dataloaders, criterion, optimizer, scheduler,
                    num_epochs=NUM_EPOCHS, start_epoch = last_epoch, save_checkpoints = True, DEST=DEST,
                    model_name=model_name, val_fqn=5, device="cuda")

# RECORD
learning_rate = LEARNING_RATE
batch_size = dataloaders['train'].batch_size
y_true, y_pred = helper.get_validation_results(model, dataloaders['val'])
per_image_accuracy = accuracy_score(torch.cat(y_true).cpu().data.numpy(), torch.cat(y_pred).cpu().data.numpy())
per_class_accuracy, _ = helper.evaluate_per_class_accuracy(model, dataloaders['val'], device)
wall_time = model.history['time_elapsed']
history = model.history
results_df = helper.record_metrics(results_df, model_name, crit, opt, sched, learning_rate,
              batch_size, per_image_accuracy, per_class_accuracy, macs, wall_time, history)
results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_results.csv", index=False)
print(results_df)
print("")

# save df
results_df.to_csv(f"{DEST_ROOT}/{dataset_name}_results.csv", index=False)
