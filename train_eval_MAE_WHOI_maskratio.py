# imports
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import sys
import math
import os 
from utils import resnet_experiment_helpers as helper
from utils import mae_experiment_helpers as mae_helper
from einops import repeat, rearrange

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# import the MAE code from local source 
mae_src = "/home/birdy/meng_thesis/code/MAE"
sys.path.append(mae_src)
from model import MAE_ViT
from model import ViT_Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512 #256
NUM_WORKERS = 4
#NUM_EPOCHS = 31
# LEARNING_RATE = 0.001
mask_ratio = 0.75
base_learning_rate = 1.5e-4
weight_decay = 0.05
warmup_epoch = 5
total_epoch = 30

dataset_name = "WHOI"
model_name = f"MAE_{dataset_name}_maskratios2"
DEST_ROOT = f"/nobackup/users/birdy/{model_name}"
os.makedirs(DEST_ROOT, exist_ok=True)



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


mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
for mask_ratio in mask_ratios.reverse():
    # logs stored in the writer
    log_dir = f"{DEST_ROOT}/logs/mae_pretrain_maskratio{mask_ratio}"
    writer = SummaryWriter(log_dir)

    base_learning_rate = 1.5e-4
    weight_decay = 0.05
    warmup_epoch = 5
    total_epoch = 30

    # create model
    model_name = f"MAE_{dataset_name}_maskratio_{mask_ratio}"
    model = MAE_ViT(mask_ratio=mask_ratio, image_size=224, patch_size=16).to(device)
    mae_model_path = f"{DEST_ROOT}/{model_name}_model_{mask_ratio}.pt"

    optim = torch.optim.AdamW(model.parameters(), lr=base_learning_rate * BATCH_SIZE / 256, betas=(0.9, 0.95), weight_decay=weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    # train MAE 
    mae_helper.train_mae(model, mask_ratio, total_epoch, dataloaders, 
                        device, optim, lr_scheduler, writer, mae_model_path)

    # evaluate encoder
    log_dir = f"{DEST_ROOT}/logs/vit_cls_ft_{mask_ratio}"
    writer = SummaryWriter(log_dir)

    base_learning_rate = 1e-3
    weight_decay = 0.05
    warmup_epoch = 5
    total_epoch = 10

    model_name = f"vit-cls-{dataset_name}-from-mae_{mask_ratio}"
    vit_output_model_path = f"{DEST_ROOT}/{model_name}.pt"

    # load up trained MAE model 
    pretrained_model_path = mae_model_path
    model = torch.load(pretrained_model_path, map_location='cpu')
    model = ViT_Classifier(model.encoder, num_classes=NUM_CLASSES).to(device)

    # set up optimizer, scheduler, loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
    optim = torch.optim.AdamW(model.parameters(), lr=base_learning_rate * BATCH_SIZE / 256, betas=(0.9, 0.999), weight_decay=weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    mae_helper.train_classifier(model, total_epoch, dataloaders, device, optim,
                                lr_scheduler, loss_fn, acc_fn, writer, 
                                vit_output_model_path) 

