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


from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# import the MAE code from local source 
mae_src = "/home/birdy/meng_thesis/code/MAE"
sys.path.append(mae_src)
from model import MAE_ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
NUM_WORKERS = 2
# NUM_EPOCHS = 31
# LEARNING_RATE = 0.001
dataset_name = "UH"
model_name = "MAE"
DEST_ROOT = f"/nobackup/users/birdy/{model_name}-{dataset_name}"

mask_ratio = 0.75
base_learning_rate = 1.5e-4
weight_decay = 0.05
warmup_epoch = 5
total_epoch = 31

dataset_name = "UH"
data_dir = "/home/birdy/meng_thesis/data/split_MGL1704_data"
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

# logs stored in the writer
log_dir = f"{DEST_ROOT}/logs/mae_pretrain"
writer = SummaryWriter(log_dir)

# create model
model = MAE_ViT(mask_ratio=mask_ratio, image_size=224, patch_size=16).to(device)
model_path = f"{DEST_ROOT}/{model_name}_{dataset_name}_model.pt"

optim = torch.optim.AdamW(model.parameters(), lr=base_learning_rate * BATCH_SIZE / 256, betas=(0.9, 0.95), weight_decay=weight_decay)
lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

step_count = 0
optim.zero_grad()
mae_loss = []
for e in range(total_epoch):
    model.train()
    losses = []
    for img, label in tqdm(iter(dataloaders["train"])):
        step_count += 1
        img = img.to(device)
        predicted_img, mask = model(img)
        loss = torch.mean((predicted_img - img) ** 2 * mask) / mask_ratio
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
    lr_scheduler.step()
    avg_loss = sum(losses) / len(losses)
    writer.add_scalar('mae_loss', avg_loss, global_step=e)
    print(f'In epoch {e}, average traning loss is {avg_loss}.')

    ''' visualize the first 16 predicted images on val dataset'''
    model.eval()
    with torch.no_grad():
        val_img = torch.stack([val_dataset[i][0] for i in range(16)])
        val_img = val_img.to(device)
        predicted_val_img, mask = model(val_img)
        predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        writer.add_image('mae_image', (img + 1) / 2, global_step=e)

    ''' save model '''
    torch.save(model, model_path)