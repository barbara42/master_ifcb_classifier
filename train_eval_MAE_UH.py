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
from einops import repeat, rearrange

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# import the MAE code from local source 
mae_src = "/home/birdy/meng_thesis/code/MAE"
sys.path.append(mae_src)
from model import MAE_ViT
from model import ViT_Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
NUM_WORKERS = 4
#NUM_EPOCHS = 31
# LEARNING_RATE = 0.001
mask_ratio = 0.75
base_learning_rate = 1.5e-4
weight_decay = 0.05
warmup_epoch = 5
total_epoch = 300

dataset_name = "UH"
model_name = f"MAE_{dataset_name}_{total_epoch}"
DEST_ROOT = f"/nobackup/users/birdy/{model_name}-{dataset_name}"
os.makedirs(DEST_ROOT, exist_ok=True)


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
mae_model_path = f"{DEST_ROOT}/{model_name}_model.pt"

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
    torch.save(model, mae_model_path)

#######################################################
######################## EVAL ########################
#######################################################

log_dir = f"{DEST_ROOT}/logs/vit_cls_ft"
writer = SummaryWriter(log_dir)

base_learning_rate = 1e-3
weight_decay = 0.05
warmup_epoch = 5
total_epoch = 50

model_name = f"vit-cls-{dataset_name}-from-mae_{total_epoch}"
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

# TRAIN
best_val_acc = 0
optim.zero_grad()
for e in range(total_epoch):
    model.train()
    losses = []
    acces = []
    for img, label in tqdm(iter(train_dataloader)):
        img = img.to(device)
        label = label.to(device)
        logits = model(img)
        loss = loss_fn(logits, label)
        acc = acc_fn(logits, label)
        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        acces.append(acc.item())
    lr_scheduler.step()
    avg_train_loss = sum(losses) / len(losses)
    avg_train_acc = sum(acces) / len(acces)
    print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

    model.eval()
    with torch.no_grad():
        losses = []
        acces = []
        for img, label in tqdm(iter(val_dataloader)):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            losses.append(loss.item())
            acces.append(acc.item())
        avg_val_loss = sum(losses) / len(losses)
        avg_val_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
        torch.save(model, vit_output_model_path)

    writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
    writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)