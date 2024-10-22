# Imports
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle


import torch
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

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cudnn.benchmark = True
plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# list devices available 
print("Cuda Device Count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, save_checkpoints = False, DEST='', model_name="ResNet"):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        #best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        best_model_params_path = f'{DEST}/{model_name}-best.pt'
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        model.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
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
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                model.history[f'{phase}_loss'].append(epoch_loss)
                model.history[f'{phase}_acc'].append(epoch_acc.item())

                if save_checkpoints:
                  PATH = f"{DEST}/{model_name}-{dt_string}-checkpoint{epoch}.pt"
                  save_checkpoint(model, optimizer, PATH, epoch)


                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

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

# Data augmentation and normalization for training
# Just normalization for validation
image_size = 224

#MGL17004 Dataset Mean and STD
mean =  [0.4843, 0.4843, 0.4843] # each channel being the same makes sense because images are grayscale
std = [0.0846, 0.0846, 0.0846]

# # ImageNet mean and STD - NO NOTICABLE IMPROVEMENT 
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

#TODO: flesh out 
data_transforms = {
    'train_none': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'train_basic': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
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

data_dir = '/home/birdy/meng_thesis/data/split_MGL1704_data'

##########################################################################
############## Basic     ################################################
##########################################################################

# LOAD UP MGL1704 data WITH AUGMIX
train_dataset = torchvision.datasets.ImageFolder(f"{data_dir}/train", data_transforms["train_basic"])
val_dataset = torchvision.datasets.ImageFolder(f"{data_dir}/test", data_transforms["val"])
class_names = train_dataset.classes

# LOAD UP MGL1704 data WITH AUGMIX
# image_dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms["train_basic"])
# train_dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms["train_augmix"])
# val_dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms["val"])
# class_names = train_dataset.classes
# train_idxs, test_idxs = torch.utils.data.random_split(range(len(train_dataset)), [0.8, 0.2])
# train_dataset = torch.utils.data.Subset(train_dataset, train_idxs)
# val_dataset = torch.utils.data.Subset(val_dataset, test_idxs)

BATCH_SIZE = 512
dataloaders  = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=None, shuffle=True, drop_last=True),
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)}

dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

print(f"{len(class_names)} classes {dataset_sizes}")

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

### TRAIN MODEL 

model_name = 'ResNet18_Basic'
DEST = f'/home/birdy/meng_thesis/code/master_ifcb_classifier/output/{model_name}'
os.mkdir(DEST, exist_ok=True)

num_epochs = 30
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs, save_checkpoints=True, DEST=DEST, model_name=model_name)
# save trained model to google drive
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
PATH = f"{DEST}/{model_name}-final-{dt_string}.pt"
#torch.save(model_ft.state_dict(), PATH)
save_checkpoint(model_ft, optimizer_ft, PATH, num_epochs)

# plot training and validation accuracy and loss
fig, ax = plt.subplots()
ax.plot(model_ft.history['train_loss'], label='Training Loss')
ax.plot(model_ft.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
# plot accuracy on the same axis with different scale
ax2 = ax.twinx()
ax2.plot(model_ft.history['train_acc'], color='orange', label='Training Accuracy')
ax2.plot(model_ft.history['val_acc'], color='green', label='Validation Accuracy')
plt.suptitle(f"{model_name}")
plt.savefig(f'{DEST}/train_val_plot_{dt_string}.png')
plt.show()
