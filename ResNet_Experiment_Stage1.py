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
# plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import in-house methods 

# Load the pretrained ResNet18 model.

# Configure the classifier layer to match the number of classes in the dataset.

# Set Up Loss Functions:

# Set Up Optimizers:

# Set Up Learning Rate Schedulers: