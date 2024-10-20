# Goal:
# CONTINUING TRAINING with a checkpoint from a previous training loop 

# this script trains three ViT-12L-3H models from scratch
# first with ImageNet1k Data 
# then with WHOI Plankton Data 
# finally with MGL1704 Data 
#
# Output for each dataset:
# The highest performing model weights will be saved 
# A plot of the training and validation loss and accuracy over epochs
# A csv of the verbose training output, including time, accuracy 

import torch
import torch.nn as nn
from typing import Tuple, Union, Optional, List
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import tqdm
import torch.optim as optim
from utils.model_utils import save_model
import torch.cuda.amp as amp
from datetime import datetime
import os
import argparse

# add commandline params to specify which model checkpoint to load 
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model on a dataset from an existing checkpoint")
parser.add_argument("--model_save_dir", type=str, required=True, help="path to the folder containing the checkpoint to load")
parser.add_argument("--model_save_name", type=str, required=True, help="name of checkpoint file")
args = parser.parse_args()

#### ViT Architecture #####
class AttentionHead(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        # dim: the dimension of the input
        # n_hidden: the dimension of the keys, queries, and values

        super().__init__()

        self.W_K = nn.Linear(dim, n_hidden) # W_K weight matrix
        self.W_Q = nn.Linear(dim, n_hidden) # W_Q weight matrix
        self.W_V = nn.Linear(dim, n_hidden) # W_V weight matrix
        self.n_hidden = n_hidden

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x                the inputs. shape: (B x T x dim)
        # attn_mask        an attention mask. If None, ignore. If not None, then mask[b, i, j]
        #                  contains 1 if (in batch b) token i should attend on token j and 0
        #                  otherwise. shape: (B x T x T)
        #
        # Outputs:
        # attn_output      the output of performing self-attention on x. shape: (Batch x Num_tokens x n_hidden)
        # alpha            the attention weights (after softmax). shape: (B x T x T)
        #

        out, alpha = None, None
        # TODO: Compute self attention on x.
        #       (1) First project x to the query Q, key K, value V.
        #       (2) Then compute the attention weights alpha as:
        #                  alpha = softmax(QK^T/sqrt(n_hidden))
        #           Make sure to take into account attn_mask such that token i does not attend on token
        #           j if attn_mask[b, i, j] == 0. (Hint, in such a case, what value should you set the weight
        #           to before the softmax so that after the softmax the value is 0?)
        #       (3) The output is a linear combination of the values (weighted by the alphas):
        #                  out = alpha V
        #       (4) return the output and the alpha after the softmax

        # ======= Answer START ========
        # First project x to the query Q, key K, value V.
        Q = torch.matmul(x, self.W_Q.weight.t())
        K = torch.matmul(x, self.W_K.weight.t())
        V = torch.matmul(x, self.W_V.weight.t())

        # Then compute the attention weights alpha as alpha = softmax(QK^T/sqrt(n_hidden))
        temp = torch.matmul(Q,K.transpose(1,2))/np.sqrt(self.n_hidden)

        # take into account attn_mask such that token i does not attend on token j if attn_mask[b, i, j] == 0.
        if attn_mask != None:
          temp = temp.masked_fill(attn_mask==0, float("-inf"))

        alpha = torch.softmax(temp, -1)

        # The output is a linear combination of the values (weighted by the alphas):
        attn_output = torch.matmul(alpha, V)
        # ======= Answer  END ========

        return attn_output, alpha

class MultiHeadedAttention(nn.Module):
    def __init__(self, dim: int, n_hidden: int, num_heads: int):
        # dim: the dimension of the input
        # n_hidden: the hidden dimenstion for the attention layer
        # num_heads: the number of attention heads
        super().__init__()

        # TODO: set up your parameters for multi-head attention. You should initialize
        #       num_heads attention heads (see nn.ModuleList) as well as a linear layer
        #       that projects the concatenated outputs of each head into dim
        #       (what size should this linear layer be?)
        # ======= Answer START ========
        self.heads = nn.ModuleList([AttentionHead(dim, n_hidden=n_hidden)]*num_heads)
        self.W_O = nn.Linear(n_hidden*num_heads, dim)
        self.n_hidden = n_hidden
        # ======= Answer  END ========

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x                the inputs. shape: (B x T x dim)
        # attn_mask        an attention mask. If None, ignore. If not None, then mask[b, i, j]
        #                  contains 1 if (in batch b) token i should attend on token j and 0
        #                  otherwise. shape: (B x T x T)
        #
        # Outputs:
        # attn_output      the output of performing multi-headed self-attention on x.
        #                  shape: (B x T x dim)
        # attn_alphas      the attention weights of each of the attention heads.
        #                  shape: (B x Num_heads x T x T)

        attn_output, attn_alphas = None, None

        # TODO: Compute multi-headed attention. Loop through each of your attention heads
        #       and collect the outputs. Concatenate them together along the hidden dimension,
        #       and then project them back into the output dimension (dim). Return both
        #       the final attention outputs as well as the alphas from each head.
        # ======= Answer START ========
        outputs = []
        alphas = []

        for head in self.heads:
          attn_output, alpha = head.forward(x, attn_mask)
          outputs.append(attn_output)
          alphas.append(alpha)

        outputs = torch.cat(outputs, 2)
        attn_alphas = torch.stack(alphas, 1)
        attn_output = torch.matmul(outputs, self.W_O.weight.t())
        # ======= Answer END ==========
        return attn_output, attn_alphas

# these are already implemented for you!

class FFN(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        # dim       the dimension of the input
        # n_hidden  the width of the linear layer

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, dim),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # x         the input. shape: (B x T x dim)

        # Outputs:
        # out       the output of the feed-forward network: (B x T x dim)
        return self.net(x)

class AttentionResidual(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int):
        # dim       the dimension of the input
        # attn_dim  the hidden dimension of the attention layer
        # mlp_dim   the hidden layer of the FFN
        # num_heads the number of heads in the attention layer
        super().__init__()
        self.attn = MultiHeadedAttention(dim, attn_dim, num_heads)
        self.ffn = FFN(dim, mlp_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x                the inputs. shape: (B x T x dim)
        # attn_mask        an attention mask. If None, ignore. If not None, then mask[b, i, j]
        #                  contains 1 if (in batch b) token i should attend on token j and 0
        #                  otherwise. shape: (B x T x T)
        #
        # Outputs:
        # attn_output      shape: (B x T x dim)
        # attn_alphas      the attention weights of each of the attention heads.
        #                  shape: (B x Num_heads x T x T)

        attn_out, alphas = self.attn(x=x, attn_mask=attn_mask)
        x = attn_out + x
        x = self.ffn(x) + x
        return x, alphas

class Transformer(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int):
        # dim       the dimension of the input
        # attn_dim  the hidden dimension of the attention layer
        # mlp_dim   the hidden layer of the FFN
        # num_heads the number of heads in the attention layer
        # num_layers the number of attention layers.
        super().__init__()

        # TODO: set up the parameters for the transformer!
        #       You should set up num_layers of AttentionResiduals
        #       nn.ModuleList will be helpful here.
        # ======= Answer START ========
        self.net = nn.ModuleList([AttentionResidual(dim, attn_dim, mlp_dim, num_heads)]*num_layers)
        # ======= Answer END ==========

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, return_attn=False)-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x                the inputs. shape: (B x T x dim)
        # attn_mask        an attention mask. Pass this to each of the AttentionResidual layers!
        #                  shape: (B x T x T)
        #
        # Outputs:
        # attn_output      shape: (B x T x dim)
        # attn_alphas      If return_attn is False, return None. Otherwise return the attention weights
        #                  of each of each of the attention heads for each of the layers.
        #                  shape: (B x Num_layers x Num_heads x T x T)

        output, collected_attns = None, None

        # TODO: Implement the transformer forward pass! Pass the input successively through each of the
        # AttentionResidual layers. If return_attn is True, collect the alphas along the way.

        # ======= Answer START ========
        if return_attn:
          collected_attns = []
        for layer in self.net:
          x, alphas = layer.forward(x, attn_mask)
          if return_attn:
            collected_attns.append(alphas)
        output = x
        if return_attn:
          collected_attns = torch.stack(collected_attns, 1)
        # ======= Answer END ==========
        return output, collected_attns
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size: int, patch_size: int, nin: int, nout: int):
        # img_size       the width and height of the image. you can assume that
        #                the images will be square
        # patch_size     the width of each square patch. You can assume that
        #                img_size is divisible by patch_size
        # nin            the number of input channels - color channels?
        # nout           the number of output channels - ? embed size?

        super().__init__()
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.num_patches = (img_size // patch_size)**2

        # TODO Set up parameters for the Patch Embedding
        # ======= Answer START ========
        self.patch_size = patch_size
        self.nout = nout
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels = nin, out_channels = nout, kernel_size=patch_size, stride=patch_size),
        )
        # ======= Answer END ==========

    def forward(self, x: torch.Tensor):
        # x        the input image. shape: (B, nin, Height, Width)
        #
        # Output
        # out      the patch embeddings for the input. shape: (B, num_patches, nout)


        # TODO: Implement the patch embedding. You want to split up the image into
        # square patches of the given patch size. Then each patch_size x patch_size
        # square should be linearly projected into an embedding of size nout.
        #
        # Hint: Take a look at nn.Conv2d. How can this be used to perform the
        #       patch embedding?
        out = None

        # ======= Answer START ========
        out = self.projection(x)
        # Rearrange('b e (h) (w) -> b (h w) e')
        out = out.reshape([x.shape[0], self.num_patches, self.nout])
        # ======= Answer END ==========

        return out
    
class VisionTransformer(nn.Module):
    def __init__(self, n_channels: int, nout: int, img_size: int, patch_size: int, dim: int, attn_dim: int,
                 mlp_dim: int, num_heads: int, num_layers: int):
        # n_channels       number of input image channels
        # nout             desired output dimension
        # img_size         width of the square image
        # patch_size       width of the square patch
        # dim              embedding dimension
        # attn_dim         the hidden dimension of the attention layer
        # mlp_dim          the hidden layer dimension of the FFN
        # num_heads        the number of heads in the attention layer
        # num_layers       the number of attention layers.
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, nin=n_channels, nout=dim)
        self.pos_E = nn.Embedding((img_size//patch_size)**2, dim) # positional embedding matrix

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # learned class embedding
        self.transformer = Transformer(
            dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=num_heads, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, nout)
        )

    def forward(self, img: torch.Tensor, return_attn=False) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # img          the input image. shape: (B, nin, img_size, img_size)
        # return_attn  whether to return the attention alphas
        #
        # Outputs
        # out          the output of the vision transformer. shape: (B, nout)
        # alphas       the attention weights for all heads and layers. None if return_attn is False, otherwise
        #              shape: (B, num_layers, num_heads, num_patches + 1, num_patches + 1)

        # generate embeddings
        embs = self.patch_embed(img) # patch embedding
        B, T, _ = embs.shape
        pos_ids = torch.arange(T).expand(B, -1).to(embs.device)
        embs += self.pos_E(pos_ids) # positional embedding

        cls_token = self.cls_token.expand(len(embs), -1, -1)
        x = torch.cat([cls_token, embs], dim=1)

        x, alphas = self.transformer(x, attn_mask=None, return_attn=return_attn)
        out = self.head(x)[:, 0]
        return out, alphas


# a utility for calculating running average
class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val: float, sz: float):
        self.num += val*sz
        self.tot += sz

    def calculate(self) -> float:
        return self.num/self.tot

def evaluate_cifar_model(model, criterion, val_loader):
    is_train = model.training
    model.eval()
    with torch.no_grad():
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        for img, labels in val_loader:
            img, labels = img.cuda(), labels.cuda()
            outputs, _ = model(img)
            loss_meter.update(criterion(outputs, labels).item(), len(img))
            acc = (outputs.argmax(-1) == labels).float().mean().item()
            acc_meter.update(acc, len(img))
    model.train(is_train)
    return loss_meter.calculate(), acc_meter.calculate()

data_root_directories = ["dir/ImageNet1k", "dir/WHOI", "dir/MGL1704"]

#### dataset loading ######

# MEAN = [0.4914, 0.4822, 0.4465]
# STD = [0.2470, 0.2435, 0.2616]

# source: https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

image_size = 224

# no augmentations 
train_transform = torchvision.transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        #transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),

    ]
)

valid_and_test_transform = transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),

    ]
)

# ImageNet1k dataset 
train_dataset = torchvision.datasets.ImageFolder("/nobackup/projects/public/ImageNet/ILSVRC2012/train", transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder("/nobackup/projects/public/ImageNet/ILSVRC2012/val", transform=valid_and_test_transform)

# # testing with CIFAR10 dataset
# train_dataset = torchvision.datasets.CIFAR10(train=True, root='data', transform=train_transform, download=True)
# val_dataset = torchvision.datasets.CIFAR10(train=False, root='data', transform=valid_and_test_transform)

batch_size = 128
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# set up the model and optimizer
# ViT-Ti layers:12, width:192, MLP:768, Heads:3 
## ViT 16
# layers = 12
# width = 192
# mlp = 768
# heads = 3

# Small ViT
layers = 6
width = 128
mlp = 123
heads = 3
model = VisionTransformer(n_channels=3, nout=1000, img_size=image_size, patch_size=8,
                          dim=width, attn_dim=64, mlp_dim=mlp, num_heads=heads, num_layers=layers).cuda()

criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 50
lr = 0.0001
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
# # uncomment for mixed precision training 
# scaler = amp.GradScaler()


# load up checkpoint from old training loop 
model_save_dir = args.model_save_dir
model_save_name = args.model_save_name
PATH = f"{model_save_dir}/{model_save_name}.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
last_epoch = checkpoint['epoch']

# create scheduler - DELETE AFTER 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - last_epoch)
train_epochs = []
train_losses = []
train_accs = []
val_epochs = []
val_losses = []
val_accs = []
best_train_acc = 0

# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# last_epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# train_epochs = checkpoint['train_epochs']
# train_losses = checkpoint['train_losses']
# train_accs = checkpoint['train_accs']
# val_epochs = checkpoint['val_epochs']
# val_losses = checkpoint['val_losses']
# val_accs = checkpoint['val_accs']
# best_train_acc = checkpoint['best_train_acc']

#### training loop ########
model.train()
for epoch in range(last_epoch, NUM_EPOCHS): 
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for img, labels in tqdm.tqdm(train_dataloader):
        img, labels = img.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs, _ = model(img)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), len(img))
        acc = (outputs.argmax(-1) == labels).float().mean().item()
        acc_meter.update(acc, len(img))
        loss.backward()
        optimizer.step()

        ## uncomment for mixed precision training 
        # with amp.autocast(dtype=torch.float16):
        #     outputs, _ = model(img)
        #     loss = criterion(outputs, labels)
        #     if loss == None:
        #         print("output size", output.shape)
        #         print("label size", labels.shape)
        #     #assert(loss != None)
        #     loss_meter.update(loss.item(), len(img))
        #     acc = (outputs.argmax(-1) == labels).float().mean().item()
        #     acc_meter.update(acc, len(img))
        
        ## uncomment for mixed precision training 
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

    scheduler.step()
    train_loss = loss_meter.calculate()
    train_acc = acc_meter.calculate()
    print(f"Train Epoch: {epoch}, Loss: {train_loss}, Acc: {train_acc}")
    train_epochs.append(epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    # checkpointing to continue training when jobs time out
    PATH = os.path.join(model_save_dir, model_save_name + '.pt') 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'train_epochs': train_epochs,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_epochs': val_epochs,
            'val_losses': val_losses,
            'val_accs': val_accs, 
            'best_train_acc': best_train_acc
            }, PATH)
    
    if train_acc > best_train_acc:
        # save the current model and weights 
        save_model(model, model_save_dir, model_name=model_save_name)
    # validation
    with torch.no_grad():
        if epoch % 10 == 0:
            val_loss, val_acc = evaluate_cifar_model(model, criterion, val_dataloader)
            val_epochs.append(epoch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Val Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}")

val_loss, val_acc = evaluate_cifar_model(model, criterion, val_dataloader)
print(f"Val Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}")
val_epochs.append(epoch)
val_losses.append(val_loss)
val_accs.append(val_acc)
print('Finished Training')

# create and save plot of accuracy and losses 
# Plotting the training and validation loss and accuracy
plt.figure(figsize=(14, 6))

# Plotting training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_epochs, train_losses, label='Training Loss')
plt.plot(val_epochs, val_losses, label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_epochs, train_accs, label='Training Accuracy')
plt.plot(val_epochs, val_accs, label='Validation Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('T`raining and Validation Accuracy')
plt.legend()
plt.suptitle(f"{model_save_name} training metrics")
plt.tight_layout()
plt.savefig(f"{model_save_dir}/{model_save_name}_training_plots.png")
