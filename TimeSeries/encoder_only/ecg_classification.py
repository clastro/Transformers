import os
import copy
import pandas as pd
import numpy as np
import math
from glob import glob

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

# Pytorch Lightning
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Torchmetric for computing accuracy
from torchmetrics.functional import accuracy

# Plotting
import matplotlib.pyplot as plt

def load_list(filepath, filename):
    
    output = glob(filepath+filename+"*.npy")
    
    return output
    
class SPatchECG(Dataset):
    def __init__(self, root, subset=None):
        assert subset is None or subset in ["train", "val", "test"], (
            "When `subset` not None, it must take a value from " + "{'train','val', 'test'}."
        )
        self.root = root
        
        self._walker = load_list(self.root, subset)
        
    def __len__(self):
        return len(self._walker)
    
    def __getitem__(self, n:int):
        
        row = self._walker[n]
        
        data = np.load(row)
        label = row.split('.')[0][-1]
        
        return data, int(label)
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#### ATTENTION
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # Same mask applied to all h heads.

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view( -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1, 2).contiguous().view( -1, self.h * self.d_k))
        
        del query
        del key
        del value
        out = self.linears[-1](x)
        return out

## BLOCKING
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class FeedForward(nn.Module):
    "Construct a FeedForward network with one hidden layer"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    "Transformer Model"
    def __init__(self, input_size, num_classes, num_heads=8, N=6, d_ff=256, dropout=0.0):
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, input_size)
        ff = FeedForward(input_size, d_ff, dropout)
        self.encoder = Encoder(EncoderBlock(input_size, c(attn), c(ff), dropout), N)
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#### ATTENTION
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # Same mask applied to all h heads.

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view( -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1, 2).contiguous().view( -1, self.h * self.d_k))
        
        del query
        del key
        del value
        out = self.linears[-1](x)
        return out

## BLOCKING
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class FeedForward(nn.Module):
    "Construct a FeedForward network with one hidden layer"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    "Transformer Model"
    def __init__(self, input_size, num_classes, num_heads=8, N=6, d_ff=256, dropout=0.0):
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, input_size)
        ff = FeedForward(input_size, d_ff, dropout)
        self.encoder = Encoder(EncoderBlock(input_size, c(attn), c(ff), dropout), N)
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
class LitSPatch(LightningDataModule):
    def __init__(self, root, batch_size, num_workers, length=200):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = root
        self.length = length
    
    def prepare_data(self):
        self.train_dataset = SPatchECG(self.path, "train")
        self.val_dataset = SPatchECG(self.path, "val")
        self.test_dataset = SPatchECG(self.path, "test")
    
    def setup(self, stage=None):
        self.prepare_data()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False, 
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        labels = []
        heartbeats = []
        for sample in batch:
            waveform, label = sample
            if len(waveform) < self.length:
                padsize = self.length - len(waveform)
                waveform = np.pad(waveform,(0,padsize))

            labels.append(torch.tensor(label).type(torch.int64))
            heartbeats.append(torch.tensor(waveform))

        labels = torch.stack(labels)
        heartbeats = torch.stack(heartbeats)
        return heartbeats, labels

class LitTransformer(LightningModule):
    def __init__(self, input_size, num_classes, num_heads, depth, max_epochs, lr,  dropout=0.1, d_ff=256):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(input_size, num_classes, num_heads, depth, d_ff, dropout)
        self.loss = torch.nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return optimizer #[optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        wavs, labels = batch
        preds = self(wavs)
        loss = self.loss(preds, labels)
        self.log('train_loss', loss)
        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)
    
    def test_step(self, batch, batch_idx):
        wavs, labels = batch
        preds = self(wavs)
        loss = self.loss(preds, labels)
        acc = accuracy(preds, labels) * 100.
        return {"preds": preds, 'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc, on_epoch=True, prog_bar=True)
        
        
path = '/work/datanvme/BeatClass/230117/beats3/'
input_size = 500
batch_size = 64
num_workers = 1
lr = 1e-4
max_epochs = 10
num_heads = 2
depth = 6
num_classes = 6
dropout = 0.0

datamodule = LitSPatch( path, batch_size, num_workers, length=input_size)
datamodule.setup()

model = LitTransformer(input_size, num_classes, num_heads, depth, max_epochs, lr, dropout)
print(model)

save_path = "./working/"
ckpt_name = "ecg-transformer"
model_checkpoint = ModelCheckpoint(
    dirpath=os.path.join(save_path, "checkpoints"),
    filename=ckpt_name,
    save_top_k=1,
    verbose=True,
    monitor='test_acc',
    mode='max',
)

trainer = Trainer(accelerator="gpu", devices=1,
                    max_epochs=max_epochs,
                    logger=None,
                    callbacks=[model_checkpoint]
                )
trainer.fit(model, datamodule=datamodule)

# Testing
model = model.load_from_checkpoint(
    os.path.join(save_path, "checkpoints", ckpt_name+".ckpt")
)
trainer.test(model, datamodule=datamodule)
