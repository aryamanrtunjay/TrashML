from PIL import Image
import torch
from torch import nn
import timm
import time
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import wandb
from sklearn.metrics import RocCurveDisplay

print("Imported packages")
print("Running on GPU: {}".format(torch.cuda.is_available()))
batch_size = 16

transform = transforms.Compose([
    transforms.Resize(224, interpolation = 3),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

train_imgs = torchvision.datasets.ImageFolder(
    root = "Split_Data/Training",
    transform = transform
)

train_ds = torch.utils.data.DataLoader(train_imgs, batch_size = batch_size, shuffle = True)

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained = True)
model.head = torch.nn.Linear(in_features = 768, out_features = 6, bias = True)
model.head_dist = torch.nn.Linear(in_features = 768, out_features = 6, bias = True)
model.cuda()

cost_function = nn.CrossEntropyLoss().cuda()
gradient_descent = torch.optim.Adam(model.parameters(), lr = 1e-6, weight_decay=1e-7)

for epoch in range(50):
    for i, (batch, target) in enumerate(train_ds):
        batch = batch.cuda()
        target = target.cuda()
        out = model(batch)
        cost = cost_function(out[0], target)
        cost.backward()
        gradient_descent.step()

        softmax_function = torch.nn.Softmax(dim=1) 
        precision, recall, fscore, support = precision_recall_fscore_support(target.cpu(), torch.argmax(softmax_function(out[0].data), dim=1).cpu(),
                                                                                zero_division=0, labels=(0,1,2,3,4,5))

        if i % 50 == 0:
            print("Epoch: {} ({}/{}) - Loss: {} - F1-Score: {}".format(epoch, i, len(train_ds), cost, fscore))


print(" --- Training Complete --- ")
print(" --- Saving Model ---")

torch.save(model.state_dict(), "Model/model.pickle")
