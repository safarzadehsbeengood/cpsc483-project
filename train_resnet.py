import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="COVID-19 Detection from X-ray Images")
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="input batch size for training (default: 16)",
)
parser.add_argument(
    "--epochs", type=int, default=100, help="number of epochs to train (default: 100)"
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="number of workers to train (default: 0)"
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.0001,
    help="learning rate (default: 0.0001)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="./data_upload_v3/",
    help="training and validation dataset",
)
parser.add_argument(
    "--decay", type=float, default=0.0001, help="Adam optimizer weight_decay param"
)

args = parser.parse_args()

print(args)

IMG_SIZE = 224

transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    ),
}

splits = ["train", "val"]

BATCH_SIZE = args.batch_size
DATA_DIR = args.dataset_path

img_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transforms[x]) for x in splits
}

print("datasets created...")

dataloaders = {
    x: DataLoader(img_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in splits
}
print("dataloaders created...")

dataset_sizes = {x: len(img_datasets[x]) for x in splits}

class_names = img_datasets["train"].classes

# model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

print("model set up...")

################################### Training ############################################


def train(model, criterion, optimizer, scheduler, batch_size, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc, val_acc = [], []

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            with tqdm(total=dataset_sizes[phase], desc=f"Epoch {epoch} ({phase})") as pbar:
                running_loss = 0.0
                running_corrects = 0.0
                running_prec = 0.0
                running_rec = 0.0
                running_f1 = 0.0

                curr_batch_idx = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    curr_acc = torch.sum(preds == labels.data).float() / batch_size
                    curr_batch_idx += 1

                    if phase == "train":
                        train_acc.append(curr_acc)
                    else:
                        val_acc.append(curr_acc)
                    pbar.update(batch_size)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                pbar.set_postfix_str(
                    f"phase {phase} - loss: {epoch_loss} acc: {epoch_acc}"
                )

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.fc.parameters(), lr=args.learning_rate, weight_decay=args.decay
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

if __name__ == "__main__":
    model, train_acc, val_acc = train(
        model, criterion, optimizer, scheduler, args.batch_size, args.epochs
    )
    model.eval()
    torch.save(model, "./resnet.pt")
