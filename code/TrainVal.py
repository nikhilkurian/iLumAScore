import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import pandas as pd
from myTransform import ElasticTransform
import dataset_sampling2
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from rce_loss import SSGCELoss, SSGCEScore
import argparse
from configure import args
import sys
import datetime

# Placeholders for data paths
train_data_path = '../data/TrainData'
val_data_path = '../data/ValData'

# Model name based on current time, script name, and provided model name
ct = datetime.datetime.now()
model_name = f"{ct}_{sys.argv[0].split('.')[0]}_{args['model_name']}"

# Hyperparameters and configurations from arguments
lr = args['lr']
weight_decay = args['weight_decay']
batch_size = args['batch_size']
device = args['device']
num_classes = args['num_classes']
loss_criterion = args['loss_criterion']
num_workers = args['num_workers']
val_interval = args['val_interval']
rec_epoch = args['record_epoch']
num_epochs = args['num_epochs']
pos_class = args['pos_class']
train_patients_txt = args['train_patients_txt']
val_patients_txt = args['val_patients_txt']
out = args['out']

# Create necessary directories for model checkpoints and CSVs
if not os.path.exists(out + f'models/{model_name}'):
    os.makedirs(out + f'models/{model_name}')
if not os.path.exists(out + f'CSVs/{model_name}'):
    os.makedirs(out + f'CSVs/{model_name}')

# Initialize SummaryWriter for TensorBoard logging
writer = SummaryWriter(out + f'runs/{model_name}')

# Define data transformations for training and validation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, hue=0.1, saturation=0.1),
    ElasticTransform(alpha=5, sigma=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device(device if torch.cuda.is_available() else "cpu")

class FullyConnected(nn.Module):
    """Fully connected layer class."""
    def __init__(self, model, num_ftrs, num_classes):
        super(FullyConnected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_3 = self.fc_4(x)
        return out_3

# Load pre-trained DenseNet121 model
model = torchvision.models.densenet121(pretrained=True)
model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
num_ftrs = model.classifier.in_features
model_final = FullyConnected(model.features, num_ftrs, num_classes)
model_final = model_final.to(device)
model_final = nn.DataParallel(model_final)
model_final.load_state_dict(torch.load('./KimiaNetPyTorchWeights.pth'))
model = model_final

sm = nn.Softmax(dim=1)
model = nn.DataParallel(model, device_ids=[0, 1]).to(device)

# Define loss criterion
if loss_criterion == 'l1': criterion = nn.L1Loss()
if loss_criterion == 'ce': criterion = nn.CrossEntropyLoss()
if loss_criterion == 'rce': criterion = SSGCELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def one_hot(arr_, num_classes):
    """Convert labels to one-hot encoding."""
    b = arr_.size()[0]
    out_ = torch.zeros(b, num_classes).to(device)
    out_[range(b), arr_.cpu().numpy()] = 1
    return out_

# Load datasets
if num_classes == 2:
    train_dataset = dataset_sampling2.hetero_train(
        path=train_data_path, 
        transforms=train_transform, 
        txt_path=train_patients_txt,
        pos_class=pos_class
    )
    val_dataset = dataset_sampling2.hetero_val(
        path=val_data_path, 
        transforms=val_transform, 
        txt_path=val_patients_txt,
        pos_class=pos_class
    )
elif num_classes == 4:
    train_dataset = dataset.hetero_train(
        path=train_data_path, 
        transforms=train_transform, 
        txt_path=train_patients_txt
    )
    val_dataset = dataset.hetero_val(
        path=val_data_path, 
        transforms=val_transform, 
        txt_path=val_patients_txt
    )

train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, pin_memory=True, 
                              shuffle=True, 
                              num_workers=num_workers)

val_dataloader = DataLoader(val_dataset, pin_memory=True, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)

df = None

# Training and validation loop
for epoch in range(1, num_epochs + 1):
    print(epoch)

    train_loss = 0.0
    train_acc = 0.0
    model.train()

    if epoch == rec_epoch:
        df = pd.DataFrame(columns=['Patch_Path', 'Loss'])

        for images, labels, patch_path in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)

            if loss_criterion == 'l1':
                loss = criterion(preds, one_hot(labels, num_classes))
            elif loss_criterion == 'rce':
                loss = criterion(preds, one_hot(labels, num_classes))
            elif loss_criterion == 'ce':
                loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, out_labels = torch.max(preds, 1)
            train_acc += torch.sum(out_labels == labels)
            optimizer.zero_grad()

            for i in range(images.size(0)):
                df = df.append({'Patch_Path': patch_path[i], 'Loss': loss.item()}, ignore_index=True)
        df = SSGCEScore(df)
        csv_file_path = os.path.join(out + f'CSVs/{model_name}', f'train_loss_epoch_{rec_epoch}.csv')
        df.to_csv(csv_file_path, index=False)

    else:
        for images, labels, patch_path in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)

            if loss_criterion == 'l1':
                loss = criterion(preds, one_hot(labels, num_classes))
            elif loss_criterion == 'rce':
                if df is not None:
                    batch_q = []
                    for path in patch_path:
                        q_value = df.loc[df['Patch_Path'] == path, 'q'].values
                        if q_value.size > 0:
                            batch_q.append(q_value[0])
                        else:
                            batch_q.append(0.7)  
                    batch_q = torch.tensor(batch_q, dtype=torch.float32, device=device)
                    loss = criterion(preds, one_hot(labels, num_classes), _q=batch_q)
                else:
                    loss = criterion(preds, one_hot(labels, num_classes), _q=0.7)
            elif loss_criterion == 'ce':
                loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, out_labels = torch.max(preds, 1)
            train_acc += torch.sum(out_labels == labels)
            optimizer.zero_grad()

    train_loss /= len(train_dataset)
    train_acc = train_acc.double() / len(train_dataset)
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)

    if epoch % val_interval == 0:
        torch.save(model.state_dict(), out + f'models/{model_name}/epoch_{epoch}')
        log_file = open(out + f'CSVs/{model_name}/{epoch}', 'w')
        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels, paths in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                if loss_criterion == 'l1':
                    loss = criterion(preds, one_hot(labels, num_classes))
                elif loss_criterion == 'ce':
                    loss = criterion(preds, labels)
                val_loss += loss.item() * images.size(0)
                _, out_labels = torch.max(preds, 1)
                val_acc += torch.sum(out_labels == labels)
                log_file.write(paths.__repr__())
                log_file.write('\n')
                log_file.write(sm(preds).__repr__())
                log_file.write('\n\n')
        val_loss /= len(val_dataset)
        val_acc = val_acc.double()
