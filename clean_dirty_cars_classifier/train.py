import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--architecture', choices=['alexnet', 'densenet161',
    'googlenet', 'inception_v3', 'mnasnet1_0', 'mobilenet_v2', 'resnet18',
    'resnext50_32x4d', 'shufflenet_v2_x1_0', 'squeezenet1_0', 'vgg16',
    'wide_resnet50_2'], default='mobilenet_v2')
parser.add_argument('--method', choices=[
    'Base'], default='Base')
parser.add_argument('--K', choices=[2], type=int,default=2)
parser.add_argument('--fold', type=int, choices=range(5),default=0)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--data_path', default='data')
args = parser.parse_args()

import numpy as np
from time import time
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import mydataset, mymodels
import wandb


wandb.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tr_ds = mydataset.MyDataset('train', mydataset.aug_transforms, args.K, args.fold,args.data_path)
tr = DataLoader(tr_ds, args.batchsize, True)
ts_ds = mydataset.MyDataset('test', mydataset.val_transforms, args.K, args.fold,args.data_path)
ts = DataLoader(ts_ds, args.batchsize)

def test(val):
    model.eval()
    val_avg_acc = 0
    for X, Y in tqdm(val):
        X = X.to(device)
        Y = Y.to(device, torch.int64)
        Yhat = model(X)
        Khat = model.to_classes(model.to_proba(Yhat), 'mode')
        val_avg_acc += (Y == Khat).float().mean() / len(val)
    return val_avg_acc

def train(tr, val, epochs=args.epochs, verbose=True):
    for epoch in range(epochs):
        if verbose:
            print(f'* Epoch {epoch+1}/{args.epochs}')
        tic = time()
        model.train()
        avg_acc = 0
        avg_loss = 0
        for X, Y in tqdm(tr):
            X = X.to(device)
            Y = Y.to(device, torch.int64)
            opt.zero_grad()
            Yhat = model(X)
            loss = model.loss(Yhat, Y)
            loss.backward()
            opt.step()
            Khat = model.to_classes(model.to_proba(Yhat), 'mode')
            avg_acc += (Y == Khat).float().mean() / len(tr)
            avg_loss += loss / len(tr)
        dt = time() - tic
        out = ' - %ds - Loss: %f, Acc: %f' % (dt, avg_loss, avg_acc)
        if val:
            model.eval()
            acc_t=test(val)
            out += ', Test Acc: %f' % acc_t
        if verbose:
            print(out)
        scheduler.step(avg_loss)
        wandb.log({'train_accuracy': avg_acc, 'train_loss': avg_loss,'test_accuracy': acc_t})
        
        
def predict_proba(data):
    model.eval()
    Phat = []
    Y_true=[]
    with torch.no_grad():
        for X, Y in data:
            phat = model.to_classes(model.to_proba(model(X.to(device))),'mode')
            Phat += list(phat.cpu().numpy())
            Y_true += list(Y.cpu().numpy())
    return Phat,Y_true


model = getattr(mymodels, args.method)(args.architecture, args.K)    
model = model.to(device)

# named_layers = dict(model.named_modules())

opt = optim.Adam(model.parameters(), args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)
train(tr, ts)


#Print some relevant metrics
from sklearn.metrics import classification_report, confusion_matrix

tsm = DataLoader(ts_ds, 1)
y_pred,y_true = predict_proba(tsm)

target_names = ['clean cars', 'dirty cars']

print(classification_report(y_true, y_pred, target_names=target_names))
print('confusion matrix:\n',confusion_matrix(y_true, y_pred))

torch.save(model, 'baseline.pth' )
