"""
Dataloader for Pytorch model
 
outputs car image (X) with label (Y)
"""
from torch.utils.data import Dataset
from torchvision import transforms
import pickle


class MyDataset(Dataset):
    def __init__(self, type, transform, K, fold,path):
        self.X, self.Y = pickle.load(open(f'{path}/K{K}.pickle', 'rb'))[fold][type]
        self.transform = transform
        self.path=path
    def __getitem__(self, i):
        X = self.X[i]
        X = self.transform(X)
        Y = self.Y[i]
        return X, Y

    def __len__(self):
        return len(self.X)

aug_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(180, (0, 0.1), (0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(saturation=(0.5, 2.0)),
    transforms.ToTensor(),  # vgg normalization
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),  # vgg normalization
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


