from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Utilities
ce = nn.CrossEntropyLoss()


# Model
class Base(nn.Module):
    def __init__(self, pretrained_model, n_outputs):
        super().__init__()
        self.n_outputs = n_outputs
        model = getattr(models, pretrained_model)(pretrained=True)
        model = nn.Sequential(*tuple(model.children())[:-1])
        last_dimension = torch.flatten(model(torch.randn(1, 3, 224, 224))).shape[0]
        self.model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(last_dimension, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, Yhat, Y):
        return ce(Yhat, Y)

    def to_proba(self, Yhat):
        return F.softmax(Yhat, 1)

    def to_classes(self, Phat, type):
        assert type in ('mode')
        if type == 'mode':
            return Phat.argmax(1)


