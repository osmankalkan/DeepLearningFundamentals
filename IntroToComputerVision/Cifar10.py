import ssl
import certifi
from jinja2.optimizer import optimize
from setuptools.namespaces import flatten
from torch.nn.functional import linear

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# image normalization  (yorum satırına alındı)
class_names = train_data.classes

image, label = train_data[1]
image = image.permute(1, 2, 0)
plt.figure(figsize=(1.2, 1.2))
plt.imshow(image)
plt.title(class_names[label])

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2470, 0.2435, 0.2616])
])

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
    target_transform=None
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

from torch.utils.data import DataLoader

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
test_dataloader = DataLoader(test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

#len(train_dataloader), len(test_dataloader)

flatten_model = nn.Flatten()
first_data = train_dataloader.dataset[0][0]
flatten_data = flatten_model(first_data)

#print(first_data.shape) #torch.Size([3, 32, 32])
#print(flatten_data.shape) #torch.Size([3, 1024])

#linear_model = nn.Linear(in_features=32,out_features=32)

class CIFAR10Classifier(nn.Module):
    def __init__(self,input_shape: int ,hidden_units : int , output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)

        )

    def forward(self,x):
        return self.layer_stack(x)


#print(first_data.shape) #3072 features
#print(class_names) #10 class


torch.manual_seed(42)
model_0 = CIFAR10Classifier(
    input_shape=3072,
    hidden_units=32,
    output_shape=10
)

model_0 = torch.compile(model_0)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters() , lr=0.01)

def calculate_accuracy(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))  * 100

    return acc
""""
epochs = 10
for epoch in range(epochs):
    train_loss = 0

    for batch , (X,y) in enumerate(train_dataloader):
        model_0.train()

        y_pred = model_0(X)
        loss = loss_fn(y_pred,y)
        train_loss+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0 :
            print(f"batch number : {batch} ")

    train_loss /= len(train_dataloader)
    test_loss = 0
    test_acc =0

    model_0.eval()
    with torch.inference_mode():
        for X, y  in test_dataloader:
            test_pred = model_0(X)
            test_loss += loss_fn(test_pred,y)
            test_acc += calculate_accuracy(y_true=y,y_pred=test_pred.argmax(dim=1))


        test_loss /=len(test_dataloader)
        test_acc /= len(test_dataloader)


    #print(f"Train loss: {loss} ,test loss: {test_loss} , test accuracy : {test_acc}")


"""""

class CIFAR10ClassifierNonLinear(nn.Module):
    def __init__(self,input_shape: int ,hidden_units : int , output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_shape)

        )

    def forward(self,x):
        return self.layer_stack(x)


torch.manual_seed(42)
model_1 = CIFAR10ClassifierNonLinear(
    input_shape=3072,
    hidden_units=32,
    output_shape=10
)

model_1 = torch.compile(model_1)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters() , lr=0.01)


epochs = 10
for epoch in range(epochs):
    train_loss = 0

    for batch , (X,y) in enumerate(train_dataloader):
        model_1.train()

        y_pred = model_0(X)
        loss = loss_fn(y_pred,y)
        train_loss+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0 :
            print(f"batch number : {batch} ")

    train_loss /= len(train_dataloader)
    test_loss = 0
    test_acc =0

    model_1.eval()
    with torch.inference_mode():
        for X, y  in test_dataloader:
            test_pred = model_1(X)
            test_loss += loss_fn(test_pred,y)
            test_acc += calculate_accuracy(y_true=y,y_pred=test_pred.argmax(dim=1))


        test_loss /=len(test_dataloader)
        test_acc /= len(test_dataloader)


    print(f"Train loss: {loss} ,test loss: {test_loss} , test accuracy : {test_acc}")
