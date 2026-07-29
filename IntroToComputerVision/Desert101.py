import torch
from jinja2.optimizer import optimize
from sympy.physics.vector.printing import params
from torch import nn
import numpy as np

from pathlib import Path

from torch.utils.tensorboard.summary import image

data_path = Path("data/")
image_path = data_path / "desert101"

import os
def check_data(dir_path):
    for dirpath , dirnames , filenames in os.walk(dir_path):
        print(f" # of directories : {len(dirnames)} and {len(filenames)} image in {dirpath} ")

#print(check_data(image_path))

train_dir = image_path / "train"
test_dir = image_path / "test"

from PIL import Image
import random


random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))

random_image = random.choice(image_path_list)
img = Image.open(random_image)
#img.show()

from torch.utils.data import DataLoader
from torchvision import datasets ,transforms

data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

train_data =datasets.ImageFolder(root=train_dir,
                     transform=data_transform,
                     target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform,
                                 target_transform=None)


#print(train_data.classes)
class_names = train_data.classes

BATCH_SIZE=32
NUM_WORKERS = 0

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)

class DesertClassifier(nn.Module):
    def __init__(self,input_shape: int , hidden_units : int , output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)


        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)


        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units *16 *16,
                      out_features=output_shape)

        )


    def forward(self,x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))



model_0= DesertClassifier(input_shape=3,
                          hidden_units=32,
                          output_shape=len(class_names))


from torchinfo import summary

#print(summary(model_0,input_size=[1,3,64,64]))

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module , optimizer : torch.optim.Optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch , (X,y) in enumerate(dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)


    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return  train_loss,train_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module ):
    model.eval()

    test_loss = 0
    test_acc  = 0

    with torch.inference_mode():
        for batch , (X,y) in enumerate(dataloader):
            test_pred_logits = model(X)
            loss =  loss_fn(test_pred_logits,y)
            test_loss +=loss.item()


            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return  test_loss ,test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs : int = 10):

    results = {"train_loss" : [],
               "train_acc" : [],
               "test_loss" : [],
               "test_acc" : []
               }

    for epochs in range(epochs):
        train_loss ,train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                           dataloader=test_dataloader,
                                           loss_fn=loss_fn
                                           )

        print(f"Epoch: {epochs} , Train Loss: {train_loss} , Train Acc: {train_acc}, Test Loss: {test_loss}, Test Acc : {test_acc}")
        results["train_loss"].append(train_loss.item()) if isinstance(train_loss,torch.Tensor) else train_loss
        results["train_acc"].append(train_acc.item()) if isinstance(train_acc,torch.Tensor) else train_acc
        results["test_loss"].append(test_loss.item())if isinstance(test_loss,torch.Tensor) else test_loss
        results["test_acc"].append(test_acc.item()) if isinstance(test_acc,torch.Tensor) else test_acc

    return results


EPOCHS = 10
model_0 = DesertClassifier(input_shape=3,
                           hidden_units=32,
                           output_shape=len(class_names))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr= 0.001)

model_0_results = train(model = model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=EPOCHS)








