import ssl
import certifi
from jinja2.optimizer import optimize
from setuptools.namespaces import flatten
from sympy.physics.vector.printing import params
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

flatten_model = nn.Flatten()
first_data = train_dataloader.dataset[0][0]
flatten_data = flatten_model(first_data)

class CIFARClassificationCNN(nn.Module):
    def __init__(self,input_shape : int ,hidden_units : int , output_shape : int ):
        super().__init__()
        self.block_1 = nn.Sequential(
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

        self.block_2 = nn.Sequential(
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
            nn.Linear(in_features=hidden_units * 8 * 8,
                      out_features=output_shape)


        )

    def forward(self,x):
        return self.classifier(self.block_2(self.block_1(x)))


#torch.manual_seed(42)

model_2 = CIFARClassificationCNN(input_shape=3,
                                 hidden_units=32,
                                 output_shape=len(class_names))


def calculate_accuracy(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))  * 100

    return acc

#model_2 = torch.compile(model_2)
loss_fn = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(params=model_2.parameters(),lr=0.001)


epochs = 10
for epoch in range(epochs):
    train_loss = 0

    for batch , (X,y) in enumerate(train_dataloader):
        model_2.train()

        y_pred = model_2(X)
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

    model_2.eval()
    with torch.inference_mode():
        for X, y  in test_dataloader:
            test_pred = model_2(X)
            test_loss += loss_fn(test_pred,y)
            test_acc += calculate_accuracy(y_true=y,y_pred=test_pred.argmax(dim=1))


        test_loss /=len(test_dataloader)
        test_acc /= len(test_dataloader)


    print(f"Train loss: {loss} ,test loss: {test_loss} , test accuracy : {test_acc}")



def make_predictions(model: torch.nn.Module, data: list):
    """
    data: [img_tensor, img_tensor, ...]  # her biri [C, H, W]
    return: [N, num_classes] olasılık tensörü
    """
    pred_probs = []
    model.eval()

    with torch.inference_mode():
        for sample in data:
            # [C, H, W] -> [1, C, H, W]
            sample = sample.unsqueeze(0)

            # Logits al
            pred_logit = model(sample)    # shape: [1, num_classes]

            # Softmax ile olasılığa çevir
            prob = torch.softmax(pred_logit, dim=1)  # [1, num_classes]

            # Batch boyutunu sıkıştır
            pred_probs.append(prob.squeeze(0))       # [num_classes]

    # Hepsini birleştir → [N, num_classes]
    return torch.stack(pred_probs)


import random


def show_random_predictions(model, dataset, class_names, n=9):
    model.eval()

    plt.figure(figsize=(4, 4))

    # random 9 index seç
    indices = random.sample(range(len(dataset)), n)

    with torch.inference_mode():
        for i, idx in enumerate(indices):
            img, true_label = dataset[idx]

            # modele uygun hale getir
            img_input = img.unsqueeze(0)
            logits = model(img_input)
            pred_label = logits.argmax(dim=1).item()

            # görseli çizmek için permute
            img_show = img.permute(1, 2, 0)

            # doğru mu yanlış mı?
            correct = (pred_label == true_label)
            color = "green" if correct else "red"

            # subplot
            plt.subplot(3, 3, i + 1)
            plt.imshow(img_show)
            plt.axis("off")

            plt.title(
                f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}",
                color=color,
                fontsize=10
            )

    plt.tight_layout()
    plt.show()



show_random_predictions(model_2, test_data, class_names)

#if you want to increase accuracy, try with 0.001 lr rather than 0.01
#you can obviously try to change hyperparameters like stride, kernel, padding and see if that works out as well