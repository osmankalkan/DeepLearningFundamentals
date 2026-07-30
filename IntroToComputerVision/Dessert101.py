import matplotlib.pyplot as plt
import torch
from jinja2.optimizer import optimize
from scipy.cluster.hierarchy import single
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
    transforms.RandomHorizontalFlip(p=0.3),
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

        print(
            f"Epoch: {epochs} | Train Loss: {train_loss:.4f} | Train Acc: %{train_acc * 100:.2f} | Test Loss: {test_loss:.4f} | Test Acc: %{test_acc * 100:.2f}")
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
optimizer = torch.optim.Adam(params=model_0.parameters(), lr= 0.0003)

model_0_results = train(model = model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=EPOCHS)

# Gerekli kütüphanenin eklendiğinden emin ol
from PIL import Image

# 1. Fotoğrafın yolunu belirle (Kendi dosya adına göre güncelle)
custom_image_path = data_path / "baklava-online.jpg"

# 2. Fotoğrafı aç
custom_image = Image.open(custom_image_path)

# 3. Test verilerine uyguladığın dönüşümleri (transform) bu fotoğrafa da uygula
# (Eğitimdeki gibi döndürme vb. yapmıyoruz, sadece boyutlandırıp tensöre çeviriyoruz)
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

custom_image_transformed = custom_image_transform(custom_image)

# 4. PyTorch modelleri [Batch, Channel, Height, Width] formatı bekler.
# Bizim resmimiz şu an [3, 64, 64]. Başına bir batch boyutu ekleyerek [1, 3, 64, 64] yapıyoruz.
custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

# 5. Modeli değerlendirme (eval) moduna al
model_0.eval()

# 6. Tahmin yap (Gereksiz gradyan hesaplamalarını kapatarak hızı artırıyoruz)
with torch.inference_mode():
    # Modeli çalıştır ve ham çıktıları (logits) al
    custom_image_pred_logits = model_0(custom_image_transformed_with_batch_size)

    # Ham çıktıları olasılıklara çevir (Softmax)
    custom_image_pred_probs = torch.softmax(custom_image_pred_logits, dim=1)

    # En yüksek olasılığa sahip sınıfın indeksini bul
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)

    # İndeksi, class_names listesindeki metin (string) karşılığına çevir
    custom_image_pred_class = class_names[custom_image_pred_label.cpu().item()]

# 7. Sonucu ekrana yazdır
print(f"\n--- YENİ FOTOĞRAF TAHMİNİ ---")
print(f"Tahmin Edilen Sınıf: {custom_image_pred_class}")
print(f"Emin Olma Oranı: %{custom_image_pred_probs.max().item() * 100:.2f}")

# 8. Fotoğrafı Matplotlib ile başlık ekleyerek göster
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(custom_image)
plt.title(f"Tahmin: {custom_image_pred_class} | Olasılık: %{custom_image_pred_probs.max().item() * 100:.2f}")
plt.axis("off")
plt.show()












