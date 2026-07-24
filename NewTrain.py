import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from networkx.algorithms.bipartite.cluster import modes
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sympy.codegen.ast import float32
from torch.nn import BCEWithLogitsLoss
from torch.special import logit
from torchmetrics.functional import accuracy

df = pd.read_csv("penguins.csv")
#print(df.info())
#print(df.head())

df = df.drop(columns=['year'])
df = df.dropna().reset_index(drop=True)
#print(df.info())

X = df[['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']]
y = df['species']

#label encoding
#we solve this problem in y with label encoding but in X we have to usw one-hat encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)

X = get_dummies(X , columns= ['island' , 'sex'] , drop_first=True,dtype=float)
#print(X.info())

from sklearn.model_selection import train_test_split
X_train , X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

from sklearn.preprocessing import StandardScaler

# train_test_split'ten SONRA, tensor'e çevirmeden ÖNCE yap
scaler = StandardScaler()

# sadece sayısal sütunları ölçeklendir (one-hot sütunlara dokunma)
numeric_cols = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])  # fit + transform TRAIN'de
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])         # sadece transform TEST'te

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train , dtype=torch.long)
y_test = torch.tensor(y_test , dtype=torch.long)

#print(X_test.shape , X_train.shape)
#print(y_train.shape,y_test.shape)

from torch import nn

class PenguinClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
        nn.Linear(7,10),
        nn.ReLU(),
        nn.Linear(10,10),
        nn.ReLU(),
        nn.Linear(10,3)

        )

    def forward(self,x):
        return self.linear_layer_stack(x)




from torchmetrics.classification import MulticlassAccuracy

accuracy = MulticlassAccuracy(num_classes=3)
epochs = 80
model = PenguinClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = 0.01)

train_losses = []
test_losses = []
train_accs = []
test_accs = []
for epoch in range(epochs):
    accuracy.reset()

    model.train()
    logits = model(X_train) #make a guess
    loss = loss_fn(logits,y_train) #compare guess and truth

    pred = torch.softmax(logits , dim=1).argmax(dim=1)
    acc = accuracy(pred,y_train).item() * 100

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_loss = loss_fn(test_logits,y_test)
        test_pred = torch.softmax(test_logits , dim=1).argmax(dim=1)
        test_acc = accuracy(test_pred , y_test).item() * 100

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_accs.append(acc)
    test_accs.append(test_acc)

    if epoch % 15 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


from sklearn.metrics import confusion_matrix
import seaborn as sns

epochs_range = range(epochs)
"""""
cm = confusion_matrix(y_test.numpy(), test_pred.numpy())
sns.heatmap(cm, annot=True, fmt='d')
#plt.show()



# Accuracy grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Train Accuracy')
plt.plot(epochs_range, test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()
"""

import joblib # Scaler'ı kaydetmek için

# 1. Eğitilmiş PyTorch modelini kaydet
torch.save(model.state_dict(), "penguin_model.pth")

# 2. StandardScaler objesini kaydet (Uygulamada kullanıcıdan gelen veriyi ölçeklemek için şart)
joblib.dump(scaler, "scaler.pkl")

print("Model ve Scaler başarıyla kaydedildi!")