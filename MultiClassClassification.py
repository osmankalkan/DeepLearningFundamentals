
import  torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jinja2.optimizer import optimize
from scipy.special import logit
from torchmetrics.functional import accuracy


df = pd.read_csv("09-iris.csv")
#print(df.info())
#print(df['Species'].value_counts())
#sns.scatterplot(x=df['PetalLengthCm'], y=df['PetalWidthCm'],hue=df['Species'])
#plt.show()
#print(df.head())

X = df[['SepalLengthCm' ,'SepalWidthCm' , 'PetalLengthCm','PetalWidthCm']].values
y = df['Species'].values
#print(X)
#print(y)

#label encoding
from  sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 ,random_state=42 ,stratify= y)
#print(y_test)

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)

y_train = torch.tensor(y_train,dtype=torch.long)
y_test = torch.tensor(y_test,dtype=torch.long)

#print(X_train.shape,X_test.shape)
#print(y_train.shape,y_test.shape)

from torch import nn

""""
#First way
class IrisClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=4 ,out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=3)

        self.relu = nn.ReLU()

    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
"""

#second way
class IrisClassification(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(4,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,3)

        )

    def forward(self,x):
        return self.linear_layer_stack(x)


model = IrisClassification()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr= 0.01)
""""
def calculate_accuracy(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


y_logits = model(X_test)
y_pred_probs = torch.softmax(y_logits ,dim=1 )

epochs = 175

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits,y_train)
    pred = torch.softmax(logits , dim=1).argmax(dim=1)
    acc = calculate_accuracy(y_train,pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    train_accuracies.append(acc)

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_loss = loss_fn(test_logits,y_test)
        test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_acc = calculate_accuracy(y_test,test_pred)

    test_losses.append(test_loss.item())
    test_accuracies.append(test_acc)

    if epoch % 20 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")



plt.plot(train_losses , label="Train Loss")
plt.plot(test_losses , label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
#plt.show()
"""

#with torchmetrics

from torchmetrics.classification import MulticlassAccuracy
accuracy = MulticlassAccuracy(num_classes=3)
epochs = 200
model2 = IrisClassification()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters() , lr=0.01)

for epoch in range (epochs):
    model2.train()
    logits = model2(X_train)
    loss = loss_fn(logits,y_train)

    pred = torch.softmax(logits,dim=1).argmax(dim=1)
    acc = accuracy(pred,y_train).item() * 100

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model2.eval()
    with torch.inference_mode():
        test_logits = model2(X_test)
        test_loss = loss_fn(test_logits,y_test)
        test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_acc = accuracy(test_pred, y_test).item() * 100

    if epoch % 15 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")













