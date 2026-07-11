import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import scatter
from seaborn import scatterplot
from sklearn.model_selection import train_test_split
from sympy.physics.vector.printing import params
from sympy.printing.pytorch import torch
from torch import nn, dtype
from torch.xpu import manual_seed
import numpy as np






df = pd.read_csv("08-seismic_activity_svm.csv")

sns.scatterplot(x=df['underground_wave_energy'],y=df['vibration_axis_variation'],hue=df['seismic_event_detected'])

plt.show()

X=df[['underground_wave_energy','vibration_axis_variation']].values
y=df['seismic_event_detected'].values

X_train , X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 =  nn.Linear(in_features=2 , out_features=4)
        self.layer_2 =  nn.Linear(in_features=4 , out_features=4)
        self.layer_3 =  nn.Linear(in_features=4 , out_features=1)

    def forward(self,x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        return x


model_0 = ClassificationModel() # Creation copy of model
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)


def CalculateAccuracy(y_test,y_pred):
    correct = torch.eq(y_test,y_pred).sum().item()
    accuracy = correct / len(y_pred) * 100
    return accuracy


y_logits = model_0(X_test)
y_pred_probs = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_probs)

torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(X_train)
    y_preds = torch.sigmoid(y_logits)

    loss = loss_fn(y_logits,y_train)
    acc = CalculateAccuracy(y_test=y_train, y_pred=y_preds) # We sent y_train cuz this loop is a train loop y_test just a label

    optimizer.zero_grad() #Zeroiation Position
    loss.backward() #Find the response of errors
    optimizer.step() #Preapare the next step

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = CalculateAccuracy(y_test=y_test, y_pred=test_pred)

    if epoch % 5 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")



def plot_decision_boundary(model, X, y):
    # Modelin CPU'da olduğundan emin ol
    model.eval()

    # 1. Çizim alanını (meshgrid) belirle
    # Verinin minimum ve maksimum değerlerine biraz boşluk (padding) bırakıyoruz
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # 2. Grid üzerindeki her noktayı modele sok
    with torch.inference_mode():
        # Veriyi düzleştirip (ravel) modelin anlayacağı formata getiriyoruz
        X_to_pred = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
        y_logits = model(X_to_pred)
        y_pred = torch.round(torch.sigmoid(y_logits))

    # 3. Çizimi yap
    y_pred = y_pred.reshape(xx.shape)  # Grid formuna geri dönüştür
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title("Modelin Karar Sınırı")
    plt.show()


#plot_decision_boundary(model_0, X_test, y_test)













