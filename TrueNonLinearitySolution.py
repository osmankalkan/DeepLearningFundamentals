import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn

""""
# Data Preparation
df = pd.read_csv("08-seismic_activity_svm.csv")
X = df[['underground_wave_energy', 'vibration_axis_variation']].values
y = df['seismic_event_detected'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# Declaring Model
class ClassificationNonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


model_1 = ClassificationNonLinearModel()
loss_fnn = nn.BCEWithLogitsLoss()
optimizer1 = torch.optim.Adam(params=model_1.parameters(), lr=0.001)


def CalculateAccuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct / len(y_pred) * 100
    return accuracy


# Training Loop
torch.manual_seed(42)
epochs = 400

for epoch in range(epochs):
    model_1.train()  # Get Training Mode


    y_logits = model_1(X_train)
    y_pred = torch.round(torch.sigmoid(y_logits))


    loss = loss_fnn(y_logits, y_train)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()


    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fnn(test_logits, y_test)
        test_acc = CalculateAccuracy(y_test, test_pred)

        if epoch % 40 == 0:
            train_acc = CalculateAccuracy(y_train, y_pred)
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

"""