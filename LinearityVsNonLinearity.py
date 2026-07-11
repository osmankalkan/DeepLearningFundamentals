import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch import nn

df = pd.read_csv("08-email_classification_svm.csv")
#print(df.info())
#print(df["email_type"].unique())

#sns.scatterplot(x=df['subject_formality_score'], y=df['sender_relationship_score'],hue=df['email_type'])
#plt.show()

X=df[['sender_relationship_score','subject_formality_score']].values
y=df['email_type'].values
#print(y)
#print(X)

X_train ,X_test ,y_train , y_test =train_test_split(X,y,test_size=0.2,random_state=42)
#print(len(X_train))

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)

y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)#new output : torch.Size([800, 2]) torch.Size([800, 1])
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1) #Unsqueeze operator, prevent shape error below th line

#print(X_train.shape,y_train.shape)#output : torch.Size([800, 2]) torch.Size([800])
#print(X_test.shape,y_test.shape)#output : torch.Size([200, 2]) torch.Size([200]) These y sides will be raise error

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2 , out_features=5)
        self.layer_2 = nn.Linear(in_features=5 , out_features=5)
        self.layer_3 = nn.Linear(in_features=5 , out_features=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


model_0 = ClassificationModel()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params= model_0.parameters() , lr =0.01) #lr = learning rate


#Logic of Logit and Sigmoid

def calculate_accuracy(y_test, y_pred):
    correct = torch.eq(y_test, y_pred).sum().item()
    accuracy = (correct / len(y_pred)) * 100
    return accuracy

# Logit = model output before sigmoid
# Sigmoid = transforms logits into probabilities [0,1]


y_logits = model_0(X_test)[:5]
y_pred_probs = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_probs)



#print(calculate_accuracy(y_test[:5],y_preds[:5]))

#Model Training

torch.manual_seed(42)
epochs = 120

for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(X_train) # 'y_logits' from outside the loop, resulting in a shape mismatch error.
    y_pred = torch.round(torch.sigmoid(y_logits)) # Convert logits to probabilities using sigmoid, then round them to get discrete predictions (0 or 1)

    loss = loss_fn(y_logits, y_train) #We use y_logits instead of y_pred because we are using BCEWithLogitsLoss
    acc = calculate_accuracy(y_test=y_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        # Forward pass on test data
        test_logits = model_0(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits))
        # Calculate test loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = calculate_accuracy(y_test=y_test, y_pred=test_pred)

        if epoch % 5 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")












