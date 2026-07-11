import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("08-email_classification_svm.csv")
#print(df.info())
#print(df["email_type"].unique())

#sns.scatterplot(x=df['subject_formality_score'], y=df['sender_relationship_score'],hue=df['email_type'])
#plt.show()

X=df[['sender_relationship_score','subject_formality_score']].values
y=df['email_type'].values
#print(y)
#print(X)

X_train ,X_test ,y_train , y_test =train_test_split(X,y,test_size=0.2,random_state=43)
#print(len(X_train))

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)

y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)#new output : torch.Size([800, 2]) torch.Size([800, 1])
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1) #Unsqueeze operator, prevent shape error below th line

#print(X_train.shape,y_train.shape)#output : torch.Size([800, 2]) torch.Size([800])
#print(X_test.shape,y_test.shape)#output : torch.Size([200, 2]) torch.Size([200]) These why sides will be raise error


















