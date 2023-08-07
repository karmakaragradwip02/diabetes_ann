#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch
import numpy as np
import pandas as pd


# In[27]:


dataset = pd.read_csv('diabetes.csv')
dataset.head()


# In[28]:


X = dataset.iloc[:,:-1].values #X=dataset.drop('DEPENDENT ARIABLE').values
y = dataset.iloc[:,-1].values  #y=dataset('DEPENDENT ARIABLE').values


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state=0)


# In[30]:


import torch.nn as nn
import torch.nn.functional as F


# In[31]:


# CREATING TENSORS
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# In[32]:


# CREATING MODEL WITH PYTORCH

class myAnn(nn.Module):
    def __init__(self, 
                 input_features=8, #no. of colums in X 
                 hidden1=20, #no. of attributes in 1st hidden
                 hidden2=20, #no. of attributes in 2nd hidden
                 out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x


# In[33]:


# CREATING AN INSTANCE FOR ANN MODEL
torch.manual_seed(20)
model = myAnn()


# In[34]:


model.parameters


# In[35]:


#BACKWARD PROPAGATION --DEFINE loss_function, optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[36]:


epochs = 3000
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred,y_train)
    final_losses.append(loss)
    if(1%10==1):
        print("epochs: {}, loss: {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[37]:


# plot the loss function
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


tensor_loss = torch.tensor(final_losses)
plt.plot(range(epochs), tensor_loss)
#plt.plot(range(epochs), final_losses.grad)
plt.ylabel('loss')
plt.xlabel('epoch')


# In[39]:


# PREDICTING IN X_TEST DATA
predictions = []
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())


# In[40]:


#confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, predictions)
cm


# In[41]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('actual values')
plt.ylabel('predicted values')


# In[42]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predictions)
score


# In[43]:


torch.save(model, 'diabetes.pt')


# In[44]:


torch.load('diabetes.pt')


# In[49]:


list(dataset.iloc[0,:-1])


# In[50]:


lst1 = [7.0, 194.0, 52.0, 40.0, 2.0, 45.6, 0.332, 45.0]


# In[53]:


new_data = torch.tensor(lst1)


# In[59]:


with torch.no_grad():
    print(model(new_data))
    print(model(new_data).argmax().item())


# In[ ]:




