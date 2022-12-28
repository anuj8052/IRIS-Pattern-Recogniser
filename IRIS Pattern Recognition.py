#!/usr/bin/env python
# coding: utf-8

# ## Classification Project Use of Logistic Regression

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Downloads/Data-Analytics-Projects-master/IRIS Pattern Recognition/data.csv")


# In[3]:


df.head()


# In[40]:


df.corr()


# In[4]:


df['Species'].unique()


# In[5]:


df.columns


# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[7]:


encode = LabelEncoder()


# In[8]:


df.Species = encode.fit_transform(df.Species)


# In[17]:


df['Species'].unique()


# In[25]:


df.head()


# In[13]:


x = df.iloc[:,:-1] ## independent features
y = df.iloc[:,-1] ## dependent features


# In[24]:


x.head()


# In[15]:


y.head()


# In[ ]:


df.drop()


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[22]:


model = LogisticRegression()


# In[26]:


model.fit(x_train, y_train)


# In[36]:


y_test


# In[37]:


y_test.shape


# In[27]:


predict = model.predict(x_test)


# In[33]:


predict


# In[30]:


encode.inverse_transform(predict)


# In[32]:


accuracy_score(y_test, predict)

