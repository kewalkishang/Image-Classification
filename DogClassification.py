
# coding: utf-8

# # Image Classification using MLP to classify different dog breeds.

# In[1]:

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import glob


# In[2]:

#importing input data
trainset = pd.read_csv(r'C:\Users\kewal\Desktop\kaggle\dog classification\trainingData.csv')
trainlabel=pd.read_csv(r'C:\Users\kewal\Desktop\kaggle\dog classification\labels.csv')


# In[3]:

#Little data analysis
trainset.head(1)


# In[5]:

trainset.isnull().any().unique()


# In[7]:

trainlabel.shape


# In[8]:

trainset.shape


# In[15]:

#Checking if labels.csv has ordering same as our traingData.csv
count=0
ImageName=[]
path = 'C:/Users/kewal/Desktop/kaggle/dog classification/train/'


# In[16]:

for infile in glob.glob( os.path.join(path, '*.jpg') ):
    count+=1
    ImageName.append(infile)


# In[17]:

count


# In[18]:

ImageName


# In[14]:

trainlabel


# In[19]:

#label encoding the different dog breeds
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[20]:

y_train=trainlabel.iloc[:,-1]


# In[21]:

y_train


# In[22]:

y_train= labelencoder_X.fit_transform(y_train)


# In[23]:

y_train


# In[25]:

np.unique(y_train)


# In[26]:

#Training the data with a MLPClassifier
from sklearn.neural_network import MLPClassifier


# In[27]:

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=500)


# In[28]:

mlp.fit(trainset,y_train)


# In[46]:

predictions = mlp.predict(trainset)


# In[47]:

#To check the accuracy of our prediction on the trainingset itself
from sklearn.metrics import classification_report,accuracy_score


# In[48]:

accuracy_score(y_train,predictions)   #accuracy achieved is 0.54 (damn low)


# In[41]:

#reading the test data
testset = pd.read_csv(r'C:\Users\kewal\Desktop\kaggle\dog classification\testData.csv')


# In[37]:

#predicting probability on the test data
Pred = mlp.predict_proba(testset)


# In[43]:

Pred


# In[49]:

Pred.shape


# In[81]:

#to get just names of the images in the test folder
Imatest=[]
path=r'C:\Users\kewal\Desktop\kaggle\dog classification\test'


# In[82]:

os.chdir(path)
for file in glob.glob("*.jpg"):
    Imatest.append(os.path.splitext(os.path.basename(file))[0])


# In[83]:

Imatest


# In[80]:

type(Pred)


# In[84]:

#saving the predictions onto a csv file 
dataf=pd.DataFrame(Pred ,index=Imatest,columns=labely)


# In[85]:

dataf

