
# coding: utf-8

# In[1]:

import skimage.io as ios
from skimage.transform import resize


# In[16]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[3]:

DOG = ios.imread(r'C:\Users\kewal\Desktop\kaggle\dog classification\train\000bec180eb18c7604dcecc8fe0dba07.jpg',as_grey=True)
DOG


# In[4]:

DOG.shape


# In[12]:

image_vector=np.empty((0,3300), int)


# In[13]:

path = 'C:/Users/kewal/Desktop/kaggle/dog classification/train/'
import glob
import os


# In[14]:

for infile in glob.glob( os.path.join(path, '*.jpg') ):
    
        logo = ios.imread(infile,as_grey=True)
        image3 =resize(logo, (50, 66))
        image_row2 = image3.reshape(1,3300)
        image_vector=np.append(image_vector,image_row2,axis=0)


# In[15]:

np.savetxt(r'C:\Users\kewal\Desktop\kaggle\dog classification\trainData.csv',image_vector)


# In[17]:

df=pd.read_csv(r'C:\Users\kewal\Desktop\kaggle\dog classification\trainData.csv')


# In[18]:

df.shape


# In[20]:

df.head(1)


# In[ ]:



