
# coding: utf-8

# # SCRIPT TO OBTAIN THE PIXEL VALUES(0-1) OF GREYSCALE IMAGES FROM AN ENTIRE FOLDER


#Importing libraries.
import skimage.io as ios
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




#Reading a random image from the folder
DOG = ios.imread(r'C:\Users\kewal\Desktop\kaggle\dog classification\train\000bec180eb18c7604dcecc8fe0dba07.jpg',as_grey=True)
DOG


#Checking out the resolution of the image
DOG.shape



#Image was found to be 375*500 
image_vector=np.empty((0,3300), int)



path = 'C:/Users/kewal/Desktop/kaggle/dog classification/train/' #Path of the folder which contains the images
import glob
import os



#for indexing columns
hd=[x for x in range(3300)] 
hds = ','.join(map(str, hd))


#iterating through the folder
for infile in glob.glob( os.path.join(path, '*.jpg') ):
        logo = ios.imread(infile,as_grey=True)  #read the image as a greyscale
        image3 =resize(logo, (50, 66))          #resizing the image 
        image_row2 = image3.reshape(1,3300)     #flatting the pixel values to a 1D array.
        image_vector=np.append(image_vector,image_row2,axis=0)  #appending the image to an array
   


#Checking the size of the final Numpy array
image_vector.shape



#Saving the array on a csv file
np.savetxt(r'C:\Users\kewal\Desktop\kaggle\dog classification\trainingData.csv',image_vector,header=hds,delimiter=",")



#Reading the csv file
df=pd.read_csv(r'C:\Users\kewal\Desktop\kaggle\dog classification\trainingData.csv')




#Double checking
df.shape



#Checking the first entry
df.head(1)





