# Image-Classification
An image classifier using python to predict the breed of dog from a dataset of images.
The train and test data are from the kaggle competition (https://www.kaggle.com/c/dog-breed-identification)
The project file consists of 2 python files. 
1. ImageCSV.py - is used to read in all the images from the test and train set and get the pixel values of the individual images and      flattens them. It then saves all the values into a single csv file which we will use as input.
2. DogClassification.py - uses the output csv from ImageCSV to train the MLP classifier .Using which we predict the probability of each individual dog from the the images in a test set.I got a very low accuracy for train set itself of nearly 54% . You can try fiddling with the classifier parameters to get better results. 

