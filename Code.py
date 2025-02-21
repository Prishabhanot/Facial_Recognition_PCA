#Data set includes 40 people, and 10 images of them with different expressions (wearing glasses), to increase variation

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from time import time 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 

#Reading Dataset and visualize it 
df = pd.read_csv("face_data.csv")
#print(df.head())

labels = df["target"]
pixel = df.drop(["target"], axis=1)

def show_orginal_images(pixels):
    #Displaying Original Images
    fig, axes = plt.subplots(6,10,figsize=(11,7),
                             subplot_kw=('xticks':[], 'yticks':[])})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pizels)[i].reshape(64,64), cmap='gray')
    plt.show()

show_orginal_images(pixel)

#Step 2: Split Dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(pixel,labels)
