import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from time import time 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 

# Reading Dataset and visualizing it 
df = pd.read_csv("face_data.csv")
#print(df.head())

labels = df["target"]
pixel = df.drop(["target"], axis=1)

def show_original_images(pixel):
    # Displaying Original Images
    fig, axes = plt.subplots(6,10,figsize=(11,7),
                             subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixel)[i].reshape(64,64), cmap='gray')
    plt.show()

#show_original_images(pixel)

# Split Dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(pixel, labels)
# x_train and y_train will have features while x_test and y_test will have targets 

# Perform PCA
pca = PCA(n_components=200).fit(x_train)

# plotting amount of variance by each component
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

def show_eigenfaces(pca):
	#Displaying Eigenfaces
	fig, axes = plt.subplots(3, 8, figsize=(9, 4),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
	    ax.set_title("PC " + str(i+1))
	plt.show()
     
show_eigenfaces(pca)
#Give us a sense about which direction we have the maximum variation of data