#Data set includes 40 people, and 10 images of them with different expressions (wearing glasses), to increase variation

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from time import time 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 