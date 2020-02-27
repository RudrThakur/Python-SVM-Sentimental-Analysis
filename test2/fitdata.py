# import support vector classifier 
from sklearn.svm import SVC # "Support Vector Classifier" 
# importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# reading csv file and extracting class column to y. 
x = pd.read_csv('C:\\Users\\Rudr\\Documents\\GitHub\\Python-ML-Review-Based-Sentimental-Analysis\\test2\\datasets\\train.tsv', sep="\t") 
a = np.array(x) 
print(a)
y  = a[:,3] # classes having 0 and 1 
print(y)
x = np.column_stack((x.Phrase)) 
x.shape # 569 samples and 2 features 
clf = SVC(kernel='linear') 
clf.fit(x,y)
