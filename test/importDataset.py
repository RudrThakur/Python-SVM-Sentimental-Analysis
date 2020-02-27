# importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
  
# reading csv file and extracting class column to y. 
x = pd.read_csv("C:\\Users\\Rudr\\Documents\\GitHub\\Python-ML-Review-Based-Sentimental-Analysis\\test\\datasets\\breast_cancer_dataset.csv") 
a = np.array(x) 
y  = a[:,1] # classes having 0 and 1 
  
# extracting two features 
x = np.column_stack((x.clump_thickness,x.uniformity_of_cell_size)) 
x.shape # 569 samples and 2 features 

print (x),(y) 